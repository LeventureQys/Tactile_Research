# -*- coding: utf-8 -*-
"""在现役 K9 框架内放开「速率类」参数：找漂移抑制 ↔ 小变化保留的帕累托前沿。

可调旋钮（按职责分两类）：
  预算类 r_slow_max          —— 慢态稳态上限 = r_slow_max·e
  速率类 slope_cap_frac      —— 慢态积分速率上限 = slope_cap_frac·max(e,1) /s
                               （从 0 涨到满额的等效时间 ≈ r_slow_max / slope_cap_frac）
  门控类 slope_gate_frac / slow_confirm_s / soft_unfreeze_s
  前馈类 ramp_slope_min / ramp_full_s   —— K9 缓坡前馈（把 0.5~60 ADC/s 的斜率当加载加速吸收）

指标（同一份数据、同口径）：
  漂移侧 = 四份恒载的首尾净变化 / 最大 5 s 窗口偏移，取联合最差
  保留侧 = 设备1测试2 真实背景 + 加载后 30 s 起的配对注入（上升 2.5 s、保持 15 s）
           分散形态：57 通道均分总 100 ADC ｜ 集中形态：单通道 300 ADC
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import (SESSIONS, detect_steps, load_session, med,  # noqa: E402
                                  metrics, run_k9)
from creep_observer_k9 import Params, fields                                  # noqa: E402

HOMES = ("设备1/测试1", "设备1/测试2", "设备2/测试1", "设备2/测试2")
INJECT_HOST = "设备1/测试2"
RISE_S = 2.5          # 贴近现场量出的上升时间
HOLD_S = 15.0
INJECTS = (("分散57×100", 57, 100.0), ("集中1×300", 1, 300.0))

CONFIGS: list[tuple[str, dict]] = [
    ("现役 r=0.1", dict(r_slow_max=0.1)),
    ("r=0.4", dict(r_slow_max=0.4)),
    ("r=0.6", dict(r_slow_max=0.6)),
    ("r=1.0", dict(r_slow_max=1.0)),
    ("r=0.6 cap=0.01", dict(r_slow_max=0.6, slope_cap_frac=0.01)),
    ("r=0.6 cap=0.005", dict(r_slow_max=0.6, slope_cap_frac=0.005)),
    ("r=0.6 cap=0.002", dict(r_slow_max=0.6, slope_cap_frac=0.002)),
    ("r=1.0 cap=0.005", dict(r_slow_max=1.0, slope_cap_frac=0.005)),
    ("r=1.0 cap=0.002", dict(r_slow_max=1.0, slope_cap_frac=0.002)),
    ("r=0.6 cap=0.005 gate=0.1", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                      slope_gate_frac=0.1)),
    ("r=0.6 cap=0.005 gate=0.2", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                      slope_gate_frac=0.2)),
    ("r=0.6 cap=0.005 confirm=8", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                       slow_confirm_s=8.0)),
    ("r=0.6 cap=0.005 unfreeze=8", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                        soft_unfreeze_s=8.0)),
    ("r=0.6 cap=0.005 ramp_full=60", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                          ramp_full_s=60.0)),
    ("r=0.6 cap=0.005 ramp_min=5", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                        ramp_slope_min=5.0)),
    ("r=0.6 cap=0.005 hold_eps=0", dict(r_slow_max=0.6, slope_cap_frac=0.005,
                                        hold_eps=0.0)),
]


def params_with(session: dict, **overrides) -> Params:
    raw = dict(session["algorithm"]["params"])
    names = {f.name for f in fields(Params)}
    values = {k: float(v) for k, v in raw.items() if k in names}
    for key, value in overrides.items():
        if key not in names:
            raise KeyError("不是 K9 参数：%s" % key)
        values[key] = float(value)
    return Params(**values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=1, choices=(0, 1, 2),
                        help="分片：1=前 8 个配置，2=后 8 个配置，0=全部")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    configs = CONFIGS if args.part == 0 else (
        CONFIGS[:8] if args.part == 1 else CONFIGS[8:])

    hosts = {}
    for label, device, name in SESSIONS:
        if label in HOMES or label == INJECT_HOST:
            hosts[label] = load_session(device, name)
    host = hosts[INJECT_HOST]
    t_h = host["t"]
    background = host["pre"].astype(np.float64)
    load = detect_steps(t_h, background.sum(1), 2000.0)[0]["t"]
    start = load + 30.0

    def injection(footprint: int, total_adc: float) -> np.ndarray:
        ramp = np.clip((t_h - start) / RISE_S, 0.0, 1.0)
        gate = (t_h >= start) & (t_h < start + HOLD_S)
        level = np.where(gate, ramp, 0.0)
        block = np.zeros((t_h.size, 57))
        block[:, :footprint] = (total_adc / footprint) * level[:, None]
        return block

    rows = []
    for name, override in configs:
        started = time.perf_counter()
        worst_net, worst_off, per_home = 0.0, 0.0, {}
        for label in HOMES:
            sess = hosts[label]
            t = sess["t"]
            params = params_with(sess["session"], **override)
            total = run_k9(sess["pre"], t, params).sum(1)
            st = detect_steps(t, sess["pre"].sum(1), 2000.0)
            m = metrics(t, total, st[0]["t"] if st else float(t[0]))
            per_home[label] = dict(net=round(m["net_change_adc"], 1),
                                   off=round(m["max_abs_offset_adc"], 1))
            worst_net = max(worst_net, abs(m["net_change_adc"]))
            worst_off = max(worst_off, m["max_abs_offset_adc"])

        params_h = params_with(host["session"], **override)
        base_out = run_k9(background, t_h, params_h).sum(1)
        holds = {}
        for iname, footprint, total_adc in INJECTS:
            injected = background + injection(footprint, total_adc)
            pair = run_k9(injected, t_h, params_h).sum(1) - base_out
            hold = float(np.median(pair[(t_h >= start + 10) & (t_h < start + HOLD_S)]))
            resid = float(np.median(pair[(t_h >= start + HOLD_S + 7)
                                        & (t_h < start + HOLD_S + 12)]))
            holds[iname] = dict(hold_adc=round(hold, 1),
                                retention=round(hold / total_adc, 3),
                                release_7_12=round(resid, 1))

        row = dict(name=name, override=override, worst_abs_net=round(worst_net, 1),
                   worst_abs_offset=round(worst_off, 1), per_home=per_home,
                   injects=holds, elapsed_s=round(time.perf_counter() - started, 1))
        rows.append(row)
        print("%-28s 漂移最差 %6.1f / %6.1f ｜ 分散 %5.1f(%5.1f%%) ｜ 集中 %5.1f(%5.1f%%) "
              "｜ 撤载 %+6.1f  [%.0fs]"
              % (name, row["worst_abs_net"], row["worst_abs_offset"],
                 holds["分散57×100"]["hold_adc"], 100 * holds["分散57×100"]["retention"],
                 holds["集中1×300"]["hold_adc"], 100 * holds["集中1×300"]["retention"],
                 holds["分散57×100"]["release_7_12"], row["elapsed_s"]), flush=True)

    out = Path(args.out) if args.out else HERE / ("tune_grid_part%d.json" % args.part)
    out.write_text(json.dumps({"rise_s": RISE_S, "inject_start_base": start,
                               "configs": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
