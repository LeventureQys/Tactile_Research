# -*- coding: utf-8 -*-
"""路线B 的能力边界：真实背景 + 受控 100 ADC 分散注入，只变「上升时间」。

背景：设备1测试2 原始录制（57 通道、100 Hz）。
注入：加载后 30 s 起、向全部 57 通道均分总 100 ADC，上升时间取 0.05~5 s，保持到 45 s 后撤去。
口径：配对（注入臂输出 − 同参数无注入背景臂输出），与路线B 自己的 final_probe 一致。
对照臂：同一背景下「仅把 r_slow_max 提到 0.4」的纯 K9。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import load_session, params_of, run_k9  # noqa: E402
from route_b import route_b                                       # noqa: E402

RISE_S = (0.05, 0.5, 1.0, 2.0, 2.5, 3.6, 5.0)
START, HOLD_END = 30.0, 45.0


def build_injection(t: np.ndarray, rise_s: float) -> np.ndarray:
    load = np.zeros((t.size, 57), dtype=np.float64)
    ramp = np.clip((t - START) / rise_s, 0.0, 1.0)
    gate = (t >= START) & (t < HOLD_END)
    level = np.where(gate, ramp, np.where(t >= HOLD_END, 0.0, 0.0))
    load[:, :] = (100.0 / 57.0) * level[:, None]
    return load


def main() -> int:
    sess = None
    for label, device, name in [("设备1/测试2", "设备1", "测试2-2026-10-13_16-43-59")]:
        sess = load_session(device, name)
    t = sess["t"]
    background = sess["pre"].astype(np.float64)
    raw_params = dict(sess["session"]["algorithm"]["params"])

    def run(method: str, values: np.ndarray) -> np.ndarray:
        if method == "k9_0p4":
            return run_k9(values, t, params_of(sess["session"], 0.4)).sum(1)
        out, _trace = route_b(values, t, dict(raw_params, r_slow_max=0.4))
        return out.sum(1)

    rows = []
    for method in ("k9_0p4", "routeb"):
        bg_out = run(method, background)
        for rise in RISE_S:
            inject = background + build_injection(t, rise)
            pair = run(method, inject) - bg_out
            hold = float(np.median(pair[(t >= HOLD_END - 5.0) & (t < HOLD_END)]))
            resid = float(np.median(pair[(t >= HOLD_END + 1.0) & (t < HOLD_END + 6.0)]))
            rows.append(dict(method=method, rise_s=rise, hold_adc=hold,
                             retention=hold / 100.0, withdraw_residual_adc=resid))
            print("  %-8s 上升 %4.2f s → 持载保持 %6.1f ADC（%5.1f%%）  撤载后 1~6 s 残差 %+6.1f"
                  % (method, rise, hold, 100.0 * hold / 100.0, resid), flush=True)

    print("\n%-10s" % "上升时间s" + "".join("%18s" % m for m in ("K9@0.4", "路线B")) + "   差值")
    for rise in RISE_S:
        cells = []
        for method in ("k9_0p4", "routeb"):
            row = next(r for r in rows if r["method"] == method and r["rise_s"] == rise)
            cells.append(row["hold_adc"])
        print("%-10.2f" % rise + "".join("%17.1f " % c for c in cells)
              + "  %+7.1f" % (cells[1] - cells[0]))

    (HERE / "routeb_injection_rise.json").write_text(
        json.dumps({"rise_grid": list(RISE_S), "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n-> %s" % (HERE / "routeb_injection_rise.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
