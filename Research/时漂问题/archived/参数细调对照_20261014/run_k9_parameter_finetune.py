# -*- coding: utf-8 -*-
"""K9 工具箱参数细调对照（独立研究目录）。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "右拇指指腹"
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(TOOLBOX))
from dptool.session_csv import ROLE_FRAME_INDEX, load_table  # noqa: E402
from plugins.k9_creep_observer import _compensate  # noqa: E402

HOMES = (
    ("设备1/测试1", "设备1", "测试1-2026-10-13_16-32-15"),
    ("设备1/测试2", "设备1", "测试2-2026-10-13_16-43-59"),
    ("设备2/测试1", "设备2", "测试1-2026-10-13_16-53-24"),
    ("设备2/测试2", "设备2", "测试2-2026-10-13_17-01-06"),
)
CANDIDATES = (
    ("原参", {}),
    ("rslow_0.6", {"r_slow_max": 0.6}),
    ("rslow_0.6_cap_0.002", {"r_slow_max": 0.6, "slope_cap_frac": 0.002}),
    ("rslow_0.6_cap_0.0005", {"r_slow_max": 0.6, "slope_cap_frac": 0.0005}),
    ("rslow_0.6_gate_0.001", {"r_slow_max": 0.6, "slope_gate_frac": 0.001}),
    ("rslow_0.6_gate_0.005", {"r_slow_max": 0.6, "slope_gate_frac": 0.005}),
    ("rslow_0.6_rfast_0", {"r_slow_max": 0.6, "r_fast": 0.0}),
    ("rslow_0.6_confirm_10", {"r_slow_max": 0.6, "slow_confirm_s": 10.0}),
)
RISE_S = 2.5
HOLD_S = 15.0
OUT_JSON = HERE / "k9_parameter_finetune_results.json"
OUT_MD = HERE / "k9_parameter_finetune_report.md"


def frame_axis(table):
    found = table.header.by_role(ROLE_FRAME_INDEX) if hasattr(table.header, "by_role") else []
    if found:
        return np.asarray(table.data[:, found[0].index], dtype=np.float64)
    return np.arange(table.data.shape[0], dtype=np.float64)


def load_session(device, name):
    folder = DATA / device / name
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0].get("observed_fps") or 100.0)
    pre_table = load_table(str(folder / "device_001_pre_seg0.csv"), use_cache=True)
    frame = frame_axis(pre_table)
    values = np.asarray(pre_table.channels(), dtype=np.float64)
    t = (frame - frame[0]) / fps
    return {"id": f"{device}/{name}", "folder": str(folder), "session": session,
            "fps": fps, "t": t, "values": values}


def load_start(t, values):
    total = np.nansum(values, axis=1)
    jumps = np.flatnonzero(np.abs(np.diff(total)) >= 2000.0)
    return float(t[jumps[0] + 1]) if jumps.size else float(t[0])


def med(x):
    return float(np.nanmedian(x)) if np.size(x) else None


def constant_metrics(t, total, start):
    a = start + 15.0
    end = float(t[-1])
    first = med(total[(t >= a) & (t < a + 5.0)])
    last = med(total[t >= max(end - 5.0, a)])
    offsets = []
    for s in np.arange(a, end - 5.0 + 1e-9, 1.0):
        z = total[(t >= s) & (t < s + 5.0)]
        if z.size and first is not None:
            offsets.append(float(np.nanmedian(z) - first))
    tail = (t >= max(end - 5.0, a)) & (t <= end)
    if np.sum(tail) >= 2:
        slope = float(np.polyfit(t[tail] - t[tail][0], total[tail], 1)[0])
    else:
        slope = None
    return {"load_start_s": start, "first5_adc": first, "last5_adc": last,
            "net_change_adc": None if first is None or last is None else last - first,
            "max_abs_offset_5s_adc": max((abs(x) for x in offsets), default=None),
            "max_offset_5s_adc": max(offsets, key=abs) if offsets else None,
            "tail_slope_adc_s": slope, "n_windows": len(offsets)}


def make_injection(values, t, start, concentrated):
    x = values.copy()
    event = (t >= start) & (t < start + HOLD_S)
    ramp = np.clip((t - start) / RISE_S, 0.0, 1.0)
    level = np.where(event, ramp, 0.0)
    if concentrated:
        x[event, 0] += level[event] * 100.0
    else:
        # 57-channel dispersed injection: identical total 100 ADC, not 300 ADC.
        n = min(57, x.shape[1])
        x[event, :n] += (level[event, None] * (100.0 / n))
    return x


def params_for(session, overrides):
    p = dict(session["algorithm"]["params"])
    p.update(overrides)
    return p


def run(values, t, params):
    out, stats = _compensate(values, t, dict(params))
    return np.asarray(out, dtype=np.float64), stats


def injection_result(values, t, params, start, concentrated):
    clean, _ = run(values, t, params)
    injected, _ = run(make_injection(values, t, start, concentrated), t, params)
    delta = np.nansum(injected - clean, axis=1)
    hold = (t >= start + 10.0) & (t < start + HOLD_S)
    release = (t >= start + HOLD_S + 7.0) & (t < start + HOLD_S + 12.0)
    h = med(delta[hold])
    r = med(delta[release])
    return {"hold_last5_adc": h, "retention": None if h is None else h / 100.0,
            "release_7_12_adc": r, "event_start_s": start, "rise_s": RISE_S,
            "hold_end_s": start + HOLD_S, "injected_total_adc": 100.0,
            "shape": "集中1通道" if concentrated else "分散57通道"}


def write_report(payload):
    rows = payload["candidates"]
    lines = ["# K9 参数细调对照报告（2026-10-14）", "",
             "本报告只使用工具箱 `plugins.k9_creep_observer._compensate`，每候选均从原始 `pre` 流独立重放；不修改 05 路线B及旧代码。",
             "恒载：设备1/测试1、设备1/测试2、设备2/测试1、设备2/测试2；加载后+15 s起，首尾5 s中值、完整5 s滚动窗口最大偏移、末5 s斜率。",
             "注入：设备1/测试2真实背景，加载后+30 s，2.5 s上升，保持至起点+15 s后撤载；分散57通道与集中1通道均为总100 ADC。候选波动不作为实际10 g验收。", "",
             "|候选|四恒载最差首尾|四恒载最差5 s偏移|最差末斜率|分散保持/率|分散撤载7~12 s|集中保持/率|集中撤载7~12 s|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        c = row["constant_summary"]
        d = row["injections"]["分散57通道"]
        q = row["injections"]["集中1通道"]
        lines.append("|%s|%.1f|%.1f|%+.3f|%.1f (%.1f%%)|%+.1f|%.1f (%.1f%%)|%+.1f|" %
                     (row["name"], c["worst_abs_net"], c["worst_abs_offset"], c["worst_abs_tail_slope"],
                      d["hold_last5_adc"], 100*d["retention"], d["release_7_12_adc"],
                      q["hold_last5_adc"], 100*q["retention"], q["release_7_12_adc"]))
    lines += ["", "## 判定", ""]
    best = min(rows, key=lambda r: (r["constant_summary"]["worst_abs_offset"], -min(r["injections"][k]["retention"] for k in r["injections"])))
    lines.append("四恒载联合是否进入300 ADC以内、以及100 ADC注入是否达到70%%保持率，按表中同一候选同时判断。最小综合偏移候选为 **%s**；完整是否达标以联合条件为准。" % best["name"])
    lines.append("实际10 g没有同步真值标签，本文只报告候选注入脉冲，不把任何真实录制波动解释成10 g验收。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sessions = {label: load_session(device, name) for label, device, name in HOMES}
    host = sessions["设备1/测试2"]
    inject_start = load_start(host["t"], host["values"]) + 30.0
    payload = {"method": {"toolbox": str(TOOLBOX), "algorithm": "plugins.k9_creep_observer._compensate", "numpy_channel_vectorized": True,
                           "session_params": "每会话读取 session.json algorithm.params，候选仅覆盖列明字段", "fps": "session devices[0].observed_fps", "time": "frame_index/fps",
                           "ten_g_note": "只输出候选脉冲，不作为实际10g验收"},
               "injection": {"host": host["id"], "start_s": inject_start, "rise_s": RISE_S, "hold_s": HOLD_S, "total_adc_each": 100.0}, "candidates": []}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, overrides in CANDIDATES:
        started = time.perf_counter()
        row = {"name": name, "overrides": overrides, "sessions": {}, "injections": {}}
        worst_net = worst_off = 0.0
        worst_slope = 0.0
        for label, sess in sessions.items():
            p = params_for(sess["session"], overrides)
            out, _ = run(sess["values"], sess["t"], p)
            m = constant_metrics(sess["t"], np.nansum(out, axis=1), load_start(sess["t"], sess["values"]))
            row["sessions"][label] = m
            worst_net = max(worst_net, abs(m["net_change_adc"]))
            worst_off = max(worst_off, m["max_abs_offset_5s_adc"])
            worst_slope = max(worst_slope, abs(m["tail_slope_adc_s"]))
        row["constant_summary"] = {"worst_abs_net": worst_net, "worst_abs_offset": worst_off, "worst_abs_tail_slope": worst_slope}
        p = params_for(host["session"], overrides)
        row["injections"]["分散57通道"] = injection_result(host["values"], host["t"], p, inject_start, False)
        row["injections"]["集中1通道"] = injection_result(host["values"], host["t"], p, inject_start, True)
        row["elapsed_s"] = round(time.perf_counter() - started, 2)
        payload["candidates"].append(row)
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(payload)
        d, q = row["injections"]["分散57通道"], row["injections"]["集中1通道"]
        print("%-26s 恒载 %.1f/%.1f  分散 %.1f(%.1f%%)  集中 %.1f(%.1f%%)  用时 %.1fs" %
              (name, worst_net, worst_off, d["hold_last5_adc"], 100*d["retention"], q["hold_last5_adc"], 100*q["retention"], row["elapsed_s"]), flush=True)
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
