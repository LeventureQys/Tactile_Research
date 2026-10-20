# -*- coding: utf-8 -*-
"""只扫一个旋钮：r_slow_max —— 现役 K9 的慢态预算上限。

A) 六份真实录制：r_slow_max ∈ {0.1…1.0} 的恒载净变化 / 最大 5 s 窗口偏移；
B) 设备1测试2 背景 + 加载后 30~45 s 向 57 通道均分注入 100 ADC 的配对保持率
   （同一口径复核 04_右拇指专项 的四臂数据）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import (JUMP_ADC, METHODS, SESSIONS, detect_steps,  # noqa: E402
                                  load_session, med, metrics, params_of, run_k9,
                                  step_retention)

GRID = (0.1, 0.2, 0.3, 0.4, 0.6, 1.0)


def main() -> int:
    table: dict[str, dict] = {}
    for label, device, name in SESSIONS:
        if "10g" in label:      # 10 g 两份最后一并看，先扫四份恒载
            continue
        started = time.perf_counter()
        sess = load_session(device, name)
        t, pre, out_rec = sess["t"], sess["pre"], sess["out"]
        raw_total = pre.sum(1)
        steps = detect_steps(t, raw_total, JUMP_ADC)
        load = steps[0]["t"] if steps else float(t[0])
        row: dict = {"load_time_s": load, "frames": int(t.size),
                     "out_rec": dict(metrics(t, out_rec.sum(1), load))}
        for r in GRID:
            out = run_k9(pre, t, params_of(sess["session"], r))
            total = out.sum(1)
            entry = dict(metrics(t, total, load))
            entry["step_retention"] = [round(v["retention"], 3) for v in
                                       step_retention(t, total, raw_total, steps, n_each=3)]
            row["%.2f" % r] = entry
        table[label] = row
        print("%-22s 完成（%.1f s）" % (label, time.perf_counter() - started), flush=True)

    # B) 注入配对：真实背景 + 100 ADC / 57 通道，30~45 s
    probe = {}
    for label, device, name in SESSIONS:
        if "测试2" not in label or "设备1" not in label:
            continue
        sess = load_session(device, name)
        t = sess["t"]
        background = sess["pre"].copy()
        inject = background.copy()
        mask = (t >= 30.0) & (t < 45.0)
        inject[mask] += 100.0 / 57.0
        for r in GRID:
            params = params_of(sess["session"], r)
            out_bg = run_k9(background, t, params).sum(1)
            out_in = run_k9(inject, t, params).sum(1)
            delta = out_in - out_bg
            keep = float(np.median(delta[(t >= 40.0) & (t < 45.0)]))
            first = float(np.median(delta[(t >= 45.01) & (t < 50.0)]))
            probe["%.2f" % r] = dict(hold_adc=keep, retention=keep / 100.0,
                                     first5s_after=first)
            print("  r_slow_max=%.2f  注入保持 %.1f ADC（%.1f%%）  撤载后首 5 s 残差 %+.1f"
                  % (r, keep, 100.0 * keep / 100.0, first), flush=True)
        break

    payload = {"grid": list(GRID), "sessions": table, "injection_probe": probe}
    (HERE / "r_slow_scan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 恒载净变化 / 最大 5 s 窗口偏移（ADC）")
    header = "%-22s" % "会话"
    for r in GRID:
        header += " %13s" % ("r=%.1f" % r)
    print(header)
    for label in table:
        if "10g" in label:
            continue
        line = "%-22s" % label
        for r in GRID:
            entry = table[label]["%.2f" % r]
            line += " %+6.0f/%5.0f" % (entry["net_change_adc"], entry["max_abs_offset_adc"])
        print(line)
    print("%-22s" % "（现役 r=0.1 参考）")
    for label in table:
        entry = table[label]["out_rec"]
        print("   %-20s 录制输出 %+.0f/%5.0f" % (label, entry["net_change_adc"],
                                               entry["max_abs_offset_adc"]))
    print("\n-> %s" % (HERE / "r_slow_scan.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
