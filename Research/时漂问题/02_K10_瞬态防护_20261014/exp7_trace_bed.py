# -*- coding: utf-8 -*-
"""1) 固件 pre 全程轨迹与事件；2) 合成测试床：寻找能复现「下探→永久偏移」的条件。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
OUT = Path(__file__).resolve().parent
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(TOOLBOX))
from plugins.k9_creep_observer import _compensate  # noqa: E402

FPS = 100.0
CH = 52


def load_stream(name: str) -> pd.DataFrame:
    path = FOLDER / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def firmware_trace():
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    pre = load_stream("device_001_pre_seg0.csv")
    seg = load_stream("device_001_seg000.csv")
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    t = (frame - frame[0]) / fps
    pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
    seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)
    trace = []
    for start in np.arange(0.0, float(t[-1]), 20.0):
        mask = (t >= start) & (t < start + 20.0)
        if not mask.any():
            continue
        trace.append(dict(t=round(float(start), 1),
                          pre=round(float(np.median(pre_total[mask])), 0),
                          seg=round(float(np.median(seg_total[mask])), 0),
                          applied=round(float(np.median((pre_total - seg_total)[mask])), 0)))
    events = []
    diff = np.diff(pre_total)
    for index in np.flatnonzero(np.abs(diff) > 200.0):
        events.append(dict(t=round(float(t[index]), 2), d=round(float(diff[index]), 0)))
    return dict(fps=round(fps, 4), trace=trace, large_events=events[:40],
                event_count=int((np.abs(diff) > 200.0).sum()))


def synthetic_probe(initial_load: bool, dip_amp: float, dip_frames: int):
    """合成受载背景 + 单通道下探，检查 applied 是否永久减少。"""
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    params = session["algorithm"]["params"]
    duration = 200.0
    time = np.arange(int(duration * FPS)) / FPS
    values = np.full((time.size, CH), 6.0)
    values[time >= 2.0] += 837.0
    if initial_load:
        values[time < 2.0] = 843.0
    clean, _ = _compensate(values, time, params)
    injected = values.copy()
    start = int(120.0 * FPS)
    injected[start:start + dip_frames, 13] -= dip_amp
    out, _ = _compensate(injected, time, params)
    ap_clean = (values - np.asarray(clean, dtype=float))
    ap_dip = (injected - np.asarray(out, dtype=float))
    before = (time >= 100.0) & (time < 119.0)
    after = (time >= 125.0) & (time < 160.0)
    return dict(initial_load=initial_load, dip_amp=dip_amp, dip_frames=dip_frames,
                ch13_applied_change=round(float(ap_dip[after, 13].mean() - ap_dip[before, 13].mean()), 2),
                total_applied_change=round(float(ap_dip[after].mean() - ap_dip[before].mean()), 2),
                clean_drift=round(float(ap_clean[after].mean() - ap_clean[before].mean()), 2))


def main() -> None:
    report = dict(firmware=firmware_trace(), synthetic=[])
    for initial_load in (False, True):
        for dip_amp in (100.0, 400.0):
            for dip_frames in (1, 10, 30):
                report["synthetic"].append(synthetic_probe(initial_load, dip_amp, dip_frames))
    (OUT / "trace_and_bed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("events>200ADC in pre:", report["firmware"]["event_count"])
    for row in report["firmware"]["large_events"][:20]:
        print("   ", row)
    print("pre trace (每20s):")
    for row in report["firmware"]["trace"][:30]:
        print("   ", row)
    print("synthetic bed:")
    for row in report["synthetic"]:
        print("   ", row)


if __name__ == "__main__":
    main()
