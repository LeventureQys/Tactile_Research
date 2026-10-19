# -*- coding: utf-8 -*-
"""跨录制下探事件统计：验证哪些数据存在「瞬态下探污染基线」现象。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data")
TARGETS = [
    ("左拇指0.6/测试1", DATA / "左拇指指腹 r_slow_max=0.6" / "测试1-2026-10-14_10-18-40"),
    ("右拇指0.6/测试1", DATA / "右拇指指腹 r_slow_max=0.6" / "测试1-2026-10-14_11-06-29"),
    ("右拇指/设备1-10g", DATA / "右拇指指腹" / "设备1" / "10g反复增减荷载-2026-10-13_17-07-18"),
    ("右拇指/设备2-10g", DATA / "右拇指指腹" / "设备2" / "10g反复增减荷载-2026-10-13_17-05-32"),
    ("右拇指/设备1-测试1", DATA / "右拇指指腹" / "设备1" / "测试1-2026-10-13_16-32-15"),
    ("右拇指/设备2-测试1", DATA / "右拇指指腹" / "设备2" / "测试1-2026-10-13_16-53-24"),
]
DROP_ADC = 150.0
MERGE_S = 0.3


def load_stream(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def med(sig, t, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def main() -> None:
    report = {}
    for label, folder in TARGETS:
        if not folder.exists():
            report[label] = {"error": "missing"}
            continue
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        pre = load_stream(folder, "device_001_pre_seg0.csv")
        seg = load_stream(folder, "device_001_seg000.csv")
        frame = np.asarray(pre.iloc[:, 2], dtype=float)
        observed = float(session["devices"][0].get("observed_fps") or 0.0)
        if observed <= 0:
            positive = np.diff(frame)
            positive = positive[positive > 0]
            observed = 1.0 / float(np.median(positive)) if positive.size else 100.0
        t = (frame - frame[0]) / observed
        pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
        seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)

        diff = np.diff(pre_total)
        candidates = np.flatnonzero(diff < -DROP_ADC)
        events = []
        for index in candidates:
            if events and t[index] - events[-1]["t"] < MERGE_S:
                if pre_total[index + 1] < events[-1]["trough"]:
                    events[-1]["trough"] = float(pre_total[index + 1])
                    events[-1]["t_trough"] = float(t[index + 1])
                continue
            events.append(dict(t=float(t[index]), level=float(pre_total[index]),
                               trough=float(pre_total[index + 1]), t_trough=float(t[index + 1])))

        rows = []
        for event in events:
            drop = event["level"] - event["trough"]
            if drop < DROP_ADC:
                continue
            recovered = med(pre_total, t, event["t_trough"] + 1.0, event["t_trough"] + 2.0)
            if recovered is None:
                continue
            fraction = (recovered - event["trough"]) / max(event["level"] - event["trough"], 1e-9)
            if fraction < 0.5:
                continue  # 未恢复 -> 真实卸载，不是瞬态
            before = med(pre_total, t, event["t"] - 20.0, event["t"] - 2.0)
            after = med(pre_total, t, event["t_trough"] + 20.0, event["t_trough"] + 40.0)
            seg_before = med(seg_total, t, event["t"] - 20.0, event["t"] - 2.0)
            seg_after = med(seg_total, t, event["t_trough"] + 20.0, event["t_trough"] + 40.0)
            if None in (before, after, seg_before, seg_after):
                continue
            rows.append(dict(t=round(event["t"], 2), drop_adc=round(float(drop), 1),
                             recovery=round(float(fraction), 3),
                             pre_change=round(float(after - before), 1),
                             seg_change=round(float(seg_after - seg_before), 1),
                             applied_shift=round(float((before - seg_before) - (after - seg_after)), 1)))
        report[label] = dict(frames=int(t.size), duration_s=round(float(t[-1]), 1),
                             fps=round(observed, 3), transient_dips=len(rows), events=rows)
        (OUT / "dip_crosscheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(label, "transient_dips=", len(rows), rows, flush=True)


if __name__ == "__main__":
    main()
