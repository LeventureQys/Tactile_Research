# -*- coding: utf-8 -*-
"""全数据下探事件检测 + 事件后算法引入的基线偏移关联统计。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data")
TARGETS = [
    ("左拇指0.6/测试1", ROOT / "左拇指指腹 r_slow_max=0.6" / "测试1-2026-10-14_10-18-40"),
    ("左拇指0.6/测试2", ROOT / "左拇指指腹 r_slow_max=0.6" / "测试2-2026-10-14_10-39-51"),
]
OUT = Path(__file__).resolve().parent

DROP_ADC = 150.0
MERGE_S = 0.3
RECOVER_FRACTION = 0.5
RECOVER_S = 0.5


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


def detect_dips(t, total):
    diff = np.diff(total)
    candidates = np.flatnonzero(diff < -DROP_ADC)
    events = []
    for index in candidates:
        if events and t[index] - events[-1]["t_drop"] < MERGE_S:
            event = events[-1]
            if total[index + 1] < event["trough"]:
                event["trough"] = float(total[index + 1])
                event["t_trough"] = float(t[index + 1])
                event["t_drop"] = float(t[index])
            continue
        events.append(dict(t_drop=float(t[index]), trough=float(total[index + 1]),
                           t_trough=float(t[index + 1]), start_level=float(total[index])))
    return events


def main() -> None:
    report = {}
    for label, folder in TARGETS:
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        fps = float(session["devices"][0]["observed_fps"])
        pre = load_stream(folder, "device_001_pre_seg0.csv")
        seg = load_stream(folder, "device_001_seg000.csv")
        frame = np.asarray(pre.iloc[:, 2], dtype=float)
        t = (frame - frame[0]) / fps
        pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
        seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)

        events = detect_dips(t, pre_total)
        rows = []
        for event in events:
            drop = event["start_level"] - event["trough"]
            level = event["start_level"]
            recovered = med(pre_total, t, event["t_trough"] + 1.0, event["t_trough"] + 2.0)
            if recovered is None:
                continue
            recovered_fraction = (recovered - event["trough"]) / max(level - event["trough"], 1e-9)
            pre_before = med(pre_total, t, event["t_drop"] - 20.0, event["t_drop"] - 2.0)
            pre_after = med(pre_total, t, event["t_trough"] + 20.0, event["t_trough"] + 40.0)
            seg_before = med(seg_total, t, event["t_drop"] - 20.0, event["t_drop"] - 2.0)
            seg_after = med(seg_total, t, event["t_trough"] + 20.0, event["t_trough"] + 40.0)
            if None in (pre_before, pre_after, seg_before, seg_after):
                continue
            rows.append(dict(
                t_drop=round(event["t_drop"], 2),
                drop_adc=round(float(drop), 1),
                drop_relative=round(float(drop / max(level, 1.0)), 4),
                recovery_fraction=round(float(recovered_fraction), 3),
                pre_change=round(float(pre_after - pre_before), 1),
                seg_change=round(float(seg_after - seg_before), 1),
                algorithm_induced=round(float((seg_after - seg_before) - (pre_after - pre_before)), 1),
            ))
        consistent = [r for r in rows if r["recovery_fraction"] > RECOVER_FRACTION]
        report[label] = dict(
            fps=round(fps, 4), duration_s=round(float(t[-1]), 1), frames=int(t.size),
            drop_threshold_adc=DROP_ADC, events=len(rows), recovered_events=len(consistent),
            baseline_shift_after_recovered_events=[r["algorithm_induced"] for r in consistent],
            median_induced_shift=None if not consistent else round(float(np.median([r["algorithm_induced"] for r in consistent])), 1),
            events_detail=rows,
        )
        (OUT / "dip_events.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {label}: frames={t.size} duration={t[-1]:.0f}s events={len(rows)} recovered={len(consistent)}")
        for row in rows:
            print("   ", row)


if __name__ == "__main__":
    main()
