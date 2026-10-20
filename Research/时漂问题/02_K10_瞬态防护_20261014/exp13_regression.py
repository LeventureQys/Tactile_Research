# -*- coding: utf-8 -*-
"""完整录制回归：瞬态防护是否改变原有漂移/稳态指标。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data")
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLBOX))
from guarded_k9 import compensate  # noqa: E402
from plugins.k9_creep_observer import _compensate  # noqa: E402

TARGETS = [
    ("左拇指0.6/测试1", DATA / "左拇指指腹 r_slow_max=0.6" / "测试1-2026-10-14_10-18-40"),
    ("左拇指0.6/测试2", DATA / "左拇指指腹 r_slow_max=0.6" / "测试2-2026-10-14_10-39-51"),
    ("右拇指0.6/测试1", DATA / "右拇指指腹 r_slow_max=0.6" / "测试1-2026-10-14_11-06-29"),
]


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


def main() -> None:
    report = {}
    for label, folder in TARGETS:
        if not folder.exists():
            report[label] = {"error": "missing"}
            continue
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        params = session["algorithm"]["params"]
        pre = load_stream(folder, "device_001_pre_seg0.csv")
        values = np.asarray(pre.iloc[:, 3:], dtype=float)
        frame = np.asarray(pre.iloc[:, 2], dtype=float)
        observed = float(session["devices"][0].get("observed_fps") or 0.0)
        if observed <= 0:
            steps = np.diff(frame)
            positive = steps[steps > 0]
            observed = 1.0 / float(np.median(positive)) if positive.size else 100.0
        time = (frame - frame[0]) / observed

        plain = np.asarray(_compensate(values, time, params)[0], dtype=float).sum(axis=1)
        guarded, stats = compensate(values, time, params, dict(enabled=True))
        guarded = np.asarray(guarded, dtype=float).sum(axis=1)

        total_pre = values.sum(axis=1)
        jumps = np.flatnonzero(np.abs(np.diff(total_pre)) >= 2000.0)
        load = float(time[jumps[0] + 1]) if jumps.size else float(time[0])
        start = load + 15.0
        head = (time >= start) & (time < start + 5.0)
        tail = time >= float(time[-1]) - 5.0

        def drift(sig):
            return float(np.median(sig[tail]) - np.median(sig[head]))

        diffs = []
        for lo in np.arange(start, float(time[-1]) - 60.0, 60.0):
            mask = (time >= lo) & (time < lo + 60.0)
            if mask.any():
                diffs.append(abs(float(np.median(plain[mask]) - np.median(guarded[mask]))))
        report[label] = dict(
            fps=round(observed, 4), frames=int(time.size), duration_s=round(float(time[-1]), 1),
            load_s=round(load, 2),
            drift_plain=round(drift(plain), 2), drift_guarded=round(drift(guarded), 2),
            drift_delta=round(drift(guarded) - drift(plain), 2),
            max_60s_median_diff=round(max(diffs), 3) if diffs else None,
            mean_60s_median_diff=round(float(np.mean(diffs)), 3) if diffs else None,
            max_instant_diff=round(float(np.max(np.abs(plain - guarded))), 2),
            fast_release_frames=int(stats.get("fast_release_frames", 0)),
        )
        (HERE / "regression_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(label, json.dumps(report[label], ensure_ascii=False))


if __name__ == "__main__":
    main()
