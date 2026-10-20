# -*- coding: utf-8 -*-
"""定位左拇指指腹 r_slow_max=0.6 测试1 中 195s 附近的震荡，并对比三层流。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
OUT = Path(__file__).resolve().parent


def load_stream(name: str) -> pd.DataFrame:
    path = FOLDER / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    frame = pd.read_csv(path, skiprows=skip, header=0, engine="c")
    return frame


def main() -> None:
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])

    raw = load_stream("device_001_raw_seg000.csv")
    pre = load_stream("device_001_pre_seg0.csv")
    seg = load_stream("device_001_seg000.csv")

    frame_index = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame_index - frame_index[0]) / fps

    raw_total = np.asarray(raw.iloc[:, 3:], dtype=float).sum(axis=1)
    pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
    seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)

    info = dict(folder=str(FOLDER), fps=round(fps, 4), frames=int(time.size),
                duration_s=round(float(time[-1]), 1),
                raw_frames=int(raw.shape[0]), pre_frames=int(pre.shape[0]), seg_frames=int(seg.shape[0]),
                raw_channels=int(raw.shape[1] - 3), pre_channels=int(pre.shape[1] - 3), seg_channels=int(seg.shape[1] - 3))

    # 找出 seg 相对 pre 的快速下探：以 1s 差分与 5s 中值基线衡量
    window = (time >= 170.0) & (time <= 215.0)
    rows = []
    grid = np.arange(170.0, 215.0, 0.5)
    for start in grid:
        mask = (time >= start) & (time < start + 0.5)
        if not mask.any():
            continue
        rows.append(dict(t=round(float(start), 2),
                         raw=round(float(np.median(raw_total[mask])), 1),
                         pre=round(float(np.median(pre_total[mask])), 1),
                         seg=round(float(np.median(seg_total[mask])), 1),
                         seg_min=round(float(seg_total[mask].min()), 1)))
    info["window_170_215"] = rows

    # 帧级细节：找 seg 在 190~200s 的单帧最大跳变
    band = (time >= 188.0) & (time <= 200.0)
    detail = []
    prev = None
    for index in np.flatnonzero(band):
        value = float(seg_total[index])
        step = None if prev is None else value - prev
        detail.append(dict(t=round(float(time[index]), 3), raw=round(float(raw_total[index]), 1),
                           pre=round(float(pre_total[index]), 1), seg=round(value, 1),
                           seg_step=None if step is None else round(step, 1)))
        prev = value
    info["detail_188_200"] = detail

    (OUT / "locate_result.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in info.items() if key not in ("detail_188_200",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
