# -*- coding: utf-8 -*-
"""用工具箱 K9（按 cpp 修正的转写）重放 pre，检查 195s 震荡是否复现 60 ADC 基线抬升。"""
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

LIMIT_S = 260.0


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


def med(sig, t, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def main() -> None:
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    params = session["algorithm"]["params"]

    pre = load_stream("device_001_pre_seg0.csv")
    seg = load_stream("device_001_seg000.csv")
    values = np.asarray(pre.iloc[:, 3:], dtype=float)
    seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)
    frame_index = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame_index - frame_index[0]) / fps
    limit = int(np.searchsorted(time, LIMIT_S))

    replay, _ = _compensate(values[:limit], time[:limit], params)
    replay_total = np.asarray(replay, dtype=float).sum(axis=1)
    t = time[:limit]

    rows = []
    for start in np.arange(188.0, 205.0, 0.5):
        rows.append(dict(t=round(float(start), 2),
                         pre=round(med(values[:limit].sum(axis=1), t, start, start + 0.5), 1),
                         fw_seg=round(med(seg_total[:limit], t, start, start + 0.5), 1),
                         replay=round(med(replay_total, t, start, start + 0.5), 1)))

    before, after = (193.5, 194.5), (196.0, 198.0)
    shift = dict(
        fw_seg=round(med(seg_total[:limit], t, *after) - med(seg_total[:limit], t, *before), 2),
        replay=round(med(replay_total, t, *after) - med(replay_total, t, *before), 2),
        pre=round(med(values[:limit].sum(axis=1), t, *after) - med(values[:limit].sum(axis=1), t, *before), 2),
    )
    # 固件 seg 与重放的基线差（确认绝对电平差异）
    level = dict(fw_minus_replay=round(med(seg_total[:limit], t, *before) - med(replay_total, t, *before), 2))

    payload = dict(fps=round(fps, 4), windows=rows, shift_across_194_5s=shift, level=level)
    (OUT / "replay_check.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"shift_across_194_5s": shift, "level": level}, ensure_ascii=False))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
