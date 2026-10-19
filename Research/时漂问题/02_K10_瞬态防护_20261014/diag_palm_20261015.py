# -*- coding: utf-8 -*-
"""K10 调参前置诊断：右手掌-2026-10-15_15-22-01 工况定标（第一步）。

口径与仓库既有脚本一致：
  - ##Data 之后为数据段；前三列为 时间戳/经过时间/帧序号
  - 录制 elapsed 列存在大量重复（工程 B 已记录 ~70%），统一按 observed_fps 重建均匀时间轴
  - 仅读主显示流 device_001_seg000.csv（本次是 K10 算法的真实输入）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右手掌-2026-10-15_15-22-01")
OUT = Path(__file__).resolve().parent / "palm_diag_20261015"
OUT.mkdir(parents=True, exist_ok=True)
MAIN = "device_001_seg000.csv"


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
    meta = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    dev = meta["devices"][0]
    fps = float(dev["observed_fps"])

    frame = load_stream(FOLDER, MAIN)
    elapsed = np.asarray(frame.iloc[:, 1], dtype=float)
    index = np.asarray(frame.iloc[:, 2], dtype=float)
    x = np.asarray(frame.iloc[:, 3:], dtype=float)
    n, channels = x.shape
    t = np.arange(n, dtype=float) / fps

    dt = np.diff(elapsed)
    index_diff = np.diff(index)
    info = dict(
        main_stream=MAIN, frames=int(n), channels=int(channels), observed_fps=round(fps, 4),
        duration_uniform_s=round(float(t[-1]), 2), duration_elapsed_s=round(float(elapsed[-1]), 2),
        elapsed_unique=int(np.unique(elapsed).size), elapsed_unique_ratio=round(float(np.unique(elapsed).size / n), 4),
        dt_median_s=float(np.median(dt)), dt_zero_or_neg_frac=round(float((dt <= 0).mean()), 4),
        index_step_min=float(index_diff.min()), index_step_max=float(index_diff.max()),
        index_step_unique=int(np.unique(index_diff).size),
        nan_count=int(np.isnan(x).sum()),
    )

    total = x.sum(1)
    rest = t < 3.0
    info["total"] = dict(
        min=float(total.min()), max=float(total.max()),
        first3s_median=float(np.median(total[rest])),
        last5s_median=float(np.median(total[t > t[-1] - 5.0])),
        first3s_std=float(np.std(total[rest])),
        last5s_std=float(np.std(total[t > t[-1] - 5.0])),
    )

    grid = []
    step = 10.0
    a = 0.0
    while a + step <= t[-1]:
        mask = (t >= a) & (t < a + step)
        grid.append(dict(t_mid=round(a + step / 2, 1), med=round(float(np.median(total[mask])), 1),
                         lo=round(float(total[mask].min()), 1), hi=round(float(total[mask].max()), 1),
                         std=round(float(np.std(total[mask])), 1)))
        a += step

    chan_median = np.median(x, axis=0)
    chan_max = x.max(axis=0)
    info["channels_stat"] = dict(
        median_of_median=float(np.median(chan_median)),
        max_of_median=float(chan_max.max()),
        active_channels_median_gt_100=int((chan_median > 100.0).sum()),
        median_of_total_reading=float(np.median(total)),
    )

    payload = dict(source=str(FOLDER), info=info, total_grid=grid,
                   channel_median=[round(float(v), 1) for v in chan_median],
                   channel_max=[round(float(v), 1) for v in chan_max])
    (OUT / "diag_step1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(info, ensure_ascii=False, indent=1))
    print("--- 每 10s 总值(中值|最低|最高) ---")
    line = []
    for g in grid:
        line.append(f"{g['t_mid']:.0f}s:{g['med']:.0f}|{g['lo']:.0f}|{g['hi']:.0f}")
        if len(line) == 5:
            print("   " + "  ".join(line))
            line = []
    if line:
        print("   " + "  ".join(line))
    print(f"输出: {OUT / 'diag_step1.json'}")


if __name__ == "__main__":
    main()
