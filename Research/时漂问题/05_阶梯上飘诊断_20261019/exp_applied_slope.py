"""用 48 min 数据判断"数小时会不会过补"的可操作判据：applied 后段是否还在一路涨。

思路：数小时过补的根源是补偿量持续增长。若在 30~48 min 的窗口里 applied 已经收敛
（斜率≈0），说明慢态积分进入平台，长时间外推不会继续吃漂移；若仍以固定速率上涨，
数小时后必然过补。这样就不必等真实长录制。

输出：out/applied_slope.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import core  # noqa: E402

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")


def load_cells(name):
    path = ROOT / name
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


def slope_per_min(t, y):
    tc = t - t.mean()
    return float((tc * (y - y.mean())).sum() / (tc ** 2).sum() * 60.0)


ts, pre = load_cells("device_001_pre_seg0.csv")
t = ts - ts[0]
inp = pre.sum(1)
m = (t >= 1500) & (t <= 2850)
tt = t[m]

rows = []
for name, ov in (("K9", {}),
                 ("gate r=0.6", {"floor_rise_gate": 1}),
                 ("gate r=0.35", {"floor_rise_gate": 1, "r_slow_max": 0.35}),
                 ("gate r=0.25", {"floor_rise_gate": 1, "r_slow_max": 0.25}),
                 ("gate r=0.15", {"floor_rise_gate": 1, "r_slow_max": 0.15})):
    v = core.run(pre, ts, ov)
    out = v.sum(1)
    ap = inp - out
    row = dict(
        arm=name,
        applied_slope_1500_2850=round(slope_per_min(tt, ap[m]), 1),
        applied_1500=round(float(np.median(ap[(t >= 1500) & (t < 1560)])), 1),
        applied_2850=round(float(np.median(ap[(t >= 2790) & (t <= 2850)])), 1),
        out_slope_1500_2850=round(slope_per_min(tt, out[m]), 1),
        in_slope_1500_2850=round(slope_per_min(tt, inp[m]), 1),
    )
    rows.append(row)
    print(f"{name:14s} applied {row['applied_1500']:8.1f} -> {row['applied_2850']:8.1f} "
          f"(斜率 {row['applied_slope_1500_2850']:+6.1f} ADC/min) | "
          f"输出斜率 {row['out_slope_1500_2850']:+6.1f} | 输入 {row['in_slope_1500_2850']:+6.1f}")

(OUT / "applied_slope.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> {OUT / 'applied_slope.json'}")
