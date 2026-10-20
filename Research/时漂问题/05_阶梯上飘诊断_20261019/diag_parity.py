"""定位产品实现与 portable 实现的第一个分叉帧。"""
import numpy as np
import pandas as pd
from pathlib import Path

import core

WORK = Path(r"D:\Temp\k10fix_check")
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


ts, pre = load_cells("device_001_pre_seg0.csv")
n = pre.shape[1]
prod = np.fromfile(WORK / "prod_g0.bin", dtype=np.float64).reshape(-1, n)
py = core.run(pre, ts, {"floor_rise_gate": 0})

for name, a, b in (("prod vs core.py", prod, py),):
    d = np.abs(a - b)
    print(f"{name}: max={d.max():.6e} mean={d.mean():.6e}")
    i = int(np.argmax(d.max(1) > 1e-9))
    print(f"  首个差>1e-9 帧 #{i}  t={(ts[i]-ts[0]):.4f}s  max|Δ|={d.max():.3e}")
    print("  pre :", pre[i][:8])
    print("  prod:", prod[i][:8])
    print("  py  :", py[i][:8])

# 与 portable 的差
ports = np.fromfile(WORK / "prod_g0.bin", dtype=np.float64).reshape(-1, n)
print("\n检查 prod 与 core.py 是否同源；若同源，说明分叉在 core.py/产品 与 portable 之间。")
