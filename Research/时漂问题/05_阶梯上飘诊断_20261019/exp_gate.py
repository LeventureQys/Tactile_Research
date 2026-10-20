"""验证 floor_rise_gate（y_floor 仅空载上升）在 2026-10-19 会话上的效果。"""
from pathlib import Path

import numpy as np
import pandas as pd

import core

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"


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
t = ts - ts[0]
raw = pre.sum(1)

CASES = {
    "K9 现役": {},
    "K10 guard=1": {"guard_enable": 1},
    "gate=1": {"floor_rise_gate": 1},
    "gate=1 + guard=1": {"floor_rise_gate": 1, "guard_enable": 1},
    "gate=1 + τ=600": {"floor_rise_gate": 1, "y_floor_tau_s": 600.0},
    "gate=1 + τ=1200": {"floor_rise_gate": 1, "y_floor_tau_s": 1200.0},
}

print(f"{'变体':22s} {'恒载漂移':>9s} {'p2p':>8s} {'补偿末':>8s} {'0-10min':>9s} {'10-30min':>9s} {'30-48min':>9s}")
for name, over in CASES.items():
    v, L, Lb, Ls = core.run(pre, ts, over, log=True)
    a = v.sum(1)
    ap = L["applied"].sum(1)
    def d(lo, hi):
        m = (t >= lo) & (t <= hi)
        return a[m][-1] - a[m][0]
    m = (t >= 1100) & (t <= 2850)
    print(f"{name:22s} {d(1100,2850):+9.1f} {a[m].max()-a[m].min():8.1f} {ap[m][-1]:8.1f}"
          f" {d(1100,1700):+9.1f} {d(1700,2300):+9.1f} {d(2300,2850):+9.1f}")
    print("     applied每200s: " + " ".join(f"{ap[np.searchsorted(t, tt)]:6.0f}"
                                            for tt in range(600, 2900, 200)))

print(f"\n参考：raw/pre 自身 1100~2850 漂移 = {raw[(t>=1100)&(t<=2850)][-1]-raw[(t>=1100)&(t<=2850)][0]:+.1f}")
print(f"      实测 K9 记录 1100~2850 漂移 = {pd.read_csv if False else ''}", end="")
_, algoadc = load_cells("device_001_algoadc_seg0.csv")
aa = algoadc.sum(1)
m = (t >= 1100) & (t <= 2850)
print(f"{aa[m][-1]-aa[m][0]:+.1f}")
