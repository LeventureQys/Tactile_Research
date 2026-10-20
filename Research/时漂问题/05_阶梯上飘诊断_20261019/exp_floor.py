"""假设检验：y_floor 的"无条件慢速上升"是否是把载荷吸进基线的主因。"""
from pathlib import Path

import numpy as np
import pandas as pd

import core

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
t = ts - ts[0]
raw_sum = pre.sum(1)

CASES = {
    "K9 基线": {},
    "K10 guard=1": {"guard_enable": 1},
    "y_floor 冻结(τ=∞) guard=0": {"y_floor_tau_s": 1e9},
    "y_floor 冻结(τ=∞) guard=1": {"y_floor_tau_s": 1e9, "guard_enable": 1},
    "y_floor τ=3000": {"y_floor_tau_s": 3000.0},
    "y_floor τ=600": {"y_floor_tau_s": 600.0},
    "guard=1 confirm=2s": {"guard_enable": 1, "guard_floor_confirm_s": 2.0},
    "guard=1 deep=1e9(不许深跌破快跟)": {"guard_enable": 1, "guard_floor_deep": 1e9},
}

print(f"{'变体':32s} {'恒载漂移':>9s} {'p2p':>8s} {'补偿末':>8s} {'末0~10min漂移':>12s} {'yfl末':>9s}")
res = {}
for name, over in CASES.items():
    v, L, Lb, Ls = core.run(pre, ts, over, log=True)
    a = v.sum(1)
    m = (t >= 1100) & (t <= 2850)
    ap = L["applied"].sum(1)
    print(f"{name:32s} {a[m][-1]-a[m][0]:+9.1f} {a[m].max()-a[m].min():8.1f} {ap[m][-1]:8.1f}"
          f" {a[(t>=1100)&(t<=1700)][-1]-a[(t>=1100)&(t<=1700)][0]:+12.1f}"
          f" {L['y_floor'].sum(1)[m][-1]:9.1f}")
    res[name] = dict(alg=a, applied=ap, xslow=L["x_slow"].sum(1), zero=L["zero"].sum(1),
                     yfl=L["y_floor"].sum(1), idle=(L["idle"] > 0.5).sum(1))
    print(f"    每 200s applied: " + " ".join(f"{ap[np.searchsorted(t, tt)]:7.0f}"
                                              for tt in range(600, 2900, 200)))
np.savez_compressed(Path(__file__).resolve().parent / "out" / "floor_test.npz", t=t,
                    **{f"{k}|{kk}": vv for k, r in res.items() for kk, vv in r.items()})
print("\n哨兵：raw/pre 自身 1100~2850 漂移 = %+.1f" % (raw_sum[(t >= 1100) & (t <= 2850)][-1]
                                                - raw_sum[(t >= 1100) & (t <= 2850)][0]))
