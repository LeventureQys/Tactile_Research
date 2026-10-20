"""在 48 min 实录上直接验参数：慢态上限 r_slow_max 与"泄漏积分"两族。

背景：gate=1 在 2 小时线性外推里输出持续下漂（−482），根因是慢态积分把"缓慢上行"当
恒定蠕变率无限积分。两个候选：
  A. 收紧上限 r_slow_max（0.6 → 0.35/0.25/0.15）；
  B. 给慢态加泄漏项（dx2 − x_slow/τ_leak），让积分在 τ 处饱和——物理上蠕变本就会饱和。

口径与 metrics.py 相同（1100~2850 s）。
输出：out/rslow_scan.json
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
LO, HI = 1100.0, 2850.0


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
m = (t >= LO) & (t <= HI)
tt = t[m]
sec = np.floor(tt).astype(np.int64)
s0 = sec - sec.min()

ARMS = {
    "K9（gate=0）": {},
    "gate r=0.6（现状）": {"floor_rise_gate": 1},
    "gate r=0.35": {"floor_rise_gate": 1, "r_slow_max": 0.35},
    "gate r=0.25": {"floor_rise_gate": 1, "r_slow_max": 0.25},
    "gate r=0.15": {"floor_rise_gate": 1, "r_slow_max": 0.15},
    "gate r=0.6 leak=1800": {"floor_rise_gate": 1, "xslow_leak_tau_s": 1800.0},
    "gate r=0.6 leak=3600": {"floor_rise_gate": 1, "xslow_leak_tau_s": 3600.0},
    "gate r=0.25 leak=3600": {"floor_rise_gate": 1, "r_slow_max": 0.25, "xslow_leak_tau_s": 3600.0},
}

res = {}
for name, ov in ARMS.items():
    v = core.run(pre, ts, ov)
    out = v.sum(axis=1)
    inp = pre.sum(axis=1)
    ap = inp - out
    sm, apm = out[m], ap[m]
    cnt = np.bincount(s0, minlength=s0[-1] + 1)
    aps = np.bincount(s0, weights=apm, minlength=cnt.size) / np.maximum(cnt, 1)
    res[name] = dict(
        drift=round(float(sm[-1] - sm[0]), 1),
        p2p=round(float(sm.max() - sm.min()), 1),
        applied_start=round(float(apm[0]), 1),
        applied_end=round(float(apm[-1]), 1),
        applied_min=round(float(apm.min()), 1),
        steps=int((np.diff(aps) < -30).sum()),
    )
    print(f"{name:24s} drift={res[name]['drift']:+8.1f} p2p={res[name]['p2p']:8.1f} "
          f"applied {res[name]['applied_start']:7.1f}->{res[name]['applied_end']:7.1f} "
          f"min={res[name]['applied_min']:7.1f} steps={res[name]['steps']:3d}")

(OUT / "rslow_scan.json").write_text(json.dumps({"window": [LO, HI], "cases": res},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> {OUT / 'rslow_scan.json'}")
