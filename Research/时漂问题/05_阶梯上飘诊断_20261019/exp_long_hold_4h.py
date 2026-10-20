"""四小时外推：确认 gate=1 的补偿是"收敛"还是"无限增长"。

上一轮 2 h 外推显示 0.6 档输出下漂 −482、0.25 档 −53。但 48 min 原始数据的末段
（1500~2850 s）applied 斜率都≈0（已收敛），两者矛盾——说明 2 h 的差异发生在过渡期。
本脚本跑到 4 h，看 applied 是否真的收敛、以及收敛后输出漂移等于多少。

输出：out/long_hold_extrap_4h.json
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
HOURS = 4.0
SEG_MIN = 30


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
m = (t >= 1100) & (t < 2850)
tt = t[m]
seg = pre[m]
tc = tt - tt.mean()
slope = (tc[:, None] * (seg - seg.mean(axis=0))).sum(axis=0) / float((tc ** 2).sum())
detrend = seg - tc[:, None] * slope

block_s = float(tt[-1] - tt[0])
n_rep = int(HOURS * 3600.0 / block_s) + 1
block = np.tile(detrend, (n_rep, 1))
tt0 = tt - tt[0]
ts_rel = np.concatenate([tt0 + k * block_s for k in range(n_rep)])
hold = block + ts_rel[:, None] * slope

lead_m = t < 60.0
lead = pre[lead_m]
n_lead = int(lead_m.sum())
values = np.vstack([lead, hold])
ts_long = np.concatenate([ts[:n_lead], ts[0] + 60.0 + ts_rel])
print(f"frames={len(ts_long)} span={ts_long[-1]-ts_long[0]:.0f}s "
      f"input_drift={values[-1].sum()-lead[-1].sum():.0f} ADC")

inp = values.sum(axis=1)
report = {"hours": HOURS, "frames": int(len(ts_long)), "span_s": round(float(ts_long[-1]-ts_long[0]), 1),
          "input_drift_total": round(float(values[-1].sum() - lead[-1].sum()), 1), "arms": {}}

for name, ov in (("gate r=0.6", {"floor_rise_gate": 1}),
                 ("gate r=0.25", {"floor_rise_gate": 1, "r_slow_max": 0.25})):
    v = core.run(values, ts_long, ov)
    out = v.sum(axis=1)
    ap = inp - out
    rel = ts_long - ts_long[0]
    rows = []
    base = float(np.median(out[rel < SEG_MIN * 60]))
    for lo in range(0, int(HOURS * 3600), SEG_MIN * 60):
        mm = (rel >= lo) & (rel < lo + SEG_MIN * 60)
        if mm.sum() == 0:
            continue
        rows.append(dict(window=f"{lo//60}-{(lo+SEG_MIN*60)//60}min",
                         inp=round(float(np.median(inp[mm])), 1),
                         out=round(float(np.median(out[mm])), 1),
                         applied=round(float(np.median(ap[mm])), 1),
                         drift=round(float(np.median(out[mm]) - base), 1)))
    report["arms"][name] = rows
    print(f"\n=== {name}（4 h 外推）===")
    for r in rows:
        print(f"  {r['window']:>11s} in={r['inp']:9.1f} out={r['out']:9.1f} "
              f"applied={r['applied']:8.1f} drift={r['drift']:+8.1f}")
    del v

(OUT / "long_hold_extrap_4h.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
print(f"\n-> {OUT / 'long_hold_extrap_4h.json'}")
