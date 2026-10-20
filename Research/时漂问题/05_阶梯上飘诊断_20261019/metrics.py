"""汇总报告所需的全部数字（单次运行，输出 JSON 便于核对）。"""
import json
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
_, algoadc = load_cells("device_001_algoadc_seg0.csv")
t = ts - ts[0]
lo, hi = 1100.0, 2850.0
m = (t >= lo) & (t <= hi)
tt = t[m]

CASES = {"K9": {}, "K10": {"guard_enable": 1}, "GATE": {"floor_rise_gate": 1},
         "GATE+guard": {"floor_rise_gate": 1, "guard_enable": 1}}
res = {}
for name, ov in CASES.items():
    v = core.run(pre, ts, ov)
    s = v.sum(1)
    ap = (pre.sum(1) - s)
    sm, apm = s[m], ap[m]
    sec = np.floor(tt).astype(np.int64)
    s0 = sec - sec.min()
    cnt = np.bincount(s0, minlength=s0[-1] + 1)
    aps = np.bincount(s0, weights=apm, minlength=cnt.size) / np.maximum(cnt, 1)
    res[name] = dict(drift=float(sm[-1] - sm[0]), p2p=float(sm.max() - sm.min()),
                     applied_start=float(apm[0]), applied_end=float(apm[-1]),
                     applied_min=float(apm.min()), applied_max=float(apm.max()),
                     steps=int((np.diff(aps) < -30).sum()),
                     first=float(sm[0]), last=float(sm[-1]))

raw = pre.sum(1)[m]
aa = algoadc.sum(1)[m]
base = dict(raw_first=float(raw[0]), raw_last=float(raw[-1]), raw_drift=float(raw[-1] - raw[0]),
            raw_p2p=float(raw.max() - raw.min()),
            live_first=float(aa[0]), live_last=float(aa[-1]), live_drift=float(aa[-1] - aa[0]),
            live_p2p=float(aa.max() - aa.min()),
            live_applied_start=float(raw[0] - aa[0]), live_applied_end=float(raw[-1] - aa[-1]))
print(json.dumps({"window": [lo, hi], "input": base, "cases": res}, ensure_ascii=False, indent=2))
(OUT / "metrics_final.json").write_text(
    json.dumps({"window": [lo, hi], "input": base, "cases": res}, ensure_ascii=False, indent=2),
    encoding="utf-8")

print("\n=== 阶梯触发前 20 个事件（K9，applied 逐秒单步跌幅 >30 ADC）===")
v9 = core.run(pre, ts, {})
ap = pre.sum(1) - v9.sum(1)
sec = np.floor(t).astype(np.int64)
cnt = np.bincount(sec, minlength=sec.max() + 1)
aps = np.bincount(sec, weights=ap, minlength=cnt.size) / np.maximum(cnt, 1)
d = np.diff(aps)
n = 0
for s in np.where(d < -30)[0]:
    if s > 2850:
        continue
    print(f"  t={s:5d}s ({s/60:5.2f}min) applied {aps[s]:8.1f} -> {aps[s+1]:8.1f} ({d[s]:+8.1f})")
    n += 1
    if n >= 20:
        break
