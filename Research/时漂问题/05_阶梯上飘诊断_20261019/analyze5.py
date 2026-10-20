"""定位：把 idle 判据彻底焊死后，崩溃是否还会发生？"""
from pathlib import Path

import numpy as np
import pandas as pd

import core

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


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

CASES = {
    "band=1e9（idle 永不成立）": {"idle_band_floor_adc": 1e9},
    "band=1000": {"idle_band_floor_adc": 1000.0},
    "band=200+zero门0.5": {"idle_band_floor_adc": 200.0, "zero_track_gate": 0.5},
    "band=200+zero门0.5+关快泄放": {"idle_band_floor_adc": 200.0, "zero_track_gate": 0.5,
                                 "tau_r_slow_idle_s": 0.0},
}

for name, over in CASES.items():
    v, L, Lb, Ls = core.run(pre, ts, over, log=True)
    ap = L["applied"].sum(1)
    a = v.sum(1)
    m = (t >= 1100) & (t <= 2850)
    print(f"\n### {name}")
    print(f"  恒载漂移={a[m][-1]-a[m][0]:+8.1f}  p2p={a[m].max()-a[m].min():8.1f}  "
          f"补偿首={ap[m][0]:8.1f} 末={ap[m][-1]:8.1f}  min={ap[m].min():8.1f}")
    print(f"  idle 帧/通道占比={float((L['idle'] > 0.5).mean())*100:.2f}%  bypass 帧={int(Lb.sum())}")
    print("   t  min  applied_sum  xslow_sum  xfast_sum  zero_sum   yfl_sum  idle_ch")
    for tt in range(900, 2200, 100):
        i = np.searchsorted(t, tt)
        print(f"  {tt:5d} {tt/60:5.1f} {ap[i]:11.1f} {L['x_slow'].sum(1)[i]:10.1f} "
              f"{L['x_fast'].sum(1)[i]:10.1f} {L['zero'].sum(1)[i]:9.1f} {L['y_floor'].sum(1)[i]:9.1f}"
              f" {int((L['idle'][i] > 0.5).sum()):7d}")
    np.savez_compressed(OUT / f"case_{abs(hash(name)) % 100000}.npz", t=t, alg=a, applied=ap,
                        xslow=L["x_slow"].sum(1), zero=L["zero"].sum(1), yfl=L["y_floor"].sum(1),
                        idle=(L["idle"] > 0.5).sum(1), name=np.array([name]))
