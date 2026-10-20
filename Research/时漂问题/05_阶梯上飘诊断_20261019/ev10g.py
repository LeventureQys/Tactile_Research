"""10g 反复增减会话上的事件保留率：K9 vs 修复。"""
from pathlib import Path

import numpy as np
import pandas as pd

import core

DATA = Path(__file__).resolve().parents[1] / "data" / "archived"
SESSION_DIRS = [
    DATA / "右拇指指腹" / "设备1" / "10g反复增减荷载-2026-10-13_17-07-18",
    DATA / "右拇指指腹" / "设备2" / "10g反复增减荷载-2026-10-13_17-05-32",
]


def read_cells(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


def local_extrema(d, min_sep_s, dt, thr):
    """返回幅度超过 thr 的局部极大/极小索引（绝对不期）。"""
    a = np.abs(d)
    gap = max(int(min_sep_s / dt), 1)
    idx = []
    i = 1
    n = len(a)
    while i < n - 1:
        if a[i] > thr and a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            idx.append(i)
            i += gap
        else:
            i += 1
    return np.array(idx, dtype=np.int64)


for d in SESSION_DIRS:
    ts, pre = read_cells(d / "device_001_pre_seg0.csv")
    t = ts - ts[0]
    dt = float(np.median(np.diff(t)))
    raw = pre.sum(1)
    win = max(int(5.0 / dt), 3)
    base = pd.Series(raw).rolling(win, center=True, min_periods=1).mean().to_numpy()
    din = raw - base
    thr = 60.0
    pk = local_extrema(din, 2.0, dt, thr)
    v9 = core.run(pre, ts, {})
    vg = core.run(pre, ts, {"floor_rise_gate": 1})
    vr = core.run(pre, ts, {"floor_rise_gate": 1, "r_slow_max": 0.25})
    outs = {"K9": v9.sum(1), "修复": vg.sum(1), "修复r0.25": vr.sum(1)}
    print(f"\n### {d.name}  帧={len(t)} 时长={t[-1]:.1f}s  事件数={len(pk)}")
    amps = np.abs(din[pk])
    print(f"  输入事件幅度: 中位={np.median(amps):.1f} 均值={amps.mean():.1f} "
          f"p10={np.percentile(amps,10):.1f} p90={np.percentile(amps,90):.1f} ADC")
    for name, s in outs.items():
        bs = pd.Series(s).rolling(win, center=True, min_periods=1).mean().to_numpy()
        dout = s - bs
        r = np.abs(dout[pk]) / amps
        print(f"  {name}: 保留率 中位={np.median(r):.3f} 均值={r.mean():.3f} "
              f"p10={np.percentile(r,10):.3f}  输出幅度均值={np.abs(dout[pk]).mean():.1f} ADC")
    # 逐事件明细（前后 8 个）
    bs9 = pd.Series(outs["K9"]).rolling(win, center=True, min_periods=1).mean().to_numpy()
    bg = pd.Series(outs["修复"]).rolling(win, center=True, min_periods=1).mean().to_numpy()
    print("   t(s)  方向  in   K9  修复   保留率K9 / 修复")
    for i in pk[:14]:
        dirn = "升" if din[i] > 0 else "降"
        a = np.abs(din[i])
        b = np.abs(outs["K9"][i] - bs9[i])
        c = np.abs(outs["修复"][i] - bg[i])
        print(f"  {t[i]:6.1f}   {dirn}  {a:5.1f} {b:5.1f} {c:5.1f}    {b/a:5.2f} / {c/a:5.2f}")
