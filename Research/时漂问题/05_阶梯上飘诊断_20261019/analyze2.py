"""二阶分析：谁把 y_floor/zero/idle 推成台阶。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"
d = np.load(OUT / "replay.npz")
t = d["t"]


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

print("=== 1. 三个时间窗的逐通道 pre 均值（荷载空间重分布？）===")
wins = [(960, 1080), (1700, 1800), (1800, 1980)]
mus = []
for a, b in wins:
    m = (t >= a) & (t < b)
    mus.append(pre[m].mean(0))
print("  win      sum      min      max    p2p")
for (a, b), mu in zip(wins, mus):
    print(f"  {a}-{b} {mu.sum():9.1f} {mu.min():8.1f} {mu.max():8.1f} {mu.max()-mu.min():6.1f}")
dd = mus[2] - mus[0]
order = np.argsort(dd)
print("\n  通道变化（1800-1980 减 960-1080），最降 12 与最升 12：")
for k in list(order[:12]) + list(order[-12:]):
    print(f"    cell{k:2d}: {mus[0][k]:8.1f} -> {mus[2][k]:8.1f}  ({dd[k]:+8.1f})")
print(f"  |delta|>20 的通道数={int((np.abs(dd) > 20).sum())}  最大|delta|={np.abs(dd).max():.1f}")

print("\n=== 2. cell6 状态轨迹（1050~2100s，每 25s）===")
print("   t   min      pre     zero      y0  y_floor   y_max   span  thr=.05span  y0-yfloor idle"
      "  x_slow  x_fast  applied  dwell   slope")
for tt in range(1050, 2125, 25):
    i = np.searchsorted(t, tt)
    pre6 = d["pre_c6"][i]; zero6 = d["k9_zero_c6"][i]
    y0 = pre6 - zero6
    yf = d["k9_y_floor_c6"][i]; ym = d["k9_y_max_c6"][i]
    span = max(ym - yf, 1.0)
    print(f"{tt:5d} {tt/60:5.1f} {pre6:8.1f} {zero6:8.1f} {y0:8.1f} {yf:8.1f} {ym:8.1f} {span:7.1f}"
          f" {0.05*span:11.1f} {y0-yf:11.1f} {int(d['k9_idle_count'][i]):4d}"
          f" {d['k9_x_slow_c6'][i]:8.1f} {d['k9_x_fast_c6'][i]:7.1f} {d['k9_applied_c6'][i]:8.1f}"
          f" {d['k9_dwell_c6'][i]:7.1f} {d['k9_slope_c6'][i]:7.2f}")

print("\n=== 3. 全通道 idle 数 / y_floor / x_slow 的拐点（每 60s，1080~2100）===")
print("   t   min   idle_ch  yfloor_sum  xslow_sum  xfast_sum  applied_sum  pre_sum  dwell_max")
for tt in range(1080, 2100, 60):
    i = np.searchsorted(t, tt)
    print(f"{tt:5d} {tt/60:5.1f} {int(d['k9_idle_count'][i]):7d} {d['k9_yfloor_sum'][i]:11.1f}"
          f" {d['k9_xslow_sum'][i]:10.1f} {d['k9_xfast_sum'][i]:10.1f} {d['k9_applied_sum'][i]:12.1f}"
          f" {d['pre_sum'][i]:9.1f} {d['k9_dwell_max'][i]:9.1f}")

print("\n=== 4. cell6 首次 y_floor 连续下跌段（相邻帧累计跌 >20）===")
yf = d["k9_y_floor_c6"]
run_start = None
acc = 0.0
i = 1
n = len(t)
events = []
while i < n:
    dy = yf[i] - yf[i - 1]
    if dy < -0.05:
        if run_start is None:
            run_start = i - 1
            acc = 0.0
        acc += dy
        i += 1
        continue
    if run_start is not None and acc < -20:
        events.append((run_start, i, acc))
    run_start = None
    acc = 0.0
    i += 1
print(f"  段数={len(events)}")
for a, b, acc in events[:20]:
    print(f"  t={t[a]:8.2f}~{t[b]:8.2f}s ({t[a]/60:5.2f}min) 累计={acc:+8.2f} "
          f"y_floor {yf[a]:8.1f} -> {yf[b]:8.1f}  pre {d['pre_c6'][a]:7.1f} -> {d['pre_c6'][b]:7.1f}")

print("\n=== 5. pre_sum 的 1s 波动（是否真有整体卸载/加载）===")
for tt in range(1080, 2000, 120):
    m = (t >= tt) & (t < tt + 5)
    seg = d["pre_sum"][m]
    print(f"{tt:5d} pre_sum mean={seg.mean():9.1f} min={seg.min():9.1f} max={seg.max():9.1f} "
          f"p2p={seg.max()-seg.min():7.1f}")
