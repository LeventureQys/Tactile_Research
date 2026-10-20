"""在重放轨迹上定位"阶梯上飘"的机制起点。"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
d = np.load(OUT / "replay.npz")
t = d["t"]
KEEP = [6, 44, 17, 2, 7, 12, 14, 1, 21, 11]


def at(arr, tt):
    return arr[np.searchsorted(t, tt)]


print("=== A. 逐秒 applied_sum / idle_count（12~36min，每 30s）===")
print(" t(s)  min   pre_sum   k9_sum  applied  x_slow  x_fast  yfloor_sum idle_cnt dwell_max")
for tt in range(720, 2160, 30):
    i = np.searchsorted(t, tt)
    print(f"{tt:5d} {tt/60:5.1f} {d['pre_sum'][i]:9.0f} {d['k9_sum'][i]:9.0f} "
          f"{d['k9_applied_sum'][i]:8.1f} {d['k9_xslow_sum'][i]:8.1f} {d['k9_xfast_sum'][i]:8.1f} "
          f"{d['k9_yfloor_sum'][i]:10.1f} {int(d['k9_idle_count'][i]):3d} {d['k9_dwell_max'][i]:6.1f}")

print("\n=== B. y_floor 单帧跌幅 >2 ADC 的事件（前 30）===")
ev = []
for c in KEEP:
    yf = d[f"k9_y_floor_c{c}"]
    dy = np.diff(yf)
    idx = np.where(dy < -2.0)[0]
    for i in idx:
        ev.append((t[i + 1], c, dy[i], yf[i], yf[i + 1]))
ev.sort()
print(f"  事件总数={len(ev)}（10 个样本通道）")
for e in ev[:30]:
    print(f"  t={e[0]:8.3f}s ({e[0]/60:5.2f}min) cell{e[1]:2d} drop={e[2]:+8.2f} "
          f"y_floor {e[3]:8.2f} -> {e[4]:8.2f}")

print("\n=== C. applied_sum 逐秒跌幅 >30 ADC 的台阶 ===")
ap = d["k9_applied_sum"]
sec = np.floor(t).astype(np.int64)
nsec = sec.max() + 1
ap_sec = np.bincount(sec, weights=ap, minlength=nsec)[:nsec] / np.maximum(np.bincount(sec, minlength=nsec), 1)
idle_sec = np.bincount(sec, weights=d["k9_idle_count"], minlength=nsec)[:nsec]
dap = np.diff(ap_sec)
steps = np.where(dap < -30)[0]
print(f"  台阶数={len(steps)}")
for s in steps[:40]:
    print(f"  t={s:5d}s ({s/60:5.2f}min) applied {ap_sec[s]:8.1f} -> {ap_sec[s+1]:8.1f} "
          f"({dap[s]:+8.1f})  idle通道数约 {idle_sec[s+1]:5.1f}")

print("\n=== D. cell6 详细轨迹（1080~1320s，每 10s）===")
hdr = "  t     pre     zero   y0=pre-zero  y_floor   y_max   x_slow  applied idle dwell  slope"
print(hdr)
for tt in range(1080, 1330, 10):
    i = np.searchsorted(t, tt)
    pre = d["pre_c6"][i]
    zero = d["k9_zero_c6"][i]
    print(f"{tt:5d} {pre:8.2f} {zero:8.2f} {pre-zero:12.2f} {d['k9_y_floor_c6'][i]:9.2f} "
          f"{d['k9_y_max_c6'][i]:8.2f} {d['k9_x_slow_c6'][i]:8.2f} {d['k9_applied_c6'][i]:8.2f} "
          f"{d['k9_idle_count'][i]:4.0f} {d['k9_dwell_c6'][i]:6.2f} {d['k9_slope_c6'][i]:8.2f}")

print("\n=== E. cell6 原始输入自身波动（1080~1320s 每 10s 的 pre 与 1s 内 min/max）===")
for tt in range(1080, 1330, 10):
    m = (t >= tt) & (t < tt + 1)
    seg = d["pre_c6"][m]
    print(f"{tt:5d}  pre mean={seg.mean():8.2f}  min={seg.min():8.2f} max={seg.max():8.2f} "
          f"p2p={seg.max()-seg.min():7.2f}")

print("\n=== F. 全通道 v 与 zero 的长期关系（每 300s 快照，cell6）===")
for tt in range(0, 2860, 300):
    i = np.searchsorted(t, tt)
    print(f"{tt:5d} pre={d['pre_c6'][i]:8.2f} zero={d['k9_zero_c6'][i]:8.2f} "
          f"y_floor={d['k9_y_floor_c6'][i]:9.2f} y_max={d['k9_y_max_c6'][i]:9.2f} "
          f"x_slow={d['k9_x_slow_c6'][i]:7.2f} applied={d['k9_applied_c6'][i]:7.2f}")
