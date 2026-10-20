"""K9 vs K10 在同一条数据上的行为对照 + 崩溃点定位。"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
d = np.load(OUT / "replay.npz")
t = d["t"]


def first_after(t, arr_bool, t0):
    idx = np.where(arr_bool & (t >= t0))[0]
    return idx[0] if len(idx) else None


print("=== A. cell6 在 t>1000s 首次 idle 的帧 ===")
idle6 = d["k9_idle_c6"] > 0.5
i0 = first_after(t, idle6, 1000)
print(f"  首帧 #{i0} t={t[i0]:.3f}s ({t[i0]/60:.2f}min)")

print("\n=== B. 该帧前后 30 帧 ===")
pre6 = d["pre_c6"]; zero6 = d["k9_zero_c6"]; yf6 = d["k9_y_floor_c6"]
ym6 = d["k9_y_max_c6"]; xs6 = d["k9_x_slow_c6"]; ap6 = d["k9_applied_c6"]
print("  idx      t   pre  zero     y0  y_floor   y_max   span   thr  y0-yfl idle  x_slow  applied")
for i in range(i0 - 12, i0 + 30):
    y0 = pre6[i] - zero6[i]
    span = max(ym6[i] - yf6[i], 1.0)
    print(f"{i:6d} {t[i]:7.3f} {pre6[i]:5.0f} {zero6[i]:5.1f} {y0:6.1f} {yf6[i]:7.1f} {ym6[i]:7.1f}"
          f" {span:6.1f} {0.05*span:5.1f} {y0-yf6[i]:6.1f} {int(idle6[i]):4d} {xs6[i]:7.2f} {ap6[i]:7.2f}")

print("\n=== C. K9 与 K10 同数据对照（每 60s）===")
print("  t   min   pre_sum   k9_sum  k10_sum  k9_applied k10_applied  k9_xslow k10_xslow  k9_yflsum k10_yflsum  k9_idle k10_idle")
for tt in range(900, 2100, 60):
    i = np.searchsorted(t, tt)
    print(f"{tt:5d} {tt/60:5.1f} {d['pre_sum'][i]:9.0f} {d['k9_sum'][i]:8.0f} {d['k10_sum'][i]:8.0f}"
          f" {d['k9_applied_sum'][i]:10.1f} {d['k10_applied_sum'][i]:11.1f}"
          f" {d['k9_xslow_sum'][i]:9.1f} {d['k10_xslow_sum'][i]:10.1f}"
          f" {d['k9_yfloor_sum'][i]:10.1f} {d['k10_yfloor_sum'][i]:11.1f}"
          f" {int(d['k9_idle_count'][i]):7d} {int(d['k10_idle_count'][i]):8d}")

print("\n=== D. 全时段 K9 vs K10 输出差（分段）===")
diff = d["k10_sum"] - d["k9_sum"]
for a in range(0, 2860, 300):
    m = (t >= a) & (t < a + 300)
    print(f"  {a:4d}-{a+300:4d}s  mean={diff[m].mean():+8.2f} max|d|={np.abs(diff[m]).max():8.2f}")

print("\n=== E. cell6 的 K10 zero / y_floor 轨迹 ===")
for tt in range(1500, 2160, 60):
    i = np.searchsorted(t, tt)
    print(f"{tt:5d} {tt/60:5.1f} k9: zero={d['k9_zero_c6'][i]:7.1f} yfl={d['k9_y_floor_c6'][i]:8.1f}"
          f" xslow={d['k9_x_slow_c6'][i]:7.2f} ap={d['k9_applied_c6'][i]:7.2f}"
          f" || k10: zero={d['k10_zero_c6'][i]:7.1f} yfl={d['k10_y_floor_c6'][i]:8.1f}"
          f" xslow={d['k10_x_slow_c6'][i]:7.2f} ap={d['k10_applied_c6'][i]:7.2f}")

print("\n=== F. 总力漂移指标（恒载段 18min 起，K9/K10/无补偿）===")
seg = (t >= 1100) & (t <= 1900)
for name, arr in (("raw/pre", d["pre_sum"]), ("K9", d["k9_sum"]), ("K10", d["k10_sum"])):
    x = arr[seg]
    print(f"  {name:8s}: 首={x[0]:9.1f} 尾={x[-1]:9.1f} 首尾差={x[-1]-x[0]:+8.1f} "
          f"max-min={x.max()-x.min():8.1f} 斜率={np.polyfit(t[seg]-t[seg][0], x, 1)[0]*60:+.3f} ADC/min")
