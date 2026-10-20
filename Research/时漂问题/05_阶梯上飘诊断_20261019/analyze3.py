"""三阶分析：找出"idle 误判 → zero 吞载 → 补偿永久失效"的逐帧触发点。"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
d = np.load(OUT / "replay.npz")
t = d["t"]
pre6 = d["pre_c6"]; zero6 = d["k9_zero_c6"]; yf6 = d["k9_y_floor_c6"]
ym6 = d["k9_y_max_c6"]; xs6 = d["k9_x_slow_c6"]; xf6 = d["k9_x_fast_c6"]
ap6 = d["k9_applied_c6"]; dw6 = d["k9_dwell_c6"]; idle6 = d["k9_idle_c6"] > 0.5
ap = d["k9_applied_c6"]

print("=== A. cell6 首次 idle 的帧 ===")
idx = np.where(idle6)[0]
print(f"  idle 帧总数={len(idx)}  首帧 #{idx[0]} t={t[idx[0]]:.3f}s ({t[idx[0]]/60:.2f}min)")

print("\n=== B. cell6 崩溃前 1560~1640s 逐帧（idle 前后 40 帧全打印）===")
lo, hi = np.searchsorted(t, 1555), np.searchsorted(t, 1645)
print("  idx     t   pre  zero     y0  y_floor   y_max   span   thr   y0-yfl  idle   x_slow  applied  dwell  slope")
for i in range(lo, hi):
    y0 = pre6[i] - zero6[i]
    span = max(ym6[i] - yf6[i], 1.0)
    thr = 0.05 * span
    mark = " <<<" if idle6[i] else ""
    print(f"{i:6d} {t[i]:7.3f} {pre6[i]:5.0f} {zero6[i]:5.1f} {y0:6.1f} {yf6[i]:7.1f} {ym6[i]:7.1f}"
          f" {span:6.1f} {thr:5.1f} {y0-yf6[i]:7.1f} {int(idle6[i]):5d} {xs6[i]:7.2f} {ap6[i]:7.2f}"
          f" {dw6[i]:6.1f} {d['k9_slope_c6'][i]:7.2f}{mark}")

print("\n=== C. 崩溃前 cell6 的 pre 单帧波动（1555~1645s）===")
seg = pre6[lo:hi]
print(f"  mean={seg.mean():.2f} min={seg.min():.0f} max={seg.max():.0f} p2p={seg.max()-seg.min():.0f} "
      f"std={seg.std():.2f} 帧数={len(seg)}")
diff = np.diff(seg)
print(f"  帧间差分: mean={diff.mean():+.2f} std={diff.std():.2f} min={diff.min():+.0f} max={diff.max():+.0f}")

print("\n=== D. 各通道 applied 塌陷到 <10% 的首次时刻（台阶的时间分布）===")
keep = [6, 44, 17, 2, 7, 12, 14, 1, 21, 11]
rows = []
for c in keep:
    a = d[f"k9_applied_c{c}"]
    base = a[(t > 900) & (t < 1000)].mean()
    if base <= 1:
        continue
    hit = np.where(a < 0.1 * base)[0]
    hit = hit[hit > 1000]
    rows.append((t[hit[0]] if len(hit) else np.nan, c, base))
for tt, c, base in sorted(rows):
    print(f"  cell{c:2d} applied 基线={base:7.2f}  首次跌到<10%: "
          f"{'t=%.1fs (%.2f min)' % (tt, tt/60) if np.isfinite(tt) else '未发生'}")

print("\n=== E. 崩溃后显示值 vs 输入（cell6，每 100s）===")
print("   t    min    pre    applied   display≈pre-applied")
for tt in range(1500, 2860, 100):
    i = np.searchsorted(t, tt)
    print(f"{tt:5d} {tt/60:5.1f} {pre6[i]:8.1f} {ap6[i]:8.2f} {pre6[i]-ap6[i]:10.1f}")
