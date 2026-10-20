"""长恒载可行性查验（一）：补偿覆盖率是否逼近上限、输入漂移是否共模。

问题：48 min 里 gate=1 漂 −74，但用户要数小时恒载。两个隐患必须量化：
  A. x_slow 的幅度上限是 r_slow_max·e。若覆盖率（applied/y0）在 48 min 内已逼近 0.6，
     数小时后补偿必然顶死，输出退回原始蠕变率。
  B. gate=1 冻结了 zero 跟踪，长期零点漂移只能靠 x_slow 吸收。若漂移是全通道共模
     （传感器本体/温漂），可以用共模估计把它和"局部接触载荷"分开；若各通道乱走，
     共模方案不成立。

输出：out/long_hold_audit.json
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
n = pre.shape[1]
rsm = core.BASE["r_slow_max"]
print(f"frames={len(t)} ch={n} span={t[-1]:.1f}s  r_slow_max={rsm}")

report = {"frames": int(len(t)), "channels": int(n), "r_slow_max": rsm}

# ── A. gate=1 全程，补偿覆盖率与状态 ──
v, L, Lb, Ls = core.run(pre, ts, {"floor_rise_gate": 1}, log=True)
y = pre - L["zero"]
ap = L["applied"]
xf, xs = L["x_fast"], L["x_slow"]
cover = ap / np.maximum(y, 1.0)
rows = []
for lo in range(0, 2850, 300):
    m = (t >= lo) & (t < lo + 300)
    if m.sum() == 0:
        continue
    rows.append(dict(
        window=f"{lo}-{lo+300}",
        y_med=round(float(np.median(y[m])), 1),
        applied_med=round(float(np.median(ap[m])), 1),
        cover_med=round(float(np.median(cover[m])), 3),
        cover_p95=round(float(np.percentile(cover[m], 95)), 3),
        xslow_med=round(float(np.median(xs[m])), 1),
        xfast_med=round(float(np.median(xf[m])), 1),
        zero_med=round(float(np.median(L["zero"][m])), 1),
        yfloor_med=round(float(np.median(L["y_floor"][m])), 1),
    ))
B = y > 1.0
hi_cover = float((cover[B] > rsm * 0.9).mean()) if B.any() else 0.0
report["coverage"] = {"rows": rows, "ratio_above_0p9rsm": round(hi_cover, 4)}
print("\n=== A. gate=1 补偿覆盖率（applied / y0）===")
print(f"{'窗口':>10s} {'y0':>8s} {'applied':>9s} {'cover中位':>9s} {'p95':>7s} {'x_slow':>8s} {'y_floor':>9s}")
for r in rows:
    print(f"{r['window']:>10s} {r['y_med']:8.1f} {r['applied_med']:9.1f} {r['cover_med']:9.3f} "
          f"{r['cover_p95']:7.3f} {r['xslow_med']:8.1f} {r['yfloor_med']:9.1f}")
print(f"  cover > 0.9·r_slow_max 的通道-帧占比 = {hi_cover*100:.2f}%")
del v, L, Lb, Ls

# ── B. 输入漂移是共模还是差模 ──
m = (t >= 1100) & (t <= 2850)
tt = t[m]
seg = pre[m]
tc = tt - tt.mean()
den = float((tc ** 2).sum())
slope = (tc[:, None] * (seg - seg.mean(axis=0))).sum(axis=0) / den * 60.0   # ADC/min
common = float(np.median(slope))
diff = slope - np.median(slope)
report["drift_decomp"] = dict(
    slope_median=round(common, 3),
    slope_mean=round(float(slope.mean()), 3),
    slope_std=round(float(slope.std()), 3),
    slope_min=round(float(slope.min()), 3),
    slope_max=round(float(slope.max()), 3),
    positive_ratio=round(float((slope > 0).mean()), 3),
    diff_median=round(float(np.median(diff)), 3),
    diff_std=round(float(diff.std()), 3),
    common_over_total=round(float(abs(common) / (abs(common) + diff.std())), 3),
)
print("\n=== B. 输入漂移分解（1100–2850 s，逐通道 ADC/min）===")
print("  " + json.dumps(report["drift_decomp"], ensure_ascii=False))

(OUT / "long_hold_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> {OUT / 'long_hold_audit.json'}")
