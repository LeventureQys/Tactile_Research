"""Diagnostics for the 61 s step magnitude under different definitions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
d = np.load(HERE / "dev1_curves.npz")
t, raw, algo = d["t"] / 1.0, d["raw_sum"], d["algo_sum"]

base = (t >= 58.0) & (t <= 61.4)
b_raw, b_algo = float(raw[base].mean()), float(algo[base].mean())
print(f"baseline 58-61.4s: raw {b_raw:.1f} mN  algo {b_algo:.1f} mN")

for x in (58.0, 60.0, 61.0, 61.3, 61.4, 61.5, 61.6, 61.8, 62.0, 62.4, 63.0, 65.0, 70.0, 90.0):
    i = int(np.searchsorted(t, x))
    i = min(i, t.size - 1)
    print(f"  t={x:6.2f}s raw {raw[i]:9.1f} algo {algo[i]:9.1f} | delta_raw {raw[i]-b_raw:8.1f} delta_algo {algo[i]-b_algo:8.1f}")


def bin_mean(t, y, tmax, step):
    keep = t <= tmax
    t, y = t[keep], y[keep]
    edges = np.arange(0.0, tmax + step, step)
    idx = np.clip(np.digitize(t, edges) - 1, 0, edges.size - 2)
    n = edges.size - 1
    sums = np.bincount(idx, weights=y, minlength=n)
    cnts = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return (edges[:-1] + edges[1:]) / 2.0, v, cnts


c, r, cnt = bin_mean(t, (raw - b_raw) / 1000.0, 92.0, 0.05)
_, a, _ = bin_mean(t, (algo - b_algo) / 1000.0, 92.0, 0.05)
print("\nbinned view (0.05 s, N):")
for x in (61.35, 61.425, 61.5, 61.6, 61.8, 62.0, 62.4, 63.0, 90.0):
    i = int(np.searchsorted(c, x))
    i = min(max(i, 0), c.size - 1)
    print(f"  bin@{c[i]:7.3f}s raw {r[i]:+.3f} N (n={cnt[i]})  algo {a[i]:+.3f} N")
print("\nnan in r:", int(np.isnan(r).sum()), "of", r.size, "| nan in a:", int(np.isnan(a).sum()))
print("interp(62.0) - interp(61.4): raw",
      round(float(np.interp(62.0, c, r) - np.interp(61.4, c, r)), 4),
      "algo", round(float(np.interp(62.0, c, a) - np.interp(61.4, c, a)), 4))
print("interp(62.4) - interp(61.4): raw",
      round(float(np.interp(62.4, c, r) - np.interp(61.4, c, r)), 4),
      "algo", round(float(np.interp(62.4, c, a) - np.interp(61.4, c, a)), 4))
print("interp(90.0): raw", round(float(np.interp(90.0, c, r)), 4),
      "algo", round(float(np.interp(90.0, c, a)), 4))
