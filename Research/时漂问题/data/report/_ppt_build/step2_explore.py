"""Exploratory plots to read the actual behavior of both sessions."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
FIG = HERE / "explore"
FIG.mkdir(exist_ok=True)

RAW_C = "#8A93A3"
ALGO_C = "#1F4E79"
TOT_C = "#C98A26"

curves = {k: np.load(HERE / f"{k}_curves.npz") for k in ("dev1", "dev2")}

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
for row, key in enumerate(("dev1", "dev2")):
    d = curves[key]
    t = d["t"]
    ax = axes[row][0]
    ax.plot(t, d["raw_sum"], lw=0.8, color=RAW_C, label="raw sum (before comp)")
    ax.plot(t, d["algo_sum"], lw=0.8, color=ALGO_C, label="algo sum (after comp)")
    ax.plot(t, d["force_total"], lw=0.6, color=TOT_C, alpha=0.7, label="forcetotal stream")
    ax.set_title(f"{key} full session  ({t[-1]:.0f}s, {t.size} frames)", fontweight="bold")
    ax.set_xlabel("elapsed s")
    ax.set_ylabel("force mN")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[row][1]
    ax.plot(t, d["raw_pos"], lw=0.7, color=RAW_C, label="raw active cells")
    ax.plot(t, d["algo_pos"], lw=0.7, color=ALGO_C, label="algo active cells")
    ax.set_title(f"{key} active-cell count", fontweight="bold")
    ax.set_xlabel("elapsed s")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "overview.png", dpi=110)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(20, 13))
for col, key in enumerate(("dev1", "dev2")):
    d = curves[key]
    t = d["t"]
    for row, (lo, hi) in enumerate(((0, 90), (0, 300), (300, t[-1]))):
        m = (t >= lo) & (t <= hi)
        ax = axes[row][col]
        ax.plot(t[m], d["raw_sum"][m], lw=0.9, color=RAW_C, label="raw")
        ax.plot(t[m], d["algo_sum"][m], lw=0.9, color=ALGO_C, label="algo")
        ax.set_title(f"{key}  {lo:.0f}-{hi:.0f}s", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "zoom.png", dpi=110)
plt.close(fig)

for key, d in curves.items():
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    print(f"=== {key} ===")
    print("t range", t[0], t[-1])
    for lo, hi in ((30, 60), (60, 90), (55, 70), (90, 120), (600, t[-1])):
        m = (t >= lo) & (t <= hi)
        if not m.any():
            continue
        print(f"{lo:6.0f}-{hi:6.0f}s raw mean {raw[m].mean():9.0f} algo mean {algo[m].mean():9.0f} "
              f"raw drift {raw[m][-1] - raw[m][0]:8.0f} algo drift {algo[m][-1] - algo[m][0]:8.0f}")
print("figs ->", FIG)
