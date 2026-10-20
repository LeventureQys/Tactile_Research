"""r_slow_max 对照图：0.60（现役慢态上限）vs 0.25（拟改）+ 合成真值检验。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE / "figures_rslow"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2 = "#0A1020", "#48587A"
BLUE, BLUE_DEEP, MUTED = "#155DFC", "#0B2E96", "#93A0B8"
GRID = "#E5EBF7"
ADC_PER_N = 1000.0
T_START = 60.0

plt.rcParams.update({
    "font.family": "Microsoft YaHei", "font.size": 12,
    "axes.edgecolor": "#C9D4E8", "axes.labelcolor": INK2, "axes.linewidth": 0.9,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

CHECK = []


def smooth(y, k=41):
    ker = np.ones(k) / k
    return np.convolve(np.pad(y, k // 2, mode="edge"), ker, mode="valid")[: len(y)]


def ax_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12.5, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12.5, labelpad=8)
    ax.grid(True, axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, labelsize=11.5)


def new_ax(w=11.8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.subplots_adjust(left=0.078, right=0.947, top=0.855, bottom=0.175)
    return fig, ax


def legend(ax, ncol=3):
    leg = ax.legend(loc="lower left", frameon=False, fontsize=12, ncol=ncol,
                    handlelength=2.4, columnspacing=1.8, bbox_to_anchor=(0.0, 1.005),
                    borderaxespad=0.0)
    for ln in leg.get_lines():
        ln.set_linewidth(2.6)


def end_label(ax, x, y, color):
    ax.annotate(f"{y[-1]:.2f}", xy=(x, y[-1]), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=12, va="center", ha="left", clip_on=False)


def plot_group(key, title):
    d = np.load(OUT / f"sweep_rslow_{key}.npz")
    t = d["t"] / 60.0
    m0 = d["t"] >= T_START
    pre = smooth(d["pre"])[m0] / ADC_PER_N
    old = smooth(d["out_0.60"])[m0] / ADC_PER_N
    new = smooth(d["out_0.25"])[m0] / ADC_PER_N

    # 绝对
    fig, ax = new_ax()
    ax.plot(t[m0], pre, color="#C3CCDD", lw=1.5, ls=(0, (5, 3)), label="原始信号", zorder=2)
    ax.plot(t[m0], old, color=MUTED, lw=1.9, label="r_slow_max = 0.60（现役）", zorder=3)
    ax.plot(t[m0], new, color=BLUE, lw=2.4, label="r_slow_max = 0.25", zorder=4)
    ax_style(ax, "持续按压时间（分钟）", "显示力值（N）")
    lo = min(old.min(), new.min())
    hi = max(old.max(), new.max())
    ax.set_ylim(lo - (hi - lo) * 0.14, hi + (hi - lo) * 0.26)
    ax.set_xlim(t[m0][0], t[m0][-1])
    legend(ax)
    end_label(ax, t[m0][-1], old, MUTED)
    end_label(ax, t[m0][-1], new, BLUE)
    fig.savefig(FIG / f"{key}_abs.png")
    CHECK.append(dict(fig=f"{key}_abs", ylim=[round(v, 2) for v in ax.get_ylim()]))
    plt.close(fig)

    # 相对起始
    o = old - old[:60].mean()
    n = new - new[:60].mean()
    fig, ax = new_ax()
    ax.axhline(0, color="#C9D4E8", lw=1.1, zorder=1)
    ax.plot(t[m0], o, color=MUTED, lw=1.9, label="r_slow_max = 0.60（现役）", zorder=3)
    ax.plot(t[m0], n, color=BLUE, lw=2.4, label="r_slow_max = 0.25", zorder=4)
    ax_style(ax, "持续按压时间（分钟）", "相对起始读数的偏差（N）")
    lo2 = min(o.min(), n.min())
    hi2 = max(o.max(), n.max())
    ax.set_ylim(lo2 - (hi2 - lo2) * 0.16, hi2 + (hi2 - lo2) * 0.30)
    ax.set_xlim(t[m0][0], t[m0][-1])
    legend(ax, 2)
    end_label(ax, t[m0][-1], o, MUTED)
    end_label(ax, t[m0][-1], n, BLUE)
    # 下探标注
    for y, c, name in ((o, MUTED, "0.60"), (n, BLUE, "0.25")):
        i = int(np.argmin(y))
        ax.annotate(f"{name} 下探 {abs(y[i]):.2f} N", xy=(t[m0][i], y[i]),
                    xytext=(0, -14), textcoords="offset points", ha="center",
                    fontsize=10.5, color=c)
    fig.savefig(FIG / f"{key}_rel.png")
    CHECK.append(dict(fig=f"{key}_rel", ylim=[round(v, 2) for v in ax.get_ylim()]))
    plt.close(fig)


def plot_synth():
    p = OUT / "synth_hold.npz"
    if not p.exists():
        return
    d = np.load(p)
    t = d["t"] / 60.0
    m0 = d["t"] >= T_START
    truth = d["truth"][m0] / ADC_PER_N
    old = (smooth(d["out_0.60"])[m0] - d["truth"][m0]) / ADC_PER_N
    new = (smooth(d["out_0.25"])[m0] - d["truth"][m0]) / ADC_PER_N
    fig, ax = new_ax()
    ax.axhline(0, color="#C9D4E8", lw=1.3, ls=(0, (4, 4)), zorder=1)
    ax.plot(t[m0], old, color=MUTED, lw=1.9, label="r_slow_max = 0.60（现役）", zorder=3)
    ax.plot(t[m0], new, color=BLUE, lw=2.4, label="r_slow_max = 0.25", zorder=4)
    ax_style(ax, "持续按压时间（分钟）", "输出相对真实载荷的误差（N）")
    lo = min(old.min(), new.min())
    hi = max(old.max(), new.max())
    ax.set_ylim(lo - (hi - lo) * 0.16, hi + (hi - lo) * 0.30)
    ax.set_xlim(t[m0][0], t[m0][-1])
    legend(ax, 2)
    ax.annotate("真实载荷恒定：理想输出 = 0", xy=(0.008, 0), xycoords=("axes fraction", "data"),
                xytext=(0, 7), textcoords="offset points", color=INK2, fontsize=11.5)
    end_label(ax, t[m0][-1], old, MUTED)
    end_label(ax, t[m0][-1], new, BLUE)
    fig.savefig(FIG / "synth_rel.png")
    CHECK.append(dict(fig="synth_rel", ylim=[round(v, 2) for v in ax.get_ylim()]))
    plt.close(fig)


def main():
    plot_group("thumb", "拇指指腹")
    plot_group("palm", "掌心")
    plot_synth()
    for f in sorted(FIG.glob("*.png")):
        print(f.name, f.stat().st_size)
    (OUT / "rslow_figure_check.json").write_text(
        json.dumps(CHECK, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
