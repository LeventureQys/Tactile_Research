"""甲方汇报用图（配色取自公司 logo：单色蓝 #155DFC + 近黑）。

输出 figures/*.png，全部数据来自 out/metrics.json 与 out/series_*.npz。
首分钟是加载瞬间（0 → 满载荷），图从 T_START 起画，只呈现恒载段的时漂表现。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- brand palette（logo.ico 采样：#155DFC / #000000）---
INK = "#0A1020"
INK2 = "#48587A"
BLUE = "#155DFC"
BLUE_DEEP = "#0B2E96"
BLUE_SOFT = "#8FB2FE"
MUTED = "#93A0B8"
GRID = "#E5EBF7"
PALE = "#F3F7FF"

ADC_PER_N = 1000.0  # 现场标定折算：1 N ≈ 1000 ADC

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 12,
    "axes.edgecolor": "#C9D4E8",
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.9,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

GROUPS = [("thumb", "拇指指腹"), ("palm", "掌心"), ("third", "四指指腹")]
LABEL_BEFORE = "修复前"
LABEL_AFTER = "修复后"
T_START = 60.0

# 力值图统一"全景"纵轴：宁可留白，也不让秒级噪声被放大成视觉抖动
YLIM_STEP = 5.0
YLIM_PAD = 5.0
Y_RANGE = [None, None]


def set_y_range(ranges):
    """由全部力值曲线的取值范围定一个共用纵轴（对齐到 5 N）。"""
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    lo = np.floor(lo / YLIM_STEP) * YLIM_STEP - YLIM_PAD
    hi = np.ceil(hi / YLIM_STEP) * YLIM_STEP
    Y_RANGE[0], Y_RANGE[1] = max(0.0, float(lo)), float(hi)
    return Y_RANGE

W_IN, H_IN = 11.8, 4.5
LEFT, RIGHT, TOP, BOTTOM = 0.078, 0.947, 0.855, 0.175

CHECK = []


def load(key):
    s = np.load(OUT / f"series_{key}.npz")
    return {k: s[k] for k in s.files}


def smooth(y, k):
    if k <= 1:
        return y
    ker = np.ones(k) / k
    pad = k // 2
    return np.convolve(np.pad(y, pad, mode="edge"), ker, mode="valid")[: len(y)]


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12.5, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12.5, labelpad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, labelsize=11.5)


def new_ax():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=200)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    return fig, ax


def legend(ax, **kw):
    leg = ax.legend(loc="lower left", frameon=False, fontsize=12, ncol=kw.pop("ncol", 2),
                    handlelength=2.4, columnspacing=1.8, bbox_to_anchor=(0.0, 1.005),
                    borderaxespad=0.0, **kw)
    for line in leg.get_lines():
        line.set_linewidth(2.6)
    return leg


def end_label(ax, x_end, y, color):
    ax.annotate(f"{y[-1]:.2f}", xy=(x_end, y[-1]), xytext=(6, 0),
                textcoords="offset points", color=color, fontsize=12,
                va="center", ha="left", clip_on=False)


def record(name, ax, series):
    CHECK.append(dict(fig=name, xlim=[round(float(v), 2) for v in ax.get_xlim()],
                      ylim=[round(float(v), 2) for v in ax.get_ylim()],
                      series={k: [round(float(np.min(v)), 2), round(float(np.max(v)), 2)]
                              for k, v in series.items()}))


def figs_abs(key, fname, win, show_input=True, show_before=True):
    """全程力值：修复前 vs 修复后（+ 原始信号参照）。"""
    d = load(key)
    m0 = d["t"] >= T_START
    t = d["t"][m0] / 60.0
    pre = smooth(d["pre"], 41)[m0] / ADC_PER_N
    before = smooth(d["out_MEAS"], 41)[m0] / ADC_PER_N
    after = smooth(d["out_FIX"], 41)[m0] / ADC_PER_N

    fig, ax = new_ax()
    if show_input:
        ax.plot(t, pre, color="#C3CCDD", lw=1.5, ls=(0, (5, 3)), zorder=2,
                label="原始信号（传感器输出）")
    if show_before:
        ax.plot(t, before, color=MUTED, lw=1.9, zorder=3, label=LABEL_BEFORE)
    ax.plot(t, after, color=BLUE, lw=2.4, zorder=4, label=LABEL_AFTER)

    w0, w1 = win[0] / 60.0, win[1] / 60.0

    tm = d["t"][m0]
    base = after[tm >= win[0]][:60].mean()
    if not show_before:
        ax.axhline(base, color="#C9D4E8", lw=1.3, ls=(0, (4, 4)), zorder=1)
        ax.annotate(f"起始基线 {base:.2f} N",
                    xy=(0.008, base), xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points", color=INK2, fontsize=11.5)

    style_axes(ax, "持续按压时间（分钟）", "显示力值（N）")
    if Y_RANGE[0] is not None:
        ax.set_ylim(Y_RANGE[0], Y_RANGE[1])
    else:
        src = [before, after] if show_before else [after, np.array([base])]
        ymin = min(a.min() for a in src)
        ymax = max(a.max() for a in src)
        pad = (ymax - ymin) * 0.14 or 0.2
        ax.set_ylim(ymin - pad, ymax + pad * 1.9)
    ax.set_xlim(t[0], t[-1])

    if show_before or show_input:
        legend(ax, ncol=int(show_before) + int(show_input))

    if show_before:
        end_label(ax, t[-1], before, MUTED)
    end_label(ax, t[-1], after, BLUE)

    out = FIG / fname
    fig.savefig(out)
    record(fname, ax, {"pre": pre, "before": before, "after": after})
    plt.close(fig)
    return out


def figs_drift(key, fname, win):
    """相对起始基线的力值漂移（时漂成果主图）。"""
    d = load(key)
    m0 = d["t"] >= T_START
    t = d["t"][m0] / 60.0
    mw = (d["t"] >= win[0]) & (d["t"] <= win[1])
    before = smooth(d["out_MEAS"], 41) / ADC_PER_N
    after = smooth(d["out_FIX"], 41) / ADC_PER_N
    before = (before - before[mw][:60].mean())[m0]
    after = (after - after[mw][:60].mean())[m0]

    fig, ax = new_ax()
    ax.axhline(0, color="#C9D4E8", lw=1.1, zorder=1)
    ax.plot(t, before, color=MUTED, lw=1.9, zorder=3, label=LABEL_BEFORE)
    ax.plot(t, after, color=BLUE, lw=2.4, zorder=4, label=LABEL_AFTER)

    style_axes(ax, "持续按压时间（分钟）", "力值漂移 ΔF（N）")
    lo = min(before.min(), after.min())
    hi = max(before.max(), after.max())
    pad = (hi - lo) * 0.16 or 0.15
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.set_xlim(t[0], t[-1])
    legend(ax, ncol=2)
    end_label(ax, t[-1], before, MUTED)
    end_label(ax, t[-1], after, BLUE)

    out = FIG / fname
    fig.savefig(out)
    record(fname, ax, {"before": before, "after": after})
    plt.close(fig)
    return out


def figs_summary(meta, fname):
    """两组数据的漂移量对比条形。"""
    keys = [k for k, _ in GROUPS if k in meta]
    labels = [meta[k]["label"] for k in keys]
    before = [meta[k]["arms"]["MEAS"]["out_drift"] / ADC_PER_N for k in keys]
    after = [meta[k]["arms"]["FIX"]["out_drift"] / ADC_PER_N for k in keys]

    fig, ax = plt.subplots(figsize=(11.2, 3.6), dpi=200)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.20)
    x = np.arange(len(keys))
    w = 0.28
    b1 = ax.bar(x - w / 2 - 0.012, before, w, color=MUTED, label=LABEL_BEFORE, zorder=3)
    b2 = ax.bar(x + w / 2 + 0.012, after, w, color=BLUE, label=LABEL_AFTER, zorder=3)
    for bars, vals, bold in ((b1, before, False), (b2, after, True)):
        for r, v in zip(bars, vals):
            ax.annotate(f"{v:+.2f} N", xy=(r.get_x() + r.get_width() / 2, v),
                        xytext=(0, 7 if v >= 0 else -18), textcoords="offset points",
                        ha="center", fontsize=13, color=INK,
                        fontweight="bold" if bold else "normal")
    ax.axhline(0, color="#C9D4E8", lw=1.1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13.5)
    style_axes(ax, "", "恒载段首尾力值漂移（N）")
    vals = before + after
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = hi - lo
    ax.set_ylim(lo - span * 0.10 if lo < 0 else -span * 0.16, hi + span * 0.24)
    ax.legend(loc="upper right", frameon=False, fontsize=12, ncol=2, handlelength=2.2)
    out = FIG / fname
    fig.savefig(out)
    record(fname, ax, {"before": before, "after": after})
    plt.close(fig)
    return out


def figs_fixonly(fname):
    """仅有修复后：各组在各自时长内的漂移曲线同框。"""
    fig, ax = new_ax()
    style = {"thumb": (BLUE, "-"), "palm": (BLUE_DEEP, "-"), "third": ("#5B8DEF", (0, (6, 3)))}
    plotted = {}
    x_end = 1.0
    for key, label in GROUPS:
        if not (OUT / f"series_{key}.npz").exists():
            continue
        d = load(key)
        m0 = d["t"] >= T_START
        t = d["t"][m0] / 60.0
        y = smooth(d["out_FIX"], 41)[m0] / ADC_PER_N
        y = y - y[:60].mean()
        color, ls = style.get(key, (MUTED, "-"))
        ax.plot(t, y, color=color, lw=2.4, ls=ls, zorder=3,
                label=f"{label}（{t[-1]:.0f} 分钟）")
        end_label(ax, t[-1], y, color)
        plotted[key] = y
        x_end = max(x_end, float(t[-1]))
    ax.axhline(0, color="#C9D4E8", lw=1.1, zorder=1)
    style_axes(ax, "持续按压时间（分钟）", "力值漂移 ΔF（N）")
    lo = min(v.min() for v in plotted.values())
    hi = max(v.max() for v in plotted.values())
    pad = (hi - lo) * 0.16 or 0.1
    ax.set_ylim(lo - pad, hi + pad * 1.6)
    ax.set_xlim(T_START / 60.0, x_end)
    legend(ax, ncol=len(plotted))
    out = FIG / fname
    fig.savefig(out)
    record(fname, ax, plotted)
    plt.close(fig)
    return out


def main():
    meta_path = OUT / "metrics.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    live = [(k, lbl) for k, lbl in GROUPS if k in meta]

    # 先用全部力值曲线定共用纵轴（全景）
    ranges = []
    for key, _ in live:
        d = load(key)
        m0 = d["t"] >= T_START
        ranges.append((float(smooth(d["pre"], 41)[m0].min()) / ADC_PER_N,
                       float(smooth(d["pre"], 41)[m0].max()) / ADC_PER_N))
        for arm in ("out_MEAS", "out_FIX"):
            y = smooth(d[arm], 41)[m0] / ADC_PER_N
            ranges.append((float(y.min()), float(y.max())))
    set_y_range(ranges)

    absnames = {"thumb": "A1_thumb_abs.png", "palm": "A3_palm_abs.png",
                "third": "A5_third_abs.png"}
    driftnames = {"thumb": "A2_thumb_drift.png", "palm": "A4_palm_drift.png",
                  "third": "A6_third_drift.png"}
    fixnames = {"thumb": "B1_thumb_fix_abs.png", "palm": "B2_palm_fix_abs.png",
                "third": "B3_third_fix_abs.png"}

    made = []
    for key, _ in live:
        w = meta[key]["arms"]["MEAS"]["window"]
        made.append(figs_abs(key, absnames[key], w))
        made.append(figs_drift(key, driftnames[key], w))
        made.append(figs_abs(key, fixnames[key], w, show_input=False, show_before=False))
    if live:
        made.append(figs_summary(meta, "A7_summary.png"))
    if live:
        made.append(figs_fixonly("B4_fixonly_compare.png"))
    for m in made:
        print(m.name, m.stat().st_size)
    print("y_range", Y_RANGE)
    (OUT / "figure_check.json").write_text(
        json.dumps(CHECK, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
