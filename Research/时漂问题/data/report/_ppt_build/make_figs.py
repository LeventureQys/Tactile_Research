"""Client-facing figures for the two-session drift-compensation comparison deck.

Every figure is written at a fixed canvas size (no tight-crop) and then checked
numerically: text/legend boxes must stay inside the canvas, and plotted data must
stay inside the axes limits. That check replaces eye inspection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import save_json  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

PRIMARY = "#1F4E79"
SENSOR = "#8A93A3"
ACCENT = "#C98A26"
FIX = "#2E7D5B"
REPORT = "#B23A32"
INK = "#1B2430"
LIGHT = "#DCE3EC"

M = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
CURVES = {k: np.load(HERE / f"{k}_curves.npz") for k in ("dev1", "dev2")}
GEOM: dict = {}
ISSUES: list = []


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
    return (edges[:-1] + edges[1:]) / 2.0, v


def style(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.5, color=INK, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.5, color=INK, labelpad=6)
    ax.grid(alpha=0.25, lw=0.8, color=LIGHT)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.tick_params(labelsize=10.5, colors="#4A5464")


def top_legend(ax, ncol, size=10.8):
    return ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=ncol, frameon=False,
                     fontsize=size, handlelength=2.2, columnspacing=2.6, borderaxespad=0.0)


def audit(fig, name):
    """Numeric layout audit: nothing may leave the canvas; data must fit axes."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.canvas.get_width_height()
    entries = []
    for ax in fig.axes:
        ab = ax.get_window_extent(rend)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for line in ax.lines:
            if line.get_transform() is not ax.transData:
                continue  # axvline / axhline live in axes coordinates
            yd = np.asarray(line.get_ydata(), dtype=float)
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = yd[np.isfinite(yd)]
            xd = xd[np.isfinite(xd)]
            tol = 1e-6 * max(1.0, abs(y1 - y0))
            if yd.size and yd.max() > y1 + tol:
                ISSUES.append(f"{name}: line above ylim ({yd.max():.2f} > {y1:.2f})")
            if xd.size and (xd.min() < x0 - 1e-6 or xd.max() > x1 + 1e-6):
                ISSUES.append(f"{name}: line data outside xlim ({xd.min():.9f}..{xd.max():.9f} vs {x0:.9f}..{x1:.9f})")
        for t in list(ax.texts):
            if not t.get_text():
                continue
            bb = t.get_window_extent(rend)
            entries.append({"text": t.get_text(), "axes": True,
                            "bbox": [round(bb.x0, 1), round(bb.y0, 1), round(bb.x1, 1), round(bb.y1, 1)],
                            "axes_bbox": [round(ab.x0, 1), round(ab.y0, 1), round(ab.x1, 1), round(ab.y1, 1)],
                            "inside_axes": bool(bb.x0 >= ab.x0 - 2 and bb.x1 <= ab.x1 + 2
                                                and bb.y0 >= ab.y0 - 2 and bb.y1 <= ab.y1 + 2),
                            "inside_canvas": bool(bb.x0 >= -1 and bb.x1 <= fw + 1
                                                  and bb.y0 >= -1 and bb.y1 <= fh + 1)})
        leg = ax.get_legend()
        if leg is not None:
            bb = leg.get_window_extent(rend)
            entries.append({"text": "LEGEND", "axes": False, "inside_axes": True,
                            "bbox": [round(bb.x0, 1), round(bb.y0, 1), round(bb.x1, 1), round(bb.y1, 1)],
                            "inside_canvas": bool(bb.x0 >= -1 and bb.x1 <= fw + 1
                                                  and bb.y0 >= -1 and bb.y1 <= fh + 1)})

    for leg in list(fig.legends):
        bb = leg.get_window_extent(rend)
        entries.append({"text": "FIG_LEGEND", "axes": False, "inside_axes": True,
                        "bbox": [round(bb.x0, 1), round(bb.y0, 1), round(bb.x1, 1), round(bb.y1, 1)],
                        "inside_canvas": bool(bb.x0 >= -1 and bb.x1 <= fw + 1
                                              and bb.y0 >= -1 and bb.y1 <= fh + 1)})
    for e in entries:
        if not e["inside_canvas"]:
            ISSUES.append(f"{name}: text leaves canvas -> {e['text'][:40]} | {e.get('bbox')}")
        elif not e["inside_axes"]:
            ISSUES.append(f"{name}: text leaves axes -> {e['text'][:40]} | {e.get('bbox')} | ax {e.get('axes_bbox')}")
    GEOM.setdefault(name, {})["window_audit"] = entries
    return fig


def save(fig, name):
    audit(fig, name)
    path = FIG / name
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    with Image.open(path).convert("RGB") as im:
        arr = np.asarray(im, dtype=np.int16)
    border = {
        "top": int((np.abs(arr[:2] - 255).sum(axis=2) > 36).sum()),
        "bottom": int((np.abs(arr[-2:] - 255).sum(axis=2) > 36).sum()),
        "left": int((np.abs(arr[:, :2] - 255).sum(axis=2) > 36).sum()),
        "right": int((np.abs(arr[:, -2:] - 255).sum(axis=2) > 36).sum()),
    }
    if any(border.values()):
        ISSUES.append(f"{name}: ink on canvas border {border}")
    GEOM.setdefault(name, {})["border_ink"] = border
    return name


# ── 图 1 / 图 3：全程对比 ───────────────────────────────────────────────────
FOCUS = {"s1": (14.0, 21.7), "s2": (10.6, 12.95)}
for sid, key, tag in (("s1", "dev1", "数据一"), ("s2", "dev2", "数据二")):
    d = CURVES[key]
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    tmax = float(t[-1])
    c, r = bin_mean(t, raw / 1000.0, tmax, 0.5)
    c = c / 60.0
    _, a = bin_mean(t, algo / 1000.0, tmax, 0.5)
    fig, ax = plt.subplots(figsize=(11.6, 4.75), layout="constrained")
    ax.fill_between(c, a, r, where=np.isfinite(r) & np.isfinite(a), color=ACCENT,
                    alpha=0.14, lw=0, label="算法自动补偿量")
    ax.plot(c, r, lw=1.5, color=SENSOR, label="补偿前（设备原始读数）")
    ax.plot(c, a, lw=2.0, color=PRIMARY, label="补偿后（输出给上位机）")
    it = M[sid]
    lo, hi = FOCUS[sid]
    tmin = tmax / 60.0
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, tmin * 1.34)
    ax.set_xticks(np.arange(0.0, np.floor(tmin) + 0.5, 2.0))
    for start, end, drift, color in (
            (it["raw"]["start_mN"] / 1000, it["raw"]["end_mN"] / 1000, it["raw"]["drift_mN"] / 1000, REPORT),
            (it["algo"]["start_mN"] / 1000, it["algo"]["end_mN"] / 1000, it["algo"]["drift_mN"] / 1000, FIX)):
        ax.plot([tmin * 1.03, tmin * 1.03], [start, end], color=color, lw=2.8,
                solid_capstyle="butt", clip_on=False)
        ax.plot([tmin * 1.03 - tmin * 0.012, tmin * 1.03 + tmin * 0.012], [end, end],
                color=color, lw=2.8, solid_capstyle="butt", clip_on=False)
        ax.plot([tmin * 1.03 - tmin * 0.012, tmin * 1.03 + tmin * 0.012], [start, start],
                color=color, lw=2.8, solid_capstyle="butt", clip_on=False)
        label = ("补偿前" if color is REPORT else "补偿后") + f"漂移 {drift:+.2f} N"
        ax.text(tmin * 1.075, (start + end) / 2, label, color=color, fontsize=11,
                fontweight="bold", ha="left", va="center")
    ax.text(0.008, 0.975, f"纵轴自 {lo:.1f} N 起（聚焦保压段；起始约 3 秒的加载过程位于此范围之下）",
            transform=ax.transAxes, fontsize=10.5, color="#5A6572", ha="left", va="top")
    style(ax, "持续保压时间（分钟）", "总作用力（N）")
    ax.spines["bottom"].set_bounds(0, tmin)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
    top_legend(ax, 3)
    name = f"fig_{sid}_full.png"
    save(fig, name)
    GEOM.setdefault(name, {}).update({
        "session": tag, "xlim_min": [0.0, tmin * 1.34], "ylim_n": [lo, hi],
        "raw_drift_n": it["raw"]["drift_mN"] / 1000, "algo_drift_n": it["algo"]["drift_mN"] / 1000,
    })

# ── 图 2：61 s 瞬时上抬局部放大（相对事件前基线的增量） ─────────────────────
d = CURVES["dev1"]
t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
base = (t >= M["s1"]["event_step"]["base_window_s"][0]) & (t <= M["s1"]["event_step"]["base_window_s"][1])
base_raw = M["s1"]["event_step"]["base_raw_mN"] / 1000.0
base_algo = M["s1"]["event_step"]["base_algo_mN"] / 1000.0
lo_t, hi_t = 52.0, 92.0
c, r = bin_mean(t, (raw / 1000.0 - base_raw), hi_t, 0.05)
_, a = bin_mean(t, (algo / 1000.0 - base_algo), hi_t, 0.05)
m = (c >= lo_t) & (c <= hi_t)
step_raw = M["s1"]["event_step"]["raw_mN"] / 1000.0
step_algo = M["s1"]["event_step"]["algo_mN"] / 1000.0
after_raw = (M["s1"]["event_step"]["after_raw_mN"] - M["s1"]["event_step"]["base_raw_mN"]) / 1000.0
after_algo = (M["s1"]["event_step"]["after_algo_mN"] - M["s1"]["event_step"]["base_algo_mN"]) / 1000.0
end_raw = float(np.interp(88.0, c, r))
end_algo = float(np.interp(88.0, c, a))
fig, ax = plt.subplots(figsize=(11.6, 4.75), layout="constrained")
ax.plot(c[m], r[m], lw=1.5, color=SENSOR, label="补偿前（设备原始读数）")
ax.plot(c[m], a[m], lw=2.0, color=PRIMARY, label="补偿后（输出给上位机）")
ax.axvline(61.5, color=ACCENT, lw=1.3, ls="--", alpha=0.9)
ax.axhline(0.0, color="#B6BFCA", lw=1.0, ls=":")
ax.set_xlim(lo_t, hi_t)
ax.set_ylim(min(-0.25, float(np.nanmin(a[m])) - 0.1, float(np.nanmin(r[m])) - 0.1),
            float(np.nanmax(r[m])) + 0.45)
ax.annotate(f"61.5 s：原始读数阶跃 +{step_raw:.2f} N", xy=(61.9, float(np.interp(61.9, c, r))),
            xycoords="data", xytext=(14, 12), textcoords="offset points",
            ha="left", va="bottom", fontsize=12, color=REPORT, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=REPORT, lw=1.4))
ax.annotate(f"同一瞬间：补偿后仅 +{step_algo:.2f} N", xy=(62.5, float(np.interp(62.5, c, a))),
            xycoords="data", xytext=(30, -30), textcoords="offset points",
            ha="left", va="top", fontsize=12, color=FIX, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=FIX, lw=1.4))
ax.annotate(f"事件后 30 s：原始读数累计抬升 +{after_raw:.2f} N，基线被永久抬高\n"
            f"补偿后保持 +{after_algo:.2f} N，不再继续漂移",
            xy=(85.0, end_raw), xycoords="data", xytext=(6, -46), textcoords="offset points",
            ha="right", va="top", fontsize=12, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
ax.text(0.008, 0.975, "纵轴为相对 61.5 s 前稳定读数（58–61.4 s 均值）的变化量",
        transform=ax.transAxes, fontsize=10.5, color="#5A6572", ha="left", va="top")
style(ax, "时间（秒）", "力值变化量（N）")
top_legend(ax, 2)
save(fig, "fig_s1_zoom61.png")
GEOM.setdefault("fig_s1_zoom61.png", {}).update({
    "window_s": [lo_t, hi_t], "ylim_n": list(ax.get_ylim()),
    "base_raw_n": base_raw, "base_algo_n": base_algo,
    "step_raw_n": step_raw, "step_algo_n": step_algo,
    "end_raw_n": end_raw, "end_algo_n": end_algo})

# ── 图 4：两组数据量化对比 ─────────────────────────────────────────────────
W = 0.34
labels = ["数据一（约 12 分钟保压）", "数据二（约 11 分钟保压）"]
raw_drift = [M["s1"]["raw"]["drift_mN"] / 1000, M["s2"]["raw"]["drift_mN"] / 1000]
algo_drift = [M["s1"]["algo"]["drift_mN"] / 1000, M["s2"]["algo"]["drift_mN"] / 1000]
rel_raw = [100 * M["s1"]["raw"]["drift_mN"] / M["s1"]["raw"]["start_mN"],
           100 * M["s2"]["raw"]["drift_mN"] / M["s2"]["raw"]["start_mN"]]
rel_algo = [100 * M["s1"]["algo"]["drift_mN"] / M["s1"]["algo"]["start_mN"],
            100 * M["s2"]["algo"]["drift_mN"] / M["s2"]["algo"]["start_mN"]]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
x = np.arange(2)
for ax, before, after, title, unit, bottom_pad in (
        (axes[0], raw_drift, algo_drift, "保压全过程力值漂移", "漂移量（N）", 0.5),
        (axes[1], rel_raw, rel_algo, "相对信号电平的漂移比例", "相对漂移（%）", 4.0)):
    b1 = ax.bar(x - W / 2, before, W, color=SENSOR, label="补偿前")
    b2 = ax.bar(x + W / 2, after, W, color=PRIMARY, label="补偿后")
    fmt = "{:+.2f} N" if unit.endswith("N）") else "{:+.1f}%"
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 6 if h >= 0 else -13),
                        ha="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.axhline(0, color="#8895A6", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.8)
    ax.set_ylim(min(min(after), 0.0) - bottom_pad, max(before) * 1.35)
    style(ax, None, unit)
fig.set_size_inches(11.6, 4.6)
fig.legend(handles=[Patch(facecolor=SENSOR, edgecolor="none", label="补偿前"),
                    Patch(facecolor=PRIMARY, edgecolor="none", label="补偿后")],
           loc="outside upper center", ncol=2, frameon=False, fontsize=11.5)
save(fig, "fig_compare.png")
GEOM.setdefault("fig_compare.png", {}).update({
    "raw_drift_n": raw_drift, "algo_drift_n": algo_drift,
    "rel_raw_pct": rel_raw, "rel_algo_pct": rel_algo})

# ── 图 5：瞬态与噪声稳定性 ────────────────────────────────────────────────
peak_labels = ["数据一 61.5 s 瞬时上抬", "数据二 全段最大瞬时波动"]
peak_raw = [M["s1"]["event_step"]["raw_mN"], M["s2"]["max_1s_rise"]["raw"]["rise_mN"]]
peak_algo = [M["s1"]["event_step"]["algo_mN"], M["s2"]["max_1s_rise"]["algo"]["rise_mN"]]
std_raw = [M["s1"]["raw"]["detrend_std_mN"], M["s2"]["raw"]["detrend_std_mN"]]
std_algo = [M["s1"]["algo"]["detrend_std_mN"], M["s2"]["algo"]["detrend_std_mN"]]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
for ax, before, after, title, unit, tlabels in (
        (axes[0], peak_raw, peak_algo, "瞬时跳变幅度", "跳变幅度（mN）", peak_labels),
        (axes[1], std_raw, std_algo, "保压段信号波动水平", "波动（mN）", ["数据一", "数据二"])):
    b1 = ax.bar(x - W / 2, before, W, color=SENSOR, label="补偿前")
    b2 = ax.bar(x + W / 2, after, W, color=PRIMARY, label="补偿后")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tlabels, fontsize=10.8)
    ax.set_ylim(0, max(before) * 1.35)
    style(ax, None, unit)
fig.set_size_inches(11.6, 4.6)
fig.legend(handles=[Patch(facecolor=SENSOR, edgecolor="none", label="补偿前"),
                    Patch(facecolor=PRIMARY, edgecolor="none", label="补偿后")],
           loc="outside upper center", ncol=2, frameon=False, fontsize=11.5)
save(fig, "fig_stability.png")
GEOM.setdefault("fig_stability.png", {}).update({
    "peak_raw_mN": peak_raw, "peak_algo_mN": peak_algo,
    "std_raw_mN": std_raw, "std_algo_mN": std_algo})

save_json(GEOM, HERE / "figure_geometry.json")
print("LAYOUT_ISSUES:", len(ISSUES))
for line in ISSUES:
    print(" -", line)
for name, g in GEOM.items():
    print(name, {k: v for k, v in g.items() if k != "window_audit"})
print("figures ->", FIG)
