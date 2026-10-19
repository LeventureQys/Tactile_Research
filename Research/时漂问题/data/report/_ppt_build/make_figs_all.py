"""Figures for the six-dataset (three body parts) client deck."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import save_json  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
FIG = HERE / "figures_all"
FIG.mkdir(exist_ok=True)

PRIMARY = "#1F4E79"
SENSOR = "#8A93A3"
ACCENT = "#C98A26"
FIX = "#2E7D5B"
REPORT = "#B23A32"
INK = "#1B2430"
LIGHT = "#DCE3EC"

M = json.loads((HERE / "metrics_all.json").read_text(encoding="utf-8"))
ORDER = ["palm1", "palm2", "fing1", "fing2", "four1", "four2"]
PART_CN = {"palm": "右手掌", "fing": "右指腹", "four": "四指指腹"}
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


def style(ax, xlabel=None, ylabel=None, ticks=10.5):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=INK, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=INK, labelpad=4)
    ax.grid(alpha=0.25, lw=0.8, color=LIGHT, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.tick_params(labelsize=ticks, colors="#4A5464")


def top_legend(ax, ncol, size=9.8):
    return ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=ncol, frameon=False,
                     fontsize=size, handlelength=2.0, columnspacing=2.2, borderaxespad=0.0)


def audit(fig, name):
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
                continue
            yd = np.asarray(line.get_ydata(), dtype=float)
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd, xd = yd[np.isfinite(yd)], xd[np.isfinite(xd)]
            if yd.size and yd.max() > y1 + 1e-6 * max(1.0, abs(y1 - y0)):
                ISSUES.append(f"{name}: line above ylim ({yd.max():.2f} > {y1:.2f})")
            if xd.size and (xd.min() < x0 - 1e-6 or xd.max() > x1 + 1e-6):
                ISSUES.append(f"{name}: line outside xlim ({xd.max():.3f} vs {x1:.3f})")
        for t in list(ax.texts):
            if not t.get_text():
                continue
            bb = t.get_window_extent(rend)
            entries.append({"text": t.get_text()[:36],
                            "bbox": [round(bb.x0, 1), round(bb.y0, 1), round(bb.x1, 1), round(bb.y1, 1)],
                            "axes_bbox": [round(ab.x0, 1), round(ab.y0, 1), round(ab.x1, 1), round(ab.y1, 1)],
                            "inside_axes": bool(bb.x0 >= ab.x0 - 2 and bb.x1 <= ab.x1 + 2
                                                and bb.y0 >= ab.y0 - 2 and bb.y1 <= ab.y1 + 2),
                            "inside_canvas": bool(bb.x0 >= -1 and bb.x1 <= fw + 1
                                                  and bb.y0 >= -1 and bb.y1 <= fh + 1)})
        leg = ax.get_legend()
        if leg is not None:
            bb = leg.get_window_extent(rend)
            entries.append({"text": "LEGEND", "bbox": [round(bb.x0, 1), round(bb.y0, 1),
                                                       round(bb.x1, 1), round(bb.y1, 1)],
                            "inside_axes": True,
                            "inside_canvas": bool(bb.x0 >= -1 and bb.x1 <= fw + 1
                                                  and bb.y0 >= -1 and bb.y1 <= fh + 1)})
    for leg in list(fig.legends):
        bb = leg.get_window_extent(rend)
        entries.append({"text": "FIG_LEGEND", "bbox": [round(bb.x0, 1), round(bb.y0, 1),
                                                       round(bb.x1, 1), round(bb.y1, 1)],
                        "inside_axes": True,
                        "inside_canvas": bool(bb.x0 >= -1 and bb.x1 <= fw + 1
                                              and bb.y0 >= -1 and bb.y1 <= fh + 1)})
    for e in entries:
        if not e["inside_canvas"]:
            ISSUES.append(f"{name}: text leaves canvas -> {e['text']} | {e['bbox']}")
        elif not e["inside_axes"]:
            ISSUES.append(f"{name}: text leaves axes -> {e['text']} | {e['bbox']} | ax {e['axes_bbox']}")
    GEOM.setdefault(name, {})["window_audit"] = entries
    return fig


def save(fig, name):
    audit(fig, name)
    path = FIG / name
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    with Image.open(path).convert("RGB") as im:
        arr = np.asarray(im, dtype=np.int16)
    border = {side: int((np.abs(sub - 255).sum(axis=2) > 36).sum()) for side, sub in
              (("top", arr[:2]), ("bottom", arr[-2:]), ("left", arr[:, :2]), ("right", arr[:, -2:]))}
    if any(border.values()):
        ISSUES.append(f"{name}: ink on canvas border {border}")
    GEOM.setdefault(name, {})["border_ink"] = border
    return name


# ── 1) 每组数据的扁平全程图 ─────────────────────────────────────────────────
PART_OF = {"palm": "右手掌", "fing": "右指腹", "four": "四指指腹"}
PART_KEYS = {"右手掌": ["palm1", "palm2"], "右指腹": ["fing1", "fing2"],
             "四指指腹": ["four1", "four2"]}


def settled_range(key):
    dd = np.load(HERE / f"{key}_curves.npz")
    tt, rr, aa = dd["t"], dd["raw_sum"], dd["algo_sum"]
    sel = tt >= M[key]["settled_from_s"]
    lo = min(np.nanmin(aa[sel]), np.nanmin(rr[sel])) / 1000.0
    hi = max(np.nanmax(rr[sel]), np.nanmax(aa[sel])) / 1000.0
    return lo, hi


# 同一部位的两组，量级接近时共用一套纵轴刻度，便于上下对照
PART_FOCUS: dict = {}
for part, ks in PART_KEYS.items():
    los, his = zip(*[settled_range(k) for k in ks])
    combined = max(his) - min(los)
    single = max(h - l for l, h in zip(los, his))
    PART_FOCUS[part] = (min(los), max(his)) if combined <= 1.30 * single else None

for key in ORDER:
    d = np.load(HERE / f"{key}_curves.npz")
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    it = M[key]
    tmax = float(t[-1])
    tmin = tmax / 60.0
    c, r = bin_mean(t, raw / 1000.0, tmax, 0.5)
    c = c / 60.0
    _, a = bin_mean(t, algo / 1000.0, tmax, 0.5)

    shared = PART_FOCUS[it["part"]]
    if shared is not None:
        lo, hi = shared
    else:
        lo, hi = settled_range(key)
    span = max(hi - lo, 0.2)
    lo -= 0.10 * span
    hi += 0.10 * span

    fig, ax = plt.subplots(figsize=(11.6, 2.75), layout="constrained")
    ax.fill_between(c, a, r, where=np.isfinite(r) & np.isfinite(a), color=ACCENT,
                    alpha=0.14, lw=0, label="算法自动补偿量")
    ax.plot(c, r, lw=1.3, color=SENSOR, label="补偿前（设备原始读数）")
    ax.plot(c, a, lw=1.7, color=PRIMARY, label="补偿后（输出给上位机）")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, tmin * 1.30)
    yspan = hi - lo
    ystep = 2.0 if yspan > 6 else (1.0 if yspan > 3 else 0.5)
    ax.set_ylim(lo, np.ceil(hi / ystep) * ystep)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    step = max(1.0, round(tmin / 6.0))
    ticks = list(np.arange(0.0, tmin + 1e-6, step))
    if tmin - ticks[-1] >= 0.6:
        ticks.append(round(tmin, 1))
    ax.set_xticks(ticks)
    for start, end, drift, color in (
            (it["raw"]["start_mN"] / 1000, it["raw"]["end_mN"] / 1000, it["raw"]["drift_mN"] / 1000, REPORT),
            (it["algo"]["start_mN"] / 1000, it["algo"]["end_mN"] / 1000, it["algo"]["drift_mN"] / 1000, FIX)):
        x = tmin * 1.03
        ax.plot([x, x], [start, end], color=color, lw=2.6, solid_capstyle="butt", clip_on=False)
        for y in (start, end):
            ax.plot([x - tmin * 0.012, x + tmin * 0.012], [y, y], color=color, lw=2.6,
                    solid_capstyle="butt", clip_on=False)
        label = ("补偿前漂移" if color is REPORT else "补偿后漂移") + \
                (" 0.00 N" if abs(drift) < 0.005 else f" {drift:+.2f} N")
        ax.text(tmin * 1.075, (start + end) / 2, label, color=color, fontsize=10.5,
                fontweight="bold", ha="left", va="center")
    style(ax, "持续保压时间（分钟）", "力值（N）", ticks=10)
    ax.spines["bottom"].set_bounds(0, tmin)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
    top_legend(ax, 3, size=9.8)
    name = f"fig_{key}_full.png"
    save(fig, name)
    GEOM.setdefault(name, {}).update({
        "key": key, "part": it["part"], "ylim_n": [float(lo), float(hi)],
        "raw_drift_n": it["raw"]["drift_mN"] / 1000, "algo_drift_n": it["algo"]["drift_mN"] / 1000})

# ── 2) 六组漂移对比（按部位分面，各自量程） ─────────────────────────────────
raw_drift = [M[k]["raw"]["drift_mN"] / 1000 for k in ORDER]
algo_drift = [M[k]["algo"]["drift_mN"] / 1000 for k in ORDER]
rel_raw = [100 * M[k]["raw"]["drift_mN"] / M[k]["raw"]["start_mN"] for k in ORDER]
rel_algo = [100 * M[k]["algo"]["drift_mN"] / M[k]["algo"]["start_mN"] for k in ORDER]


def bar_label(v):
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}"


W = 0.34
PANELS = [("右手掌", ["palm1", "palm2"]), ("右指腹", ["fing1", "fing2"]),
          ("四指指腹", ["four1", "four2"])]
fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), layout="constrained")
for ax, (part, ks) in zip(axes, PANELS):
    idxs = [ORDER.index(k) for k in ks]
    xpos = np.arange(len(ks))
    before = [raw_drift[i] for i in idxs]
    after = [algo_drift[i] for i in idxs]
    b1 = ax.bar(xpos - W / 2, before, W, color=SENSOR)
    b2 = ax.bar(xpos + W / 2, after, W, color=PRIMARY)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(bar_label(h), (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 5 if h >= 0 else -14),
                        ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.axhline(0, color="#8895A6", lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["第 1 组", "第 2 组"], fontsize=11)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    top = max(before) * 1.22
    ax.set_ylim(min(min(after), 0.0) - top * 0.10 - 0.02, top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(part, fontsize=13.5, color=INK, fontweight="bold", pad=9)
    style(ax, None, "漂移量（N）")
fig.set_size_inches(11.6, 4.2)
fig.legend(handles=[Patch(facecolor=SENSOR, edgecolor="none", label="补偿前"),
                    Patch(facecolor=PRIMARY, edgecolor="none", label="补偿后")],
           loc="outside upper center", ncol=2, frameon=False, fontsize=11.5)
save(fig, "fig_all_drift.png")
GEOM.setdefault("fig_all_drift.png", {}).update({
    "raw_drift_n": raw_drift, "algo_drift_n": algo_drift,
    "rel_raw_pct": rel_raw, "rel_algo_pct": rel_algo,
    "panels": [p for p, _ in PANELS]})

# ── 3) 波动水平（按部位分面） ──────────────────────────────────────────────
std_raw = [M[k]["raw"]["detrend_std_mN"] for k in ORDER]
std_algo = [M[k]["algo"]["detrend_std_mN"] for k in ORDER]
event_raw = [max(M[k]["event"]["raw_1s_mN"], M[k]["event"]["step_raw_mN"]) for k in ORDER]
event_algo = [max(M[k]["event"]["algo_1s_mN"], M[k]["event"]["step_algo_mN"]) for k in ORDER]

fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), layout="constrained")
for ax, (part, ks) in zip(axes, PANELS):
    idxs = [ORDER.index(k) for k in ks]
    xpos = np.arange(len(ks))
    before = [std_raw[i] for i in idxs]
    after = [std_algo[i] for i in idxs]
    b1 = ax.bar(xpos - W / 2, before, W, color=SENSOR)
    b2 = ax.bar(xpos + W / 2, after, W, color=PRIMARY)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["第 1 组", "第 2 组"], fontsize=11)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylim(0, max(before) * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(part, fontsize=13.5, color=INK, fontweight="bold", pad=9)
    style(ax, None, "波动（mN）")
fig.set_size_inches(11.6, 4.2)
fig.legend(handles=[Patch(facecolor=SENSOR, edgecolor="none", label="补偿前"),
                    Patch(facecolor=PRIMARY, edgecolor="none", label="补偿后")],
           loc="outside upper center", ncol=2, frameon=False, fontsize=11.5)
save(fig, "fig_all_stability.png")
GEOM.setdefault("fig_all_stability.png", {}).update({
    "std_raw_mN": std_raw, "std_algo_mN": std_algo,
    "event_raw_mN": event_raw, "event_algo_mN": event_algo})

# ── 4) 右手掌 61.5 s 瞬时扰动（相对事件前基线的增量） ───────────────────────
d = np.load(HERE / "palm1_curves.npz")
t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
ev = M["palm1"]["event"]
base_raw = ev["base_raw_mN"] / 1000.0
base_algo = ev["base_algo_mN"] / 1000.0
lo_t, hi_t = 52.0, 92.0
c, r = bin_mean(t, raw / 1000.0 - base_raw, hi_t, 0.05)
_, a = bin_mean(t, algo / 1000.0 - base_algo, hi_t, 0.05)
m = (c >= lo_t) & (c <= hi_t)
step_raw, step_algo = ev["step_raw_mN"] / 1000.0, ev["step_algo_mN"] / 1000.0
after_raw = (ev["after_raw_mN"] - ev["base_raw_mN"]) / 1000.0
after_algo = (ev["after_algo_mN"] - ev["base_algo_mN"]) / 1000.0
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
top_legend(ax, 2, size=10.8)
save(fig, "fig_palm1_event.png")
GEOM.setdefault("fig_palm1_event.png", {}).update({
    "window_s": [lo_t, hi_t], "base_raw_n": base_raw, "base_algo_n": base_algo,
    "step_raw_n": step_raw, "step_algo_n": step_algo,
    "after_raw_n": after_raw, "after_algo_n": after_algo})

save_json(GEOM, HERE / "figure_geometry_all.json")
print("LAYOUT_ISSUES:", len(ISSUES))
for line in ISSUES:
    print(" -", line)
for name, g in GEOM.items():
    print(name, {k: v for k, v in g.items() if k != "window_audit"})
print("figures ->", FIG)
