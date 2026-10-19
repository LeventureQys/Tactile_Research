# -*- coding: utf-8 -*-
"""Kelvin-Voigt 模型专题 · 数据演算与论文图件生成。

- 数据加载：dptool.api（数据解析工具对外稳定层）
- 图件渲染：dptool.snapshot 构造 + dptool.figure_export 导出（工具的冻结绘图契约）
- 全部数字汇总输出 numbers.json，供文章引用。

运行：python make_figures.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

TOOL = r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具"
sys.path.insert(0, TOOL)
from dptool import api                                    # noqa: E402
from dptool.snapshot import CurveSnapshot, FigureSnapshot, PanelSnapshot  # noqa: E402
from dptool.figure_export import export_png               # noqa: E402

SESSION = (r"D:\workshop\Github\CVResearch\Research\时漂-蠕变问题研究"
           r"\data\四指指腹\20260926_194738_single_device_c272fc")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figure"))
os.makedirs(FIG_DIR, exist_ok=True)

# 配色（沿用 dptool.merge.PALETTE 前 8 色）
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_BROWN, C_PINK, C_GRAY = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f")

NUM = {}   # 文章引用数字汇总


def save_fig(name, panels, ncol, suptitle, width=13.5, ph=2.9, dpi=150,
             legend_all=True):
    import time
    for i, p in enumerate(panels):
        if p.show_legend is None:
            p.show_legend = legend_all
    snap = FigureSnapshot(suptitle=suptitle, panels=panels, ncol=ncol,
                          width_in=width, panel_height_in=ph, dpi=dpi,
                          legend_panel=0, max_points_per_curve=8000)
    out = os.path.join(FIG_DIR, name)
    last = None
    for attempt in range(4):        # 瞬态文件锁重试（Errno 22 偶发）
        try:
            res = export_png(snap, out, check=True)
            print("saved %s  (%d bytes)" % (name, res["bytes"]))
            return out
        except OSError as e:
            last = e
            print("retry %d for %s: %s" % (attempt + 1, name, e))
            time.sleep(1.0 + attempt)
    raise last


# ────────────────────────── 数据加载（dptool） ──────────────────────────
table, failures = api.load([SESSION])
assert not failures, failures
T = table.x.astype(np.float64)
cols = table.groups["out"]
V = np.column_stack([table.col(c).astype(np.float64) for c in cols])
N_CH = V.shape[1]
TOTAL = V.sum(axis=1)
print("session loaded: %d frames, %d channels, %.1f s" % (len(T), N_CH, T[-1]))

# 分段边界（由总力与代表通道的细看确定）
T_LOAD = 1.0      # 加载沿开始
T_RAMP_END = 1.7  # 加载斜坡结束
T_SLOW0, T_SLOW1 = 5.0, 54.0   # 慢漂段（本文研究窗口）
T_UNLOAD = 87.9   # 整片卸载时刻
T_RELOAD = 89.9   # 重载时刻
T_END = float(T[-1])


def seg(t0, t1):
    m = (T >= t0) & (T < t1)
    return T[m], V[m], m


def kv_creep(x, vinf, A, tau):
    return vinf - A * np.exp(-x / tau)


# ────────────────────────── 逐通道慢漂拟合 ──────────────────────────
i0, i1 = np.searchsorted(T, T_SLOW0), np.searchsorted(T, T_SLOW1)
ts = T[i0:i1] - T_SLOW0
vs = V[i0:i1]

TAU_BOUND = 500.0
WIN = T_SLOW1 - T_SLOW0          # 49 s 慢漂窗口
TAU_SAT = WIN / math.log(10.0)   # 21.3 s：窗口内恰好爬满 90% 的 τ


def fit_channel(v):
    """多起点拟合，返回 (params, rms) —— 降低初值依赖。"""
    v0, vmax = v[0], v.max()
    best = None
    for tau0 in (2.0, 8.0, 30.0, 120.0):
        A0 = max(vmax - v0, 1.0) * 1.2
        try:
            p, _ = curve_fit(kv_creep, ts, v, p0=[v0 + A0, A0, tau0],
                             bounds=([v0, 0.0, 0.5], [np.inf, np.inf, TAU_BOUND]),
                             maxfev=40000)
            r = float(np.sqrt(np.mean((v - kv_creep(ts, *p)) ** 2)))
            if best is None or r < best[1]:
                best = (p, r)
        except Exception:
            pass
    return best


fits = []
for c in range(N_CH):
    v = vs[:, c]
    if v.mean() < 200:
        continue
    out = fit_channel(v)
    if out is None:
        print("ch%02d fit failed" % c)
        continue
    p, rms = out
    vinf, A, tau = map(float, p)
    resid = v - kv_creep(ts, *p)
    mb = T < 0.9
    noise = float(V[mb, c].std())
    rate_fit = A / tau
    k5 = np.searchsorted(ts, 5.0)
    rate_obs = float((v[k5] - v[0]) / 5.0)
    mid = 0.5 * (kv_creep(0.0, *p) + kv_creep(ts[-1], *p))
    idx = np.searchsorted(-kv_creep(ts, *p), -mid)
    t50 = float(ts[min(idx, len(ts) - 1)])
    tau_half = t50 / math.log(2.0)
    fits.append(dict(ch=c, v0=float(v[0]), vinf=vinf, A=A, tau=tau, rms=rms,
                     noise=noise, rate_fit=rate_fit, rate_obs=rate_obs,
                     tau_half=tau_half, bounded=tau >= TAU_BOUND * 0.999,
                     near_linear=tau > TAU_SAT))

fits.sort(key=lambda r: -r["A"])
rms_all_placeholder = None  # （分组见下方统计块）
rms_all = np.array([f["rms"] for f in fits])
noise_all = np.array([f["noise"] for f in fits])

NUM["n_active"] = len(fits)
NUM["n_at_bound"] = int(sum(f["bounded"] for f in fits))
NUM["tau_sat_criterion"] = TAU_SAT

# 三群划分（物理判据）：A<20 ADC 为无显著漂移；其余按 τ 是否超出窗口可辨识界
A_MIN = 20.0
grp_none = [f for f in fits if f["A"] < A_MIN]
grp_sat = [f for f in fits if f["A"] >= A_MIN and f["tau"] <= TAU_SAT]
grp_lin = [f for f in fits if f["A"] >= A_MIN and f["tau"] > TAU_SAT]
taus_sat = np.array([f["tau"] for f in grp_sat])
amps_sat = np.array([f["A"] / f["vinf"] for f in grp_sat])
NUM["A_min"] = A_MIN
NUM["n_none"] = len(grp_none)
NUM["n_sat"] = len(grp_sat)
NUM["n_lin"] = len(grp_lin)
NUM["tau_sat_median"] = float(np.median(taus_sat))
NUM["tau_sat_p25"] = float(np.percentile(taus_sat, 25))
NUM["tau_sat_p75"] = float(np.percentile(taus_sat, 75))
NUM["amp_ratio_sat_median"] = float(np.median(amps_sat))
NUM["amp_ratio_sat_range"] = [float(amps_sat.min()), float(amps_sat.max())]
NUM["lin_channels"] = [(f["ch"], round(f["tau"], 1), round(f["A"], 1)) for f in grp_lin]

rms_all = np.array([f["rms"] for f in fits])
noise_all = np.array([f["noise"] for f in fits])
NUM["rms_median"] = float(np.median(rms_all))
NUM["rms_max"] = float(rms_all.max())
NUM["noise_median"] = float(np.median(noise_all))
NUM["frac_rms_le_noise"] = float(np.mean(rms_all <= noise_all * 1.5))

# 代表通道（覆盖 可饱和/慢饱和/近线性 三种形态 + 最高幅度）
REP = [4, 6, 20, 7]  # tau≈9 / 17 / 94 / ≥500
rep_fits = {f["ch"]: f for f in fits}

# 拟合窗口平移稳健性（t0=6.5 重拟合，看 τ 稳不定）
TAU_SHIFT = {}
for c in REP:
    j0, j1 = np.searchsorted(T, 6.5), np.searchsorted(T, T_SLOW1)
    tt, vv = T[j0:j1] - 6.5, V[j0:j1, c]
    f0 = rep_fits[c]
    try:
        p, _ = curve_fit(kv_creep, tt, vv,
                         p0=[f0["vinf"], f0["A"] * 0.98, f0["tau"]],
                         bounds=([0, 0, 0.5], [np.inf, np.inf, TAU_BOUND]),
                         maxfev=40000)
        TAU_SHIFT[c] = float(p[2])
    except Exception:
        TAU_SHIFT[c] = float("nan")
NUM["tau_shift"] = TAU_SHIFT

# 留出检验：拟合 [5,40]，预测 [40,54]
k40 = np.searchsorted(ts, 35.0)
extr = []
for f in fits:
    c = f["ch"]
    try:
        p, _ = curve_fit(kv_creep, ts[:k40], vs[:k40, c],
                         p0=[f["vinf"], f["A"], f["tau"]],
                         bounds=([0, 0, 0.5], [np.inf, np.inf, TAU_BOUND]),
                         maxfev=60000)
        err = vs[k40:, c] - kv_creep(ts[k40:], *p)
        extr.append(dict(ch=c, bias=float(err.mean()),
                         absmax=float(np.abs(err).max()),
                         rel=float(err.mean() / max(f["A"], 1.0))))
    except Exception:
        pass
bias = np.array([e["bias"] for e in extr])
NUM["extr_bias_median"] = float(np.median(bias))
NUM["extr_bias_p90abs"] = float(np.percentile(np.abs(bias), 90))
NUM["extr_bias_maxabs"] = float(np.abs(bias).max())
NUM["extr_bias_rel_median"] = float(np.median([abs(e["rel"]) for e in extr]))

# 双时间常数对照（前 18 个通道）
def kv2(x, vinf, A1, tau1, A2, tau2):
    return vinf - A1 * np.exp(-x / tau1) - A2 * np.exp(-x / tau2)

ratios = []
for f in fits[:18]:
    c = f["ch"]
    v = vs[:, c]
    best = None
    for t20 in (3.0, 8.0, 25.0, 80.0, 250.0):
        try:
            p, _ = curve_fit(kv2, ts, v,
                             p0=[f["vinf"], f["A"] * .6, f["tau"], f["A"] * .4, t20],
                             bounds=([0, 0, 0.5, 0, 3.0], [np.inf, np.inf, 300, np.inf, 5000]),
                             maxfev=100000)
            r = float(np.sqrt(np.mean((v - kv2(ts, *p)) ** 2)))
            if best is None or r < best[1]:
                best = (p, r)
        except Exception:
            pass
    if best:
        ratios.append(best[1] / f["rms"])
NUM["dual_ratio_median"] = float(np.median(ratios))
NUM["dual_ratio_min"] = float(np.min(ratios))

# 总力拟合
ptot, _ = curve_fit(kv_creep, ts, TOTAL[i0:i1],
                   p0=[TOTAL[i1 - 1] + 500, 1200, 20.0],
                   bounds=([0, 0, 0.5], [np.inf, np.inf, 2000]), maxfev=60000)
rtot = float(np.sqrt(np.mean((TOTAL[i0:i1] - kv_creep(ts, *ptot)) ** 2)))
NUM["total_fit"] = dict(vinf=float(ptot[0]), A=float(ptot[1]), tau=float(ptot[2]),
                        rms=rtot, climb=float(TOTAL[i1 - 1] - TOTAL[i0]))

# 卸载窗口定量：dur−base 与 KV 保留预测
mbase = T < 0.9
base = V[mbase].mean(axis=0)
dur = V[(T >= 88.0) & (T < 89.5)].mean(axis=0)
pre = V[(T >= 85.0) & (T < 87.5)].mean(axis=0)
post = V[(T >= 95.0) & (T < 96.2)].mean(axis=0)
unload_gap = []
for f in fits[:12]:
    c = f["ch"]
    ret_pred = f["A"] * math.exp(-(T_RELOAD - T_UNLOAD) / f["tau"])
    unload_gap.append(dict(ch=c, obs=float(dur[c] - base[c]),
                           pred=float(ret_pred)))
NUM["unload_gap"] = unload_gap
NUM["unload_window_s"] = T_RELOAD - T_UNLOAD

# 加载沿瞬时跃升占比（以 t=1.7 s 电平对照 KV 阶跃响应）
load_frac = {}
for c in REP:
    f = rep_fits[c]
    mpre = T < 0.9
    vb = float(V[mpre, c].mean())
    mramp = (T >= T_RAMP_END) & (T < T_RAMP_END + 0.3)
    v_ramp = float(V[mramp, c].mean())
    climb_total = f["vinf"] - vb          # KV 口径的总爬升（渐近−基线）
    frac_obs = (v_ramp - vb) / climb_total
    frac_kv = 1.0 - math.exp(-(T_RAMP_END - T_LOAD) / f["tau"])
    load_frac[c] = dict(frac_obs=float(frac_obs), frac_kv=float(frac_kv))
NUM["load_frac"] = load_frac

# 重载段拟合（窗口短，仅报告形态一致性）
j0, j1 = np.searchsorted(T, 90.8), np.searchsorted(T, 96.2)
tj, vj = T[j0:j1] - 90.8, V[j0:j1]
reload_fit = {}
for c in REP:
    try:
        p, _ = curve_fit(kv_creep, tj, vj[:, c],
                         p0=[vj[-1, c] + 20, 60, 8.0],
                         bounds=([0, 0, 0.5], [np.inf, np.inf, 2000]),
                         maxfev=60000)
        reload_fit[c] = dict(A=float(p[1]), tau=float(p[2]),
                             start=float(p[0] - p[1]),
                             pre_unload=float(pre[c]), post=float(post[c]))
    except Exception:
        pass
NUM["reload_fit"] = reload_fit

# 半幅时间法与全曲线拟合的口径差（供附录参考）
rel_half = [abs(f["tau_half"] - f["tau"]) / f["tau"] for f in grp_sat]
NUM["half_dev_median"] = float(np.median(rel_half)) if rel_half else None

# ────────────────────────── 图 1：场景与慢漂段 ──────────────────────────
def vline(ax_x0, ax_x1, y0, y1, color, style="dash", width=0.9, label=None):
    return CurveSnapshot(name=label or "", x=np.array([ax_x0, ax_x0]),
                         y=np.array([y0, y1]), color=color, style=style,
                         width=width)


# 阶段色带定义：名称 / 起止 / 颜色（起止为 None 表示到数据末端/从 0 开始）
STAGES = [
    ("空载 0~1.0 s",              0.0,     T_LOAD,     C_GRAY),
    ("加载沿 1.0~1.7 s",          T_LOAD,  T_RAMP_END, C_BROWN),
    ("快相 1.7~5 s",              T_RAMP_END, T_SLOW0, C_GREEN),
    ("慢漂段 5~54 s（研究窗口）",  T_SLOW0, T_SLOW1,   C_RED),
    ("扰动段 54~87.9 s",          T_SLOW1, T_UNLOAD,   C_PURPLE),
    ("整片卸载 87.9~89.9 s",      T_UNLOAD, T_RELOAD,  C_PINK),
    ("重载段 89.9 s~末",          T_RELOAD, None,      C_ORANGE),
]


def stage_curves(y_ribbon, y_top):
    """顶部彩色阶段色带（粗横条，入图例）+ 同色边界竖线（不入图例）。
    返回 (色带曲线列表, 边界线曲线列表)。"""
    ribbons, marks = [], []
    for name, t0, t1, color in STAGES:
        x0 = t0 if t0 is not None else float(T[0])
        x1 = t1 if t1 is not None else float(T[-1])
        ribbons.append(CurveSnapshot(name=name, x=np.array([x0, x1]),
                                     y=np.array([y_ribbon, y_ribbon]),
                                     color=color, width=5.0))
    # 边界竖线：颜色取"由该时刻开始的阶段"的颜色
    for (name, t0, t1, color) in STAGES[1:]:
        marks.append(vline(t0, t0, 0.0, y_top, color, "dash", 0.8, None))
    return ribbons, marks


p1a = PanelSnapshot(
    title="(a) 全程总力与分段（顶部色带 = 阶段）", x_label="相对时间 t (s)",
    y_label="总力（通道求和, ADC）")
ytop_a = TOTAL.max() * 1.20
rib_a, mark_a = stage_curves(TOTAL.max() * 1.08, ytop_a)
p1a.curves.append(CurveSnapshot(name="总力", x=T, y=TOTAL, color=C_BLUE, width=0.9))
p1a.curves += rib_a + mark_a
p1a.ylim = (0.0, ytop_a)

ch4 = V[:, 4]
p1b = PanelSnapshot(
    title="(b) 代表通道 ch04 全程（同一套阶段色带）", x_label="相对时间 t (s)",
    y_label="读数 (ADC)")
ytop_b = ch4.max() * 1.28
rib_b, mark_b = stage_curves(ch4.max() * 1.13, ytop_b)
p1b.curves.append(CurveSnapshot(name="ch04", x=T, y=ch4, color=C_BLUE, width=0.9))
p1b.curves += rib_b + mark_b
p1b.ylim = (0.0, ytop_b)
save_fig("fig01_场景与慢漂段.png", [p1a, p1b], ncol=2,
         suptitle="恒载录制总览：加载沿—快相—慢漂段—卸载—重载（实测，100 Hz / 52 通道 / 96 s）")

# ────────────────────────── 图 2：解析形态（归一化） ──────────────────────────
tg = np.linspace(0.0, 6.0, 1201)
p2a = PanelSnapshot(title="(a) 蠕变解 J(t)·E/σ0 = 1−exp(−t/τ)",
                    x_label="t/τ", y_label="归一化应变")
p2a.curves += [
    CurveSnapshot(name="蠕变曲线 1−e^(−t/τ)", x=tg, y=1 - np.exp(-tg), color=C_BLUE, width=1.6),
    CurveSnapshot(name="初始切线 t/τ（斜率 σ0/η）", x=tg, y=tg, color=C_ORANGE, style="dash", width=1.2),
    CurveSnapshot(name="渐近线 1（σ0/E）", x=np.array([0, 6]), y=np.array([1, 1]),
                  color=C_GREEN, style="dot", width=1.2),
    CurveSnapshot(name="t=τ → 63.2%", x=np.array([1, 1]), y=np.array([0, 0.632]),
                  color=C_RED, style="dash", width=1.0),
    CurveSnapshot(name="t=3τ → 95.0%", x=np.array([3, 3]), y=np.array([0, 0.950]),
                  color=C_RED, style="dash", width=1.0),
    CurveSnapshot(name="", x=np.array([0, 1]), y=np.array([0.632, 0.632]),
                  color=C_RED, style="dash", width=1.0),
]
p2a.ylim = (0, 1.35)

p2b = PanelSnapshot(title="(b) 半对数直线化 ln[(v∞−v)/A] = −t/τ",
                    x_label="t/τ", y_label="ln((v∞−v)/A)")
for tau, col, lab in [(1.0, C_BLUE, "τ=1"), (2.0, C_ORANGE, "τ=2"), (4.0, C_GREEN, "τ=4")]:
    p2b.curves.append(CurveSnapshot(name=lab, x=tg, y=-tg / tau, color=col, width=1.4))
p2b.curves.append(CurveSnapshot(name="斜率 = −1/τ", x=np.array([0, 6]),
                                y=np.array([0, -6]), color=C_GRAY, style="dash", width=0.9))

p2c = PanelSnapshot(title="(c) 卸载恢复 ε(t)=ε1·exp(−t/τ)（与蠕变同 τ）",
                    x_label="t/τ", y_label="归一化残余应变")
p2c.curves += [
    CurveSnapshot(name="恢复 e^(−t/τ)", x=tg, y=np.exp(-tg), color=C_BLUE, width=1.6),
    CurveSnapshot(name="半衰期 τ·ln2 → 50%", x=np.array([math.log(2)] * 2),
                  y=np.array([0, 0.5]), color=C_RED, style="dash", width=1.0),
    CurveSnapshot(name="", x=np.array([0, math.log(2)]), y=np.array([0.5, 0.5]),
                  color=C_RED, style="dash", width=1.0),
    CurveSnapshot(name="3τ → 5.0%", x=np.array([3, 3]), y=np.array([0, math.exp(-3)]),
                  color=C_GREEN, style="dash", width=1.0),
]
p2c.ylim = (0, 1.1)

p2d = PanelSnapshot(title="(d) 应力松弛：KV 不松弛，Maxwell 指数松弛",
                    x_label="t/τ", y_label="σ/σ(0)")
p2d.curves += [
    CurveSnapshot(name="Kelvin-Voigt：σ = Eε0（常数）", x=np.array([0, 6]),
                  y=np.array([1, 1]), color=C_BLUE, width=1.6),
    CurveSnapshot(name="Maxwell：exp(−t/τ)", x=tg, y=np.exp(-tg),
                  color=C_ORANGE, style="dash", width=1.4),
]
p2d.ylim = (0, 1.15)
save_fig("fig02_解析形态.png", [p2a, p2b, p2c, p2d], ncol=2,
         suptitle="Kelvin-Voigt 模型的解析形态（归一化推导图，非实测）")

# ────────────────────────── 图 3：蠕变过程的阶段划分 ──────────────────────────
f4 = rep_fits[4]
tg3 = np.linspace(0.0, 6.0, 1201)
p3a = PanelSnapshot(title="(a) 蠕变解的阶段划分（归一化）", x_label="t/τ", y_label="归一化应变")
p3a.curves += [
    CurveSnapshot(name="蠕变曲线 1−e^(−t/τ)", x=tg3, y=1 - np.exp(-tg3), color=C_BLUE, width=1.6),
    CurveSnapshot(name="渐近线 1", x=np.array([0, 6]), y=np.array([1, 1]),
                  color=C_GREEN, style="dot", width=1.0),
    vline(1.0, 1.0, 0, 1.30, C_RED, "dash", 1.1, "阶段I→II：t=τ，完成 63.2%"),
    vline(3.0, 3.0, 0, 1.30, C_ORANGE, "dash", 1.1, "阶段II→III：t=3τ，完成 95.0%"),
    CurveSnapshot(name="", x=np.array([0, 1]), y=np.array([0.632, 0.632]),
                  color=C_RED, style="dot", width=0.9),
    CurveSnapshot(name="", x=np.array([0, 3]), y=np.array([0.950, 0.950]),
                  color=C_ORANGE, style="dot", width=0.9),
]
p3a.ylim = (0, 1.30)

p3b = PanelSnapshot(title="(b) 蠕变速率：KV 全程减速，无稳速段", x_label="t/τ",
                    y_label="归一化蠕变速率")
p3b.curves += [
    CurveSnapshot(name="Kelvin-Voigt：e^(−t/τ)（只有减速的一期）", x=tg3, y=np.exp(-tg3),
                  color=C_BLUE, width=1.6),
    CurveSnapshot(name="Maxwell：恒速（稳态流）", x=np.array([0, 6]), y=np.array([1, 1]),
                  color=C_GRAY, style="dash", width=1.3),
    vline(1.0, 1.0, 0, 1.15, C_RED, "dash", 0.9, "t=τ：速率衰减到初值 36.8%"),
]
p3b.ylim = (0, 1.15)

# (c) 实测通道全程的阶段标注（用慢漂段拟合的 τ 划分阶段边界）
fit_y = kv_creep(ts, f4["vinf"], f4["A"], f4["tau"])
t_s1 = T_SLOW0 + f4["tau"]          # 阶段I 末：63.2%
t_s2 = T_SLOW0 + 3.0 * f4["tau"]    # 阶段II 末：95%
mc = (T >= 0.5) & (T <= 55.0)
p3c = PanelSnapshot(title="(c) 实测全程的阶段标注（ch04，拟合 τ=%.1f s）" % f4["tau"],
                    x_label="t (s)", y_label="读数 (ADC)")
p3c.curves += [
    CurveSnapshot(name="实测", x=T[mc], y=V[mc, 4], color=C_GRAY, width=0.8),
    CurveSnapshot(name="KV 蠕变律拟合（慢漂段）", x=ts + T_SLOW0, y=fit_y,
                  color=C_RED, width=1.5),
    vline(T_RAMP_END, T_RAMP_END, 0, 1800, C_BROWN, "dash", 0.9, "沿末 t=1.7 s（瞬跳，模型外）"),
    vline(T_SLOW0, T_SLOW0, 0, 1800, C_GREEN, "dash", 0.9, "快相结束 t0=5 s（慢漂起点）"),
    vline(t_s1, t_s1, 0, 1800, C_RED, "dash", 0.9, "阶段I 末 t0+τ=%.1f s（63.2%%）" % t_s1),
    vline(t_s2, t_s2, 0, 1800, C_ORANGE, "dash", 0.9, "阶段II 末 t0+3τ=%.1f s（95%%）" % t_s2),
]
p3c.ylim = (0, 1800)
save_fig("fig03_蠕变阶段.png", [p3a, p3b, p3c], ncol=3,
         suptitle="蠕变过程的阶段划分：起步段—过渡段—饱和段（(a)(b) 推导图，(c) 实测标注）")

# ────────────────────────── 图 4：动态行为 ──────────────────────────
p4a = PanelSnapshot(title="(a) 谐波载荷下的滞后圈（应变-应力）",
                    x_label="应变 ε（归一化）", y_label="应力 σ（归一化）")
th = np.linspace(0, 2 * math.pi, 721)
for wt, col in [(0.2, C_BLUE), (1.0, C_ORANGE), (5.0, C_GREEN)]:
    eps = np.sin(th - math.atan(wt)) / math.sqrt(1 + wt * wt)
    sig = np.sin(th)
    p4a.curves.append(CurveSnapshot(name="ωτ=%.1f（滞后角 δ=atan ωτ）" % wt,
                                    x=eps, y=sig, color=col, width=1.3))
p4a.curves.append(CurveSnapshot(name="", x=np.array([0, 0]), y=np.array([-1, 1]),
                                color=C_GRAY, style="dot", width=0.8))

wx = np.logspace(-2, 2, 801)
p4b = PanelSnapshot(title="(b) 存储模量 E′、损耗模量 E″ 与损耗因子 tanδ",
                    x_label="ωτ（对数）", y_label="E′/E 、 E″/E 、 tanδ")
p4b.curves += [
    CurveSnapshot(name="E′/E = 1", x=wx, y=np.ones_like(wx), color=C_BLUE, width=1.4),
    CurveSnapshot(name="E″/E = ωτ", x=wx, y=wx, color=C_ORANGE, width=1.4),
    CurveSnapshot(name="tanδ = ωτ", x=wx, y=wx, color=C_GREEN, style="dash", width=1.4),
    CurveSnapshot(name="ωτ=1：损耗最大", x=np.array([1, 1]), y=np.array([1e-2, 1e2]),
                  color=C_RED, style="dash", width=1.0),
]
p4b.xlim = (1e-2, 1e2)
p4b.ylim = (1e-2, 1e2)
save_fig("fig04_动态行为.png", [p4a, p4b], ncol=2,
         suptitle="Kelvin-Voigt 模型的动态行为（推导图，非实测）")

# ────────────────────────── 图 5：慢漂段逐通道拟合 ──────────────────────────
panels4 = []
res_pan = PanelSnapshot(title="(e) 拟合残差（数据−模型）", x_label="t−t0 (s)",
                        y_label="残差 (ADC)")
lab_rep = {4: "a", 6: "b", 20: "c", 7: "d"}
for c in REP:
    f = rep_fits[c]
    v = vs[:, c]
    fitc = kv_creep(ts, f["vinf"], f["A"], f["tau"])
    tau_txt = ("≥%.0f s（窗口内不饱和）" % TAU_BOUND) if f["bounded"] else ("%.1f s" % f["tau"])
    p = PanelSnapshot(
        title="(%s) ch%02d：τ=%s，A=%.0f，RMS=%.1f（噪声 σ=%.1f）"
              % (lab_rep[c], c, tau_txt, f["A"], f["rms"], f["noise"]),
        x_label="t−t0 (s)", y_label="读数 (ADC)")
    p.curves += [
        CurveSnapshot(name="实测", x=ts, y=v, color=C_GRAY, width=0.8),
        CurveSnapshot(name="KV 蠕变律拟合", x=ts, y=fitc, color=C_RED, width=1.6),
    ]
    panels4.append(p)
    res_pan.curves.append(CurveSnapshot(
        name="ch%02d" % c, x=ts, y=v - fitc,
        color=[C_BLUE, C_ORANGE, C_GREEN, C_PURPLE][REP.index(c)], width=0.8))
res_pan.curves.append(CurveSnapshot(name="0", x=np.array([0, ts[-1]]),
                                    y=np.array([0, 0]), color="#000000",
                                    style="dot", width=0.8))
panels4.append(res_pan)
save_fig("fig05_慢漂段拟合.png", panels4, ncol=3,
         suptitle="慢漂段（t=5~54 s）逐通道拟合：v = v∞ − A·exp(−(t−t0)/τ)（实测+模型）")

# ────────────────────────── 图 6：群体统计 ──────────────────────────
def stems(entries, color, label, cap=None):
    """逐通道 τ 茎状图（NaN 断开）；cap 截断显示上限。"""
    xs, ys = [], []
    for i, f in enumerate(entries):
        top = f["tau"]
        if cap:
            top = min(top, cap)
        xs += [f["ch"], f["ch"], np.nan]
        ys += [0.0, top, np.nan]
    return CurveSnapshot(name=label, x=np.array(xs), y=np.array(ys),
                         color=color, width=1.1)

CAP = 100.0
p5a = PanelSnapshot(
    title="(a) 逐通道拟合 τ（虚线=窗口可辨识界 %.1f s；截断于 %.0f s；另有 %d 个通道 A<%d ADC 无显著漂移，不参与）"
          % (TAU_SAT, CAP, len(grp_none), A_MIN),
    x_label="通道号", y_label="拟合 τ (s)")
p5a.curves += [
    stems(grp_sat, C_BLUE, "可辨识饱和组（%d 个，中位 τ=%.1f s）" % (len(grp_sat), NUM["tau_sat_median"])),
    stems(grp_lin, C_ORANGE, "窗口内不饱和组（%d 个，拟合 τ 截断显示，实际 %.0f~%.0f s）"
          % (len(grp_lin), min(f["tau"] for f in grp_lin), max(f["tau"] for f in grp_lin))),
    CurveSnapshot(name="", x=np.array([0, 51]), y=np.array([TAU_SAT, TAU_SAT]),
                  color=C_RED, style="dash", width=1.0),
]
p5a.ylim = (0, CAP * 1.05)

order = np.argsort(rms_all)
p5b = PanelSnapshot(title="(b) 拟合残差 RMS 与基线噪声", x_label="通道（按 RMS 排序）",
                    y_label="ADC")
p5b.curves += [
    CurveSnapshot(name="拟合残差 RMS", x=np.arange(len(rms_all)),
                  y=rms_all[order], color=C_RED, width=1.4),
    CurveSnapshot(name="基线噪声 σ", x=np.arange(len(noise_all)),
                  y=noise_all[order], color=C_GRAY, style="dash", width=1.2),
    CurveSnapshot(name="1.5×噪声", x=np.arange(len(noise_all)),
                  y=noise_all[order] * 1.5, color=C_GREEN, style="dot", width=1.2),
]

bias_arr = np.array([e["bias"] for e in extr])
order_b = np.argsort(bias_arr)
p5c = PanelSnapshot(title="(c) 留出检验：前 35 s 拟合预测后 14 s 的偏差",
                    x_label="通道（按偏差排序）", y_label="平均偏差 (ADC)")
p5c.curves += [
    CurveSnapshot(name="预测偏差（正=实测高于模型）", x=np.arange(len(bias_arr)),
                  y=bias_arr[order_b], color=C_BLUE, width=1.4),
    CurveSnapshot(name="0", x=np.array([0, len(bias_arr) - 1]), y=np.array([0, 0]),
                  color="#000000", style="dot", width=0.8),
]
fig5_path = save_fig("fig06_群体统计.png", [p5a, p5b, p5c], ncol=3,
                     suptitle="38 个活跃通道的群体统计（慢漂段，实测）")

# ────────────────────────── 图 7：模型边界（实测反例） ──────────────────────────
f4 = rep_fits[4]
vb4 = float(V[T < 0.9, 4].mean())
# (a) 加载沿：实测归一化 vs KV 阶跃响应（τ 取慢漂段拟合值）
mz = (T >= 0.6) & (T <= 8.0)
climb = f4["vinf"] - vb4
obs_norm = (V[mz, 4] - vb4) / climb
kv_step = 1 - np.exp(-(T[mz] - T_LOAD) / f4["tau"])
kv_step = np.where(T[mz] < T_LOAD, 0.0, kv_step)
frac = NUM["load_frac"][4]
kv_ramp_norm = frac["frac_obs"] * np.ones_like(T[mz])   # 串联弹簧后的“瞬跳+蠕变”示意
kv_ramp_norm = np.where(T[mz] < T_RAMP_END,
                        frac["frac_obs"] * np.maximum(0.0, (T[mz] - T_LOAD)) / (T_RAMP_END - T_LOAD),
                        frac["frac_obs"] + (1 - frac["frac_obs"])
                        * (1 - np.exp(-(T[mz] - T_RAMP_END) / f4["tau"])))
p6a = PanelSnapshot(title="(a) 加载沿：KV 无瞬时弹性（ch04，归一化到渐近爬升）",
                    x_label="t (s)", y_label="归一化爬升")
p6a.curves += [
    CurveSnapshot(name="实测", x=T[mz], y=obs_norm, color=C_GRAY, width=0.9),
    CurveSnapshot(name="KV 阶跃响应（τ=%.1f s）" % f4["tau"], x=T[mz], y=kv_step,
                  color=C_RED, width=1.5),
    CurveSnapshot(name="串联瞬时弹性后的三参数示意", x=T[mz], y=kv_ramp_norm,
                  color=C_GREEN, style="dash", width=1.3),
]
p6a.ylim = (0, 1.15)

# (b) 卸载窗口：实测回零 vs KV 延迟保留
mu = (T >= 86.0) & (T <= 93.5)
ttx = T[mu]
mask_dur = (ttx >= T_UNLOAD) & (ttx < T_RELOAD)
mask_post = ttx >= T_RELOAD
pred_dur = vb4 + f4["A"] * np.exp(-(ttx - T_UNLOAD) / f4["tau"])
retained = f4["A"] * math.exp(-(T_RELOAD - T_UNLOAD) / f4["tau"])
# 重载后延迟分量从保留值重新生长到 A：x(t) = A − (A−retained)·e^{−(t−t_r)/τ}
# 拟合的 vinf 已含零载基线，故预测 = vinf − (A−retained)·e^{...}
pred_post = f4["vinf"] - (f4["A"] - retained) * np.exp(-(ttx - T_RELOAD) / f4["tau"])
p6b = PanelSnapshot(title="(b) 卸载即回零：实测不携带延迟分量（ch04）",
                    x_label="t (s)", y_label="读数 (ADC)")
p6b.curves += [
    CurveSnapshot(name="实测", x=ttx, y=V[mu, 4], color=C_GRAY, width=0.9),
    CurveSnapshot(name="KV 保留预测（卸载后按 τ 回落）", x=ttx[mask_dur],
                  y=pred_dur[mask_dur], color=C_RED, width=1.6),
    CurveSnapshot(name="KV 保留预测（重载后从保留态续爬）", x=ttx[mask_post],
                  y=pred_post[mask_post], color=C_RED, style="dash", width=1.4),
    vline(T_UNLOAD, T_UNLOAD, 0, 1800, C_GREEN, "dash", 0.9, "卸载"),
    vline(T_RELOAD, T_RELOAD, 0, 1800, C_GREEN, "dash", 0.9, "重载"),
]
p6b.ylim = (0, 1800)

# (c) 重载段：从低电平重新蠕变，而非从卸载前电平继续
mr = (T >= 89.0) & (T <= T_END)
p6c = PanelSnapshot(title="(c) 重载后从低电平重新蠕变（ch04）",
                    x_label="t (s)", y_label="读数 (ADC)")
p6c.curves += [
    CurveSnapshot(name="实测（重载段）", x=T[mr], y=V[mr, 4], color=C_GRAY, width=0.9),
    CurveSnapshot(name="卸载前电平（实测 %.0f）" % pre[4],
                  x=np.array([89.0, T_END]), y=np.array([pre[4], pre[4]]),
                  color=C_ORANGE, style="dot", width=1.3),
    CurveSnapshot(name="KV 保留预测（重载起点≈弹性+保留）", x=ttx[mask_post],
                  y=pred_post[mask_post], color=C_RED, style="dash", width=1.4),
    CurveSnapshot(name="KV 渐近电平 v∞−A=%.0f" % (f4["vinf"] - f4["A"]),
                  x=np.array([89.0, T_END]),
                  y=np.array([f4["vinf"] - f4["A"]] * 2),
                  color=C_GREEN, style="dot", width=1.2),
]
p6c.ylim = (0, 1800)
save_fig("fig07_模型边界.png", [p6a, p6b, p6c], ncol=3,
         suptitle="单 Kelvin-Voigt 的三条实测边界：无瞬跳 / 卸载即回零 / 重新蠕变")

# ────────────────────────── 数字汇总 ──────────────────────────
NUM["rep_fits"] = {str(c): rep_fits[c] for c in REP}
NUM["all_fits"] = fits
NUM["n_frames"] = int(len(T))
NUM["dur_s"] = float(T[-1])
NUM["fps_obs"] = float(len(T) / T[-1])
with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf-8") as fh:
    json.dump(NUM, fh, ensure_ascii=False, indent=1)
print("\n=== numbers.json 摘要 ===")
for k in ("n_active", "n_none", "n_sat", "n_lin", "n_at_bound", "tau_sat_criterion",
          "tau_sat_median", "tau_sat_p25", "tau_sat_p75", "amp_ratio_sat_median",
          "amp_ratio_sat_range", "lin_channels",
          "rms_median", "rms_max", "noise_median",
          "frac_rms_le_noise", "extr_bias_median", "extr_bias_p90abs",
          "extr_bias_rel_median", "dual_ratio_median", "half_dev_median",
          "total_fit", "tau_shift", "load_frac", "reload_fit"):
    print(k, "=", json.dumps(NUM[k], ensure_ascii=False))
print("unload_gap[:4] =", json.dumps(NUM["unload_gap"][:4], ensure_ascii=False))
