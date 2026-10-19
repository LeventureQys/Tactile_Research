# -*- coding: utf-8 -*-
"""串扰分析与压制核心库。

术语：
    * 主方向（detail axis）：阵列中串扰沿其传播的几何方向
        - 大拇指指腹：横向（同一行内、相邻列之间）
        - 四指指腹  ：纵向（同一列内、相邻行之间）
    * 主点 p：一帧内响应最大的单元（真实接触中心）
    * 串扰点 q：与 p 同处主方向线上、响应被 p 抬起来的单元

核心数据模型（线性泄漏假设）：
    y = x + K·x        =>   x = (I + K)^(-1) · y
其中 y 为观测响应，x 为真实接触响应，K 只在主方向上非零。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# 基线 / 响应
# --------------------------------------------------------------------------- #

def rolling_baseline(data: np.ndarray, win: int = 400, q: float = 5.0) -> np.ndarray:
    """逐通道滚动低分位基线，跟踪零点漂移。data (T,N) -> base (T,N)"""
    T, N = data.shape
    base = np.empty_like(data, dtype=float)
    half = max(1, win // 2)
    for n in range(N):
        col = data[:, n]
        for b in np.array_split(np.arange(T), max(1, T // half)):
            if len(b) == 0:
                continue
            lo, hi = max(0, b[0] - half), min(T, b[-1] + half + 1)
            base[b, n] = np.percentile(col[lo:hi], q)
    return base


def response(data: np.ndarray, win: int = 400, q: float = 5.0) -> np.ndarray:
    """观测响应 d = data - baseline（负值截断）。"""
    d = data - rolling_baseline(data, win, q)
    np.clip(d, 0, None, out=d)
    return d


def response_with_baseline(data: np.ndarray, win: int = 400, q: float = 5.0):
    """返回 (响应 d, 基线 base)。

    响应域与数据域必须成对使用：算法在响应域工作，写回时必须加回 base，
    否则会把每个通道的静息底值一并抹掉。
    """
    base = rolling_baseline(data, win, q)
    d = data - base
    np.clip(d, 0, None, out=d)
    return d, base


def restore_to_raw(y_resp: np.ndarray, base: np.ndarray) -> np.ndarray:
    """响应域 -> 数据域：data = response + baseline。"""
    return y_resp + base


def to_frames(rec, d: np.ndarray) -> np.ndarray:
    """(T, N) -> (T, rows, cols)，非活跃格 nan。"""
    out = np.full((d.shape[0], rec.rows, rec.cols), np.nan)
    idx = rec.index_map
    for r in range(rec.rows):
        for c in range(rec.cols):
            if idx[r, c] >= 0:
                out[:, r, c] = d[:, idx[r, c]]
    return out


# --------------------------------------------------------------------------- #
# 单点帧挑选
# --------------------------------------------------------------------------- #

def collect_single_point_frames(rec, frames: np.ndarray,
                                peak_q: float = 99.5, peak_ratio: float = 0.30,
                                orth_frac: float = 0.06, local_peak: bool = True,
                                return_peak: bool = False):
    """挑出"干净单点"帧。

    条件：
        1) 帧内峰值 >= peak_ratio × P99.5
        2) 峰值是时间局部极大（可选）
        3) 主点在横向与纵向上的伙伴响应都不超过 orth_frac × 峰值
           —— 确保该帧只有一处真实接触，其余响应只可能来自串扰。
    """
    T = frames.shape[0]
    P_all = np.nanmax(frames, axis=(1, 2))
    P_filled = np.where(np.isfinite(P_all), P_all, -np.inf)
    thr = peak_ratio * np.nanpercentile(P_all, peak_q)
    prev = np.r_[-np.inf, P_filled[:-1]]
    nxt = np.r_[P_filled[1:], -np.inf]
    sel = []
    for t in range(T):
        P = P_filled[t]
        if not np.isfinite(P) or P < thr:
            continue
        if local_peak and not (P >= prev[t] and P >= nxt[t]):
            continue
        r0, c0 = np.unravel_index(np.nanargmax(frames[t]), (rec.rows, rec.cols))
        colv = np.delete(frames[t, :, c0], r0)
        rowv = np.delete(frames[t, r0, :], c0)
        m_col = np.nanmax(colv) if np.any(np.isfinite(colv)) else 0.0
        m_row = np.nanmax(rowv) if np.any(np.isfinite(rowv)) else 0.0
        m_orth = min(m_col, m_row)
        if not np.isfinite(m_orth) or m_orth > orth_frac * P:
            continue
        sel.append(t)
    if return_peak:
        return sel, thr, P_filled
    return sel, thr


# --------------------------------------------------------------------------- #
# 方向判定
# --------------------------------------------------------------------------- #

def direction_energy(rec, frames: np.ndarray, sel) -> tuple[float, float]:
    """主点同行其余格之和 / 同列其余格之和（相对主点值），取中位数。

    返回 (横向能量, 纵向能量)；哪个大，就是串扰的主方向。
    """
    rs, cs = [], []
    for t in sel:
        f = frames[t]
        r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
        P = f[r0, c0]
        rs.append(np.nansum(np.delete(f[r0, :], c0)) / P)
        cs.append(np.nansum(np.delete(f[:, c0], r0)) / P)
    if not rs:
        return 0.0, 0.0
    return float(np.median(rs)), float(np.median(cs))


# --------------------------------------------------------------------------- #
# 耦合表估计
# --------------------------------------------------------------------------- #

def estimate_k_table(rec, frames: np.ndarray, sel, direction: str, min_n: int = 6):
    """估计串扰耦合比例表。

    direction='row'（横向）：源 = 主点列 c0，目标 = 同行其它列 c
    direction='col'（纵向）：源 = 主点行 r0，目标 = 同列其它行 r

    返回 dict:
        L          : 方向上的单元数
        overall    : (L, L)   全局中位数比例，overall[src, dst]
        q1, q3     : (L, L)   四分位（稳定性评估）
        n          : (L, L)   样本数
        by_line    : (n_line, L, L) 按主点所在行/列细分的比例（样本 >= min_n 才填）
        n_by_line  : (n_line, L, L)
        direction  : 传入值
    """
    if direction == "row":
        L, n_line = rec.cols, rec.rows
    else:
        L, n_line = rec.rows, rec.cols

    all_samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    line_samples: dict[tuple[int, int, int], list[float]] = defaultdict(list)

    for t in sel:
        f = frames[t]
        r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
        P = f[r0, c0]
        src = c0 if direction == "row" else r0
        line_idx = r0 if direction == "row" else c0
        for j in range(L):
            if j == src:
                continue
            q = f[r0, j] if direction == "row" else f[j, c0]
            if not np.isfinite(q):
                continue
            all_samples[(src, j)].append(float(q / P))
            line_samples[(line_idx, src, j)].append(float(q / P))

    overall = np.full((L, L), np.nan)
    q1 = np.full((L, L), np.nan)
    q3 = np.full((L, L), np.nan)
    n = np.zeros((L, L), dtype=int)
    for (src, j), v in all_samples.items():
        a = np.array(v)
        overall[src, j] = np.median(a)
        q1[src, j] = np.percentile(a, 25)
        q3[src, j] = np.percentile(a, 75)
        n[src, j] = len(a)

    by_line = np.full((n_line, L, L), np.nan)
    n_by_line = np.zeros((n_line, L, L), dtype=int)
    for (li, src, j), v in line_samples.items():
        n_by_line[li, src, j] = len(v)
        if len(v) >= min_n:
            by_line[li, src, j] = np.median(np.array(v))

    return dict(L=L, overall=overall, q1=q1, q3=q3, n=n, by_line=by_line,
                n_by_line=n_by_line, direction=direction)
