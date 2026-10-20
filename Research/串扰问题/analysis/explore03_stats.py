# -*- coding: utf-8 -*-
"""第三步探索：单点事件的串扰定量统计。

对每个"干净单点帧"（主点显著、线外无响应）统计：
  * 同行（横向）响应 / 主点值
  * 同列（纵向）响应 / 主点值
并输出"主点位置 -> 目标位置"的串扰比例矩阵。
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rolling_baseline(data: np.ndarray, win: int = 400, q: float = 5.0) -> np.ndarray:
    """滚动低分位基线（逐通道），用于跟踪零点漂移。"""
    T, N = data.shape
    base = np.empty_like(data, dtype=float)
    half = win // 2
    for n in range(N):
        col = data[:, n]
        # 用小窗口分块 + 分位数，避免 O(T*win) 全量
        idx = np.arange(T)
        blocks = np.array_split(idx, max(1, T // half))
        for b in blocks:
            if len(b) == 0:
                continue
            lo = max(0, b[0] - half)
            hi = min(T, b[-1] + half + 1)
            v = np.percentile(col[lo:hi], q)
            base[b, n] = v
    return base


def analyze(rec, min_peak_ratio=0.35, outside_frac=0.06, verbose=True, fh=None):
    d = rec.data - rolling_baseline(rec.data)
    np.clip(d, 0, None, out=d)
    frames = np.full((rec.n_frames, rec.rows, rec.cols), np.nan)
    idx = rec.index_map
    for r in range(rec.rows):
        for c in range(rec.cols):
            if idx[r, c] >= 0:
                frames[:, r, c] = d[:, idx[r, c]]

    P_all = np.nanmax(frames, axis=(1, 2))
    peak_thr = 0.35 * np.nanpercentile(P_all, 99.5)
    events = []
    for t in range(rec.n_frames):
        P = P_all[t]
        if not np.isfinite(P) or P < peak_thr:
            continue
        r0, c0 = np.unravel_index(np.nanargmax(frames[t]), (rec.rows, rec.cols))
        f = frames[t]
        # 主点之外的最大值（排除同行同列）
        mask_out = np.ones((rec.rows, rec.cols), dtype=bool)
        mask_out[r0, :] = False
        mask_out[:, c0] = False
        m_out = np.nanmax(np.where(mask_out, f, np.nan))
        if not np.isfinite(m_out) or m_out > outside_frac * P:
            continue
        events.append((t, r0, c0, P, f.copy()))

    if verbose:
        print(f"\n{'#' * 100}")
        print(f"# {rec.label}  单点事件数={len(events)}  峰值阈值={peak_thr:.0f}  "
              f"帧数={rec.n_frames}")
    if not events:
        return events

    # 横向/纵向比例矩阵
    row_ratio = np.full((rec.rows, rec.cols, rec.cols), np.nan)
    col_ratio = np.full((rec.rows, rec.rows, rec.cols), np.nan)
    for t, r0, c0, P, f in events:
        for c in range(rec.cols):
            if c == c0 or not np.isfinite(f[r0, c]):
                continue
            row_ratio[r0, c0, c] = f[r0, c] / P
        for r in range(rec.rows):
            if r == r0 or not np.isfinite(f[r, c0]):
                continue
            col_ratio[r0, r, c0] = f[r, c0] / P

    if verbose:
        def show(mat3, axis_name, other_name):
            print(f"\n--- {axis_name}串扰比例：主点{other_name} → 目标{other_name} "
                  f"(中位数 | n>=3) ---")
            if axis_name == "横向":
                for c0 in range(rec.cols):
                    vals = mat3[:, c0, :]
                    line = []
                    for c in range(rec.cols):
                        v = vals[:, c]
                        v = v[np.isfinite(v)]
                        line.append("     ." if len(v) < 3 else f"{np.median(v)*100:5.1f}%({len(v)})")
                    print(f"  主点列{c0}: " + " ".join(line))
            else:
                for r0 in range(rec.rows):
                    line = []
                    for r in range(rec.rows):
                        vv = mat3[r0, r, :]
                        vv = vv[np.isfinite(vv)]
                        line.append("     ." if len(vv) < 3 else f"{np.median(vv)*100:5.1f}%({len(vv)})")
                    print(f"  主点行{r0}: " + " ".join(line))
        show(row_ratio, "横向", "列")
        show(col_ratio, "纵向", "行")

        # 主点值 -> 串扰量与比例关系
        print("\n--- 串扰绝对幅值 / 比例 vs 主点幅值（按主点值分档）---")
        Ps = np.array([e[3] for e in events])
        buckets = [(0, 400), (400, 800), (800, 1200), (1200, 1600), (1600, 4000)]
        for lo, hi in buckets:
            sel = [e for e in events if lo <= e[3] < hi]
            if not sel:
                continue
            row_r, col_r, row_a, col_a = [], [], [], []
            for t, r0, c0, P, f in sel:
                rr = [f[r0, c] / P for c in range(rec.cols) if c != c0 and np.isfinite(f[r0, c])]
                cc = [f[r, c0] / P for r in range(rec.rows) if r != r0 and np.isfinite(f[r, c0])]
                ra = [f[r0, c] for c in range(rec.cols) if c != c0 and np.isfinite(f[r0, c])]
                ca = [f[r, c0] for r in range(rec.rows) if r != r0 and np.isfinite(f[r, c0])]
                row_r.append(np.sum(rr)); col_r.append(np.sum(cc))
                row_a.append(np.sum(ra)); col_a.append(np.sum(ca))
            print(f"  主点[{lo},{hi}): n={len(sel):4d}  横向: 比值和={np.median(row_r)*100:5.1f}% "
                  f"幅值和={np.median(row_a):7.1f} | 纵向: 比值和={np.median(col_r)*100:5.1f}% "
                  f"幅值和={np.median(col_a):7.1f} | 主点中位={np.median([e[3] for e in sel]):6.0f}")
    return events


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore03.txt"), "w", encoding="utf-8")
    all_events = {}
    for rec in iter_all_recordings(RAW):
        ev = analyze(rec)
        all_events[rec.label] = len(ev)


if __name__ == "__main__":
    main()
