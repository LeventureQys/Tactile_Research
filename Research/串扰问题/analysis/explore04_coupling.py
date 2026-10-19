# -*- coding: utf-8 -*-
"""第四步：成对耦合回归。

对同一行 / 同一列内的任意两格 (p, q)，用所有帧做"过原点最小二乘"：
    d[q] ≈ k * d[p]
若 k 小且拟合优度高（R^2 高），说明 q 的响应是被 p 的泄漏"抬起来"的，
k 即该方向上的串扰耦合系数。

只在 d[p] 足够大（信号区）时统计，避免噪声主导。
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
    T, N = data.shape
    base = np.empty_like(data, dtype=float)
    half = win // 2
    for n in range(N):
        col = data[:, n]
        for b in np.array_split(np.arange(T), max(1, T // half)):
            if len(b) == 0:
                continue
            lo, hi = max(0, b[0] - half), min(T, b[-1] + half + 1)
            base[b, n] = np.percentile(col[lo:hi], q)
    return base


def pair_fit(x: np.ndarray, y: np.ndarray, gate: float):
    """过原点 OLS: y ≈ k*x（只用在 x>gate 的帧上）。返回 (k, R2, n)。"""
    sel = x > gate
    if sel.sum() < 30:
        return np.nan, np.nan, int(sel.sum())
    xs, ys = x[sel], y[sel]
    k = float(xs @ ys / max(xs @ xs, 1e-9))
    resid = ys - k * xs
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / max(ss_tot, 1e-9)
    return k, r2, int(sel.sum())


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore04.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = rec.data - rolling_baseline(rec.data)
        np.clip(d, 0, None, out=d)
        frames = np.full((rec.n_frames, rec.rows, rec.cols), np.nan)
        idx = rec.index_map
        for r in range(rec.rows):
            for c in range(rec.cols):
                if idx[r, c] >= 0:
                    frames[:, r, c] = d[:, idx[r, c]]
        gate = 0.25 * np.nanpercentile(np.nanmax(frames, axis=(1, 2)), 99)
        print("=" * 110)
        print(f"{rec.label}  帧={rec.n_frames} gate={gate:.0f}")

        for axis, name in ((2, "横向(同行, 列间)"), (1, "纵向(同列, 行间)")):
            print(f"\n--- {name} 耦合系数 k（中位数）与拟合优度 R2（中位数），gate={gate:.0f} ---")
            n_src = rec.cols if axis == 2 else rec.rows
            kmat = np.full((n_src, n_src), np.nan)
            rmat = np.full((n_src, n_src), np.nan)
            nmat = np.zeros((n_src, n_src), dtype=int)
            for i in range(n_src):
                for j in range(n_src):
                    if i == j:
                        continue
                    ks, r2s = [], []
                    for other in range(rec.rows if axis == 2 else rec.cols):
                        a = (other, i) if axis == 2 else (i, other)
                        b = (other, j) if axis == 2 else (j, other)
                        if idx[a] < 0 or idx[b] < 0:
                            continue
                        k, r2, n = pair_fit(frames[:, a[0], a[1]], frames[:, b[0], b[1]], gate)
                        if np.isfinite(k) and r2 > 0.5:
                            ks.append(k); r2s.append(r2); nmat[i, j] = max(nmat[i, j], n)
                    if ks:
                        kmat[i, j] = np.median(ks)
                        rmat[i, j] = np.median(r2s)
            hdr = "      " + "".join(f"{j:>13d}" for j in range(n_src))
            print("源\\目标" + hdr)
            for i in range(n_src):
                cells = []
                for j in range(n_src):
                    if i == j:
                        cells.append(f"{'--':>13s}")
                    elif np.isnan(kmat[i, j]):
                        cells.append(f"{'.':>13s}")
                    else:
                        cells.append(f"{kmat[i,j]*100:6.1f}%/{rmat[i,j]:4.2f}".rjust(13))
                print(f"{i:5d}  " + "".join(cells))
            print("  (格式: 耦合k% / R2;  '.' = 无有效配对)")


if __name__ == "__main__":
    main()
