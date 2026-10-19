# -*- coding: utf-8 -*-
"""第六步：串扰假设的因果检验。

对主方向上的高样本配对 (p -> q) 做三项检验：
  1. 时间同步性：互相关在 lag=0 是否最大（串扰是瞬时电学泄漏，无延迟）
  2. 线性度：E[y_q | y_p] 是否过原点线性（串扰是按比例泄漏）
  3. 独立响应：是否存在"y_q 显著而 y_p ≈ 0"的帧（若存在则 q 能独立触发，
     说明它是真实触点的一部分，不能当串扰处理）
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import (collect_single_point_frames, direction_energy,  # noqa: E402
                            response, to_frames)
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def pair_report(yp: np.ndarray, yq: np.ndarray, name: str, out=None):
    P = np.maximum(yp, 1e-9)
    hi = np.percentile(yp, 99.5)
    gate = 0.2 * hi
    sel = yp > gate
    if sel.sum() < 50:
        return
    k = float(yp[sel] @ yq[sel] / (yp[sel] @ yp[sel]))
    r = float(np.corrcoef(yp[sel], yq[sel])[0, 1])
    # lag 相关
    lags = {}
    for lag in range(-4, 5):
        a = yp
        b = np.roll(yq, lag)
        idx = slice(max(0, -lag), len(yp) - max(0, lag))
        lags[lag] = float(np.corrcoef(a[idx], b[idx])[0, 1])
    best_lag = max(lags, key=lambda k2: lags[k2])
    # 分箱线性度
    bins = np.linspace(gate, hi, 6)
    binned = []
    for i in range(len(bins) - 1):
        m = (yp >= bins[i]) & (yp < bins[i + 1])
        if m.sum() >= 10:
            binned.append((0.5 * (bins[i] + bins[i + 1]), float(np.median(yq[m])),
                           float(np.percentile(yq[m], 25)), float(np.percentile(yq[m], 75))))
    # 独立响应：y_q 大但 y_p 小
    q_hi = 0.5 * np.percentile(yq, 99.5)
    indep = ((yq > q_hi) & (yp < 0.1 * hi)).sum()
    qhi_cnt = (yq > q_hi).sum()
    line = (f"  {name:>18s} k={k*100:5.1f}% r={r:5.2f} bestLag={best_lag:+d} "
            f"| q独立触发帧={indep}/{max(qhi_cnt,1)} ({100*indep/max(qhi_cnt,1):4.1f}%) "
            f"| 分箱(主点->串扰): " +
            " ".join(f"{a:5.0f}->{b:5.1f}" for a, b, _, _ in binned))
    print(line, file=out)


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore06.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        sel, thr = collect_single_point_frames(rec, frames)
        e_row, e_col = direction_energy(rec, frames, sel)
        direction = "row" if e_row >= e_col else "col"
        print("=" * 118)
        print(f"{rec.label}: 主方向={'横向(同行不同列)' if direction=='row' else '纵向(同列不同行)'} "
              f"横向能量={e_row:.3f} 纵向能量={e_col:.3f}")
        idx = rec.index_map
        # 抽样配对：主方向上、样本充足的位置对
        from collections import defaultdict
        cnt = defaultdict(int)
        for t in sel:
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            src = c0 if direction == "row" else r0
            L = rec.cols if direction == "row" else rec.rows
            for j in range(L):
                if j == src:
                    continue
                q = f[r0, j] if direction == "row" else f[j, c0]
                if np.isfinite(q):
                    cnt[(src, j)] += 1
        pairs = [k for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]) if v >= 25][:10]
        # 收集每个源位置代表的通道序列（把所有行/列平均化）
        print("  [配对检验] 源->目标（沿主方向），样本来自所有行的同列配对合并")
        for (src, dst) in pairs:
            yp_all, yq_all = [], []
            outer = rec.rows if direction == "row" else rec.cols
            for other in range(outer):
                a = (other, src) if direction == "row" else (src, other)
                b = (other, dst) if direction == "row" else (dst, other)
                if idx[a] < 0 or idx[b] < 0:
                    continue
                yp_all.append(frames[:, a[0], a[1]])
                yq_all.append(frames[:, b[0], b[1]])
            if not yp_all:
                continue
            yp = np.concatenate(yp_all)
            yq = np.concatenate(yq_all)
            pair_report(yp, yq, f"{src}->{dst}", out=sys.stdout)

        # 典型事件的时间序列展示
        print("\n  [典型事件时间演变] 取帧级峰值 top1 事件附近 8 帧")
        P_all = np.nanmax(frames, axis=(1, 2))
        t0 = int(np.nanargmax(P_all))
        r0, c0 = np.unravel_index(np.nanargmax(frames[t0]), (rec.rows, rec.cols))
        print(f"  主点=({r0},{c0}) 峰值={P_all[t0]:.0f} t={t0}")
        for t in range(max(0, t0 - 3), min(rec.n_frames, t0 + 5)):
            f = frames[t]
            if direction == "row":
                vals = [f"{c}:{f[r0,c]:.0f}" for c in range(rec.cols) if np.isfinite(f[r0, c])]
                print(f"    t={t:6d} 行{r0}响应  " + " ".join(vals))
            else:
                vals = [f"{r}:{f[r,c0]:.0f}" for r in range(rec.rows) if np.isfinite(f[r, c0])]
                print(f"    t={t:6d} 列{c0}响应  " + " ".join(vals))


if __name__ == "__main__":
    main()
