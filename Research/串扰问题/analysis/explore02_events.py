# -*- coding: utf-8 -*-
"""第二步探索：响应事件与单点触发形态。

思路：
  1) 逐通道稳健基线（低分位）→ 响应量 d = data - base（截断负值）
  2) 帧级峰值检测 → 峰值帧的阵列热图
  3) 统计"准单点"事件（少数通道响应）的形态：主点 + 串扰点分布
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_lib import iter_all_recordings, print_grid  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def baseline(data: np.ndarray) -> np.ndarray:
    """逐通道稳健基线：取 5% 分位与中位数的较小者再向下取整。"""
    p05 = np.percentile(data, 5, axis=0)
    med = np.median(data, axis=0)
    return np.minimum(p05, med)


def response(rec, base=None):
    if base is None:
        base = baseline(rec.data)
    d = rec.data - base
    np.clip(d, 0, None, out=d)
    return d


def heat_text(rec, frame_vals: np.ndarray, width: int = 6) -> str:
    """把一维响应帧铺成网格文本，用字符强度表示大小。"""
    g = rec.to_grid(frame_vals)
    mx = np.nanmax(g) if np.any(~np.isnan(g)) else 1.0
    lines = []
    for r in range(rec.rows):
        row = []
        for c in range(rec.cols):
            v = g[r, c]
            if np.isnan(v):
                row.append(" " * width)
            elif mx <= 0:
                row.append("0".rjust(width))
            else:
                frac = v / mx
                ch = "." if frac < 0.05 else ("123456789" [min(8, int(frac * 9))])
                row.append(f"{ch}{int(round(v)):>{width - 1}}")
        lines.append("|".join(row))
    return "\n".join(lines)


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore02.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        base = baseline(rec.data)
        d = response(rec, base)
        peak = d.max(axis=1)
        argp = d.argmax(axis=1)
        thr_single = 0.25 * np.percentile(peak, 99.5)
        # 响应比：次大点相对主点的比例
        order = np.sort(d, axis=1)
        top1 = order[:, -1]
        top2 = order[:, -2]
        top3 = order[:, -3]
        ratio2 = np.divide(top2, top1, out=np.zeros_like(top1), where=top1 > 1e-6)

        # 事件：帧级峰值的时间局部极大
        cand = np.where((peak > thr_single) &
                        (peak >= np.r_[peak[0], peak[:-1]]) &
                        (peak >= np.r_[peak[1:], peak[-1]]))[0]
        print("=" * 100)
        print(f"{rec.label}  frames={rec.n_frames} 基线均值={base.mean():.1f} "
              f"峰值p50={np.median(peak):.0f} p99={np.percentile(peak,99):.0f} max={peak.max():.0f}")
        print(f"准单点候选峰:{len(cand)}  top2/top1 分位: p10={np.percentile(ratio2,10):.2f} "
              f"p50={np.percentile(ratio2,50):.2f} p90={np.percentile(ratio2,90):.2f}")

        # 展示若干最强峰值帧
        show = cand[np.argsort(peak[cand])[::-1]][:6] if len(cand) else []
        for t in show:
            idx = rec.index_map
            print(f"\n--- t={rec.elapsed[t]:.2f}s frame={rec.frame_index[t]} 峰值={peak[t]:.0f} "
                  f"主点数据列={argp[t]} ---")
            hot_cells = [(r, c, d[t, idx[r, c]]) for r in range(rec.rows) for c in range(rec.cols)
                         if idx[r, c] >= 0 and d[t, idx[r, c]] > 0.15 * peak[t]]
            print("响应单元(>15%峰值): " + ", ".join(f"({r},{c})={v:.0f}" for r, c, v in hot_cells))
            print(heat_text(rec, d[t]))
        # 最"干净"的单点帧（top2/top1 最小且主点足够大）
        ok = np.where((top1 > thr_single) & (ratio2 < 0.35))[0]
        if len(ok):
            best = ok[np.argsort(ratio2[ok])[:3]]
            for t in best:
                idx = rec.index_map
                r0, c0 = np.unravel_index(np.argmax(rec.to_grid(d[t])), (rec.rows, rec.cols))
                print(f"\n=== 干净单点 t={rec.elapsed[t]:.2f}s 主点=({r0},{c0}) 峰值={top1[t]:.0f} "
                      f"次点比={ratio2[t]:.2f} 三点比={top3[t]/max(top1[t],1):.2f}")
                hot_cells = [(r, c, d[t, idx[r, c]]) for r in range(rec.rows) for c in range(rec.cols)
                             if idx[r, c] >= 0 and d[t, idx[r, c]] > 0]
                print("全部非零单元: " + ", ".join(f"({r},{c})={v:.0f}" for r, c, v in hot_cells))
                print(heat_text(rec, d[t]))


if __name__ == "__main__":
    main()
