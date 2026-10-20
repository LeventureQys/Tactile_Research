# -*- coding: utf-8 -*-
"""第十步：新工况数据摸底。

新数据集（Alpha v1.0.6）与上一轮（v1.0.4）的差异未知，先做无假设摸底：
  布局 / 基线 / 峰值分布 / 事件形态 / 是否存在多点与大面积帧
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def heat(rec, y, width=6, mx=None):
    idx = rec.index_map
    if mx is None:
        vals = y[np.isfinite(y)]
        mx = float(np.nanmax(vals)) if vals.size else 1.0
    lines = []
    for r in range(rec.rows):
        row = []
        for c in range(rec.cols):
            v = y[r, c]
            row.append(" " * width if not np.isfinite(v) else f"{int(round(v)):>{width}}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore10_newdata.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        P = np.nanmax(frames, axis=(1, 2)) if frames.size else np.array([0.0])
        P = np.where(np.isfinite(P), P, 0.0)
        print("=" * 100)
        print(f"{rec.label}")
        print(f"  rows={rec.rows} cols={rec.cols} active={rec.n_active} "
              f"frames={rec.n_frames} dur={rec.elapsed[-1]-rec.elapsed[0]:.1f}s "
              f"fps={rec.n_frames/max(rec.elapsed[-1]-rec.elapsed[0],1e-9):.1f}")
        print("  布局:")
        for r in range(rec.rows):
            print("    " + "".join("x" if rec.grid[r, c] else "." for c in range(rec.cols)))
        print(f"  原始数据: min={rec.data.min():.0f} max={rec.data.max():.0f} "
              f"mean={rec.data.mean():.2f} 分位[1,5,50,95,99]="
              f"{np.percentile(rec.data, [1,5,50,95,99]).round(1)}")
        print(f"  帧峰值: p50={np.median(P):.0f} p95={np.percentile(P,95):.0f} "
              f"p99.5={np.percentile(P,99.5):.0f} max={P.max():.0f}")
        print(f"  非零帧占比 (P>50): {100*np.mean(P>50):.1f}%   (P>300): {100*np.mean(P>300):.1f}%")

        # 每通道静息水平与最大值
        print("  每通道 中位数/最大值:")
        idx = rec.index_map
        med = np.median(rec.data, axis=0)
        for r in range(rec.rows):
            row = []
            for c in range(rec.cols):
                k = idx[r, c]
                row.append("        ." if k < 0 else f"{med[k]:5.0f}/{rec.data[:,k].max():5.0f}")
            print("    " + " ".join(row))

        # 多点/大面积候选：统计每帧活跃行、活跃列数
        thr = max(50.0, 0.25 * np.percentile(P, 99.5))
        ar_list, ac_list = [], []
        multi = []
        for t in range(rec.n_frames):
            if P[t] < thr:
                continue
            f = frames[t]
            re = np.array([np.nansum(f[r, :]) / max(np.sum(rec.grid[r, :]), 1)
                           for r in range(rec.rows)])
            ce = np.array([np.nansum(f[:, c]) / max(np.sum(rec.grid[:, c]), 1)
                           for c in range(rec.cols)])
            ar = int(np.sum(re > 0.25 * np.nanmax(re)))
            ac = int(np.sum(ce > 0.25 * np.nanmax(ce)))
            ar_list.append(ar); ac_list.append(ac)
            if ar >= 2 and ac >= 2:
                multi.append(t)
        if ar_list:
            print(f"  超阈帧={len(ar_list)}  活跃行数分布={dict(zip(*[x.tolist() for x in np.unique(ar_list, return_counts=True)]))}"
                  f"  活跃列数分布={dict(zip(*[x.tolist() for x in np.unique(ac_list, return_counts=True)]))}")
            print(f"  多行多列(候选大面积)帧数={len(multi)}")
            for t in sorted(multi, key=lambda tt: -P[tt])[:3]:
                print(f"   --- 候选大面积 t={t} 峰值={P[t]:.0f}")
                print("   " + heat(rec, frames[t], mx=P[t]).replace("\n", "\n   "))

        # 最强峰值帧
        t = int(np.argmax(P))
        print(f"  --- 最强单帧 t={t} 峰值={P[t]:.0f}")
        print("   " + heat(rec, frames[t], mx=P[t]).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
