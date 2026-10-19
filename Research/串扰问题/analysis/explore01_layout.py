# -*- coding: utf-8 -*-
"""第一步探索：布局、基线、帧统计。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_lib import iter_all_recordings, print_grid  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"


def main():
    import io
    import sys as _s
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "explore01.txt")
    _s.stdout = io.open(out_path, "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        g = rec.grid
        print("=" * 72)
        print(f"{rec.label}  rows={rec.rows} cols={rec.cols} active={rec.n_active} "
              f"frames={rec.n_frames} dur={rec.elapsed[-1] - rec.elapsed[0]:.2f}s "
              f"fps={rec.n_frames / max(rec.elapsed[-1] - rec.elapsed[0], 1e-9):.1f}")
        print("布局掩码 (x=活跃, .=空):")
        for r in range(rec.rows):
            print("  " + "".join("x" if g[r, c] else "." for c in range(rec.cols)))
        d = rec.data
        print(f"全局: min={d.min():.0f} max={d.max():.0f} mean={d.mean():.2f}")
        # 每列统计
        med = np.median(d, axis=0)
        p99 = np.percentile(d, 99, axis=0)
        mx = d.max(axis=0)
        print("每单元中位数 / P99 / 最大值 (网格视图，顺序为数据列序):")
        idx = rec.index_map
        for r in range(rec.rows):
            row = []
            for c in range(rec.cols):
                k = idx[r, c]
                row.append("        ." if k < 0 else f"{med[k]:5.0f}/{p99[k]:5.0f}/{mx[k]:5.0f}")
            print("  " + " ".join(row))
        # 帧级：每帧非零偏离度
        base = med
        dev = d - base
        active_any = np.abs(dev).max(axis=1)
        thr = max(3.0, np.percentile(active_any, 50) * 3)
        nz = np.sum(active_any > thr)
        print(f"帧级峰值偏离: 中位={np.median(active_any):.1f} 阈值={thr:.1f} 超阈帧={nz}/{rec.n_frames}")


if __name__ == "__main__":
    main()
