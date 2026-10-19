# -*- coding: utf-8 -*-
"""输出对比：原始 / zero 模式 / subtract 模式 在数据域的数值是否落在静息水平上。"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_lib import iter_all_recordings, load_recording  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def dump(rec, t, idx, out):
    for r in range(rec.rows):
        out.write("  " + " ".join("      ." if idx[r, c] < 0 else "%7.0f" % rec.data[t, idx[r, c]]
                                  for c in range(rec.cols)) + "\n")


def main():
    sys.stdout = io.open(os.path.join(OUT, "check_output.txt"), "w", encoding="utf-8")
    recs = list(iter_all_recordings(RAW))
    rec_src = next(r for r in recs if "四指" in r.label)
    rec0 = load_recording(os.path.join(OUT, "test_zero.csv"), "zero")
    rec1 = load_recording(os.path.join(OUT, "test_sub.csv"), "subtract")
    idx = rec_src.index_map

    # 找一个主点最强的单点帧
    P = np.nanmax(rec_src.data, axis=1)
    t = int(np.argmax(P))
    print(f"== 案例帧 t={t} (原始峰值={P[t]:.0f}) ==")
    print("--- 原始 ---"); dump(rec_src, t, idx, sys.stdout)
    print("--- zero 模式 ---"); dump(rec0, t, idx, sys.stdout)
    print("--- subtract 模式 ---"); dump(rec1, t, idx, sys.stdout)

    print("\n== 数据域整体统计 ==")
    for tag, rec in (("原始", rec_src), ("zero", rec0), ("subtract", rec1)):
        d = rec.data
        print(f"  {tag:9s} 最小={d.min():6.1f} 均值={d.mean():7.2f} 分位[1,5,25,50]="
              f"{np.percentile(d, [1,5,25,50]).round(1)}")

    # 被压制位置的最终值分布（与原始静息值对比）
    print("\n== 压制位置恢复后的数值（应为静息底值，原始静息约 5~8）==")
    mask = np.abs(rec1.data - rec_src.data) > 1e-6
    print(f"  被改动单元-帧数={mask.sum()}")
    if mask.sum():
        changed = rec1.data[mask]
        print(f"  改动后取值: 最小={changed.min():.1f} p25={np.percentile(changed,25):.1f} "
              f"中位={np.median(changed):.1f} p75={np.percentile(changed,75):.1f} 最大={changed.max():.1f}")
        # 未被改动的静息值参考
        quiet = rec_src.data[rec_src.data < 20]
        print(f"  原始静息值(参考, <20 的部分): 最小={quiet.min():.1f} 中位={np.median(quiet):.1f} "
              f"p75={np.percentile(quiet,75):.1f}")

    # zero 模式对照
    mask0 = np.abs(rec0.data - rec_src.data) > 1e-6
    if mask0.sum():
        c0 = rec0.data[mask0]
        print(f"  [对照] zero 模式改动后取值: 最小={c0.min():.1f} 中位={np.median(c0):.1f} 最大={c0.max():.1f}")


if __name__ == "__main__":
    main()
