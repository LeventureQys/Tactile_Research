# -*- coding: utf-8 -*-
"""第五步：自适应方向判定 + 精确耦合表。

输出每个录制：
  * 横向 / 纵向能量 -> 自动判定串扰主方向
  * 主方向上的耦合比例矩阵（全局 + 按行/列细分），含样本数与四分位
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import (collect_single_point_frames, direction_energy,  # noqa: E402
                            estimate_k_table, response, to_frames)
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def fmt_matrix(mat, n=None, diag="   -- "):
    L = mat.shape[0]
    lines = []
    for i in range(L):
        cells = []
        for j in range(L):
            if i == j:
                cells.append(f"{diag:>13s}")
            elif not np.isfinite(mat[i, j]):
                cells.append(f"{'.':>13s}")
            elif n is None:
                cells.append(f"{mat[i,j]*100:6.1f}%".rjust(13))
            else:
                cells.append(f"{mat[i,j]*100:6.1f}%({n[i,j]})".rjust(13))
        lines.append(f"{i:3d}  " + "".join(cells))
    return "\n".join(lines)


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore05.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        sel, thr = collect_single_point_frames(rec, frames)
        e_row, e_col = direction_energy(rec, frames, sel)
        direction = "row" if e_row >= e_col else "col"
        print("=" * 118)
        print(f"{rec.label}: 帧={rec.n_frames} 单点帧={len(sel)} 峰值门槛={thr:.0f}  "
              f"横向能量={e_row:.3f} 纵向能量={e_col:.3f}  ==> 主方向="
              f"{'横向(同行, 列间)' if direction == 'row' else '纵向(同列, 行间)'}")

        kt = estimate_k_table(rec, frames, sel, direction)
        L = kt["L"]
        unit = "列" if direction == "row" else "行"
        print(f"\n--- 耦合比例 overall[{unit}源, {unit}目标] = E[ y_q / y_p ] (中位数, 括号为样本数) ---")
        print("源\\目标" + "".join(f"{j:>13d}" for j in range(L)))
        print(fmt_matrix(kt["overall"], n=kt["n"]))
        print("\n--- 四分位间距 IQR = (q3-q1) 绝对值，衡量比例稳定性（越小越像固定串扰）---")
        print(fmt_matrix(np.abs(kt["q3"] - kt["q1"])))
        print("\n--- 按主点所在" + ("行" if direction == "row" else "列") + "细分（样本>=6）---")
        bl, nb = kt["by_line"], kt["n_by_line"]
        for li in range(bl.shape[0]):
            if not np.any(np.isfinite(bl[li])):
                continue
            print(f"  [line {li}]  源{unit}->目标{unit} 比例(样本)")
            print("   " + fmt_matrix(bl[li], n=nb[li]).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
