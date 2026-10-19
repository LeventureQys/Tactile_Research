# -*- coding: utf-8 -*-
"""第十二步：新工况下的参数敏感性与判据有效性验证。

验证三件事：
  A) 串扰增强后，line_energy / line_other 这两个"线内保护"阈值是否还有余量
  B) 现有 area 判据(≥2 行且 ≥2 列)在新数据上放行了什么 —— 是真面状还是"主点+背景接触"
  C) 把 area 门槛提到 3 后，被放行的帧剩什么形态
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response_with_baseline, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def heat(rec, y, mx, width=7):
    idx = rec.index_map
    lines = []
    for r in range(rec.rows):
        row = []
        for c in range(rec.cols):
            v = y[r, c]
            row.append(" " * width if not np.isfinite(v) else f"{int(round(v)):>{width}}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def analyse(rec, frames, P, thr):
    """返回每帧的判据量：方向、两类保护的越限比例、活跃行列数。"""
    idx = rec.index_map
    out = []
    for t in range(rec.n_frames):
        if P[t] < thr:
            continue
        f = frames[t]
        r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
        pk = P[t]
        e_row = float(np.nansum(np.delete(f[r0, :], c0)))
        e_col = float(np.nansum(np.delete(f[:, c0], r0)))
        line = f[r0, :] if e_row >= e_col else f[:, c0]
        k0 = c0 if e_row >= e_col else r0
        others = np.delete(line, k0)
        others = others[np.isfinite(others)]
        s_other = float(np.sum(np.clip(others, 0, None)))
        m_other = float(np.max(others)) if others.size else 0.0
        re = np.array([np.nansum(f[r, :]) / max(np.sum(rec.grid[r, :]), 1) for r in range(rec.rows)])
        ce = np.array([np.nansum(f[:, c]) / max(np.sum(rec.grid[:, c]), 1) for c in range(rec.cols)])
        ar = int(np.sum(re > 0.25 * np.nanmax(re)))
        ac = int(np.sum(ce > 0.25 * np.nanmax(ce)))
        out.append(dict(t=t, r0=int(r0), c0=int(c0), peak=float(pk),
                        e_row=e_row, e_col=e_col,
                        le=s_other / pk, lo=m_other / pk, ar=ar, ac=ac))
    return out


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore12_params.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d, base = response_with_baseline(rec.data)
        frames = to_frames(rec, d)
        P = np.nanmax(frames, axis=(1, 2))
        P = np.where(np.isfinite(P), P, 0.0)
        thr = 0.25 * np.percentile(P, 99.5)
        stats = analyse(rec, frames, P, thr)
        if not stats:
            continue
        le = np.array([s["le"] for s in stats])
        lo = np.array([s["lo"] for s in stats])
        ar = np.array([s["ar"] for s in stats])
        ac = np.array([s["ac"] for s in stats])

        print("=" * 104)
        print(f"{rec.label}   超阈帧={len(stats)}")
        print(f"  A) 线内保护余量:")
        print(f"     line_energy (阈值3.0): p50={np.median(le):.2f} p95={np.percentile(le,95):.2f} "
              f"p99={np.percentile(le,99):.2f} max={le.max():.2f}  → 超阈帧数={int(np.sum(le>3.0))}")
        print(f"     line_other  (阈值0.90): p50={np.median(lo):.2f} p95={np.percentile(lo,95):.2f} "
              f"p99={np.percentile(lo,99):.2f} max={lo.max():.2f}  → 超阈帧数={int(np.sum(lo>0.90))}")
        print(f"  B) 活跃行列分布: 行={dict(zip(*[x.tolist() for x in np.unique(ar, return_counts=True)]))}")
        print(f"                   列={dict(zip(*[x.tolist() for x in np.unique(ac, return_counts=True)]))}")
        a2 = int(np.sum((ar >= 2) & (ac >= 2)))
        a3 = int(np.sum((ar >= 3) & (ac >= 3)))
        print(f"     area判据命中: (行>=2 且 列>=2)={a2}   (行>=3 且 列>=3)={a3}")
        # 被 area 拦住但本可压制的帧：抽样看形态
        blocked = [s for s in stats if s["ar"] >= 2 and s["ac"] >= 2]
        if blocked:
            print(f"  C) 被(>=2,>=2)拦下的帧里，最典型的形态:")
            for s in sorted(blocked, key=lambda z: -z["peak"])[:2]:
                f = frames[s["t"]]
                print(f"     --- t={s['t']} 峰值={s['peak']:.0f} 主点=({s['r0']},{s['c0']}) "
                      f"行活跃={s['ar']} 列活跃={s['ac']} line_other={s['lo']:.2f} line_energy={s['le']:.2f}")
                print("     " + heat(rec, f, s["peak"]).replace("\n", "\n     "))
            # 这些帧里主点所在行是否明显是串扰形态
            print(f"     主点行非主点能量 / 主点 的中位: "
                  f"{np.median([s['le'] for s in blocked]):.2f}")


if __name__ == "__main__":
    main()
