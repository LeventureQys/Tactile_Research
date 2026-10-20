# -*- coding: utf-8 -*-
"""第十一步：新工况定量诊断。

重点回答三件事：
  1) 串扰规律（方向、比例、衰减）在新数据上是否仍然成立
  2) 新出现的"多行同时活跃"帧是什么形态：真实多点 / 跨行接触 / 长期背景接触
  3) 标定的静息水平(rest)是否被长期接触污染
"""

import io
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response_with_baseline, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def heat(rec, y, mx=None, width=7):
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
    sys.stdout = io.open(os.path.join(OUT, "explore11_newdata.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d, base = response_with_baseline(rec.data)
        frames = to_frames(rec, d)
        P = np.nanmax(frames, axis=(1, 2))
        P = np.where(np.isfinite(P), P, 0.0)
        idx = rec.index_map
        print("=" * 104)
        print(f"{rec.label}")
        print(f"  基线: min={base.min():.1f} max={base.max():.1f} 均值={base.mean():.2f}")
        print(f"  响应: p50={np.median(d):.1f} p95={np.percentile(d,95):.1f} max={d.max():.0f}")

        # ---- 1) 方向能量 ----
        thr = 0.3 * np.percentile(P, 99.5)
        row_e, col_e = [], []
        peak_pos = []
        for t in range(rec.n_frames):
            if P[t] < thr:
                continue
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            e_row = float(np.nansum(np.delete(f[r0, :], c0)))
            e_col = float(np.nansum(np.delete(f[:, c0], r0)))
            row_e.append(e_row / P[t]); col_e.append(e_col / P[t])
            peak_pos.append((r0, c0))
        print(f"  ▶ 方向能量 (n={len(row_e)}): 横向中位={np.median(row_e)*100:.1f}%  "
              f"纵向中位={np.median(col_e)*100:.1f}%  "
              f"=> 主方向={'横向' if np.median(row_e) >= np.median(col_e) else '纵向'}")

        # ---- 2) 串扰比例 vs 距离 ----
        direction = "row" if np.median(row_e) >= np.median(col_e) else "col"
        dist = defaultdict(list)
        for t in range(rec.n_frames):
            if P[t] < thr:
                continue
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            if direction == "row":
                for c in range(rec.cols):
                    if c != c0 and np.isfinite(f[r0, c]):
                        dist[abs(c - c0)].append(float(f[r0, c] / P[t]))
            else:
                for r in range(rec.rows):
                    if r != r0 and np.isfinite(f[r, c0]):
                        dist[abs(r - r0)].append(float(f[r, c0] / P[t]))
        s = "  ▶ 串扰/主点 随距离: " + " ".join(
            f"d{k}={np.median(v)*100:.1f}%(n={len(v)})" for k, v in sorted(dist.items()))
        print(s)

        # ---- 3) 多行活跃帧的时间分布 ----
        ar_series = np.zeros(rec.n_frames, dtype=int)
        ac_series = np.zeros(rec.n_frames, dtype=int)
        active = P >= thr
        for t in range(rec.n_frames):
            if not active[t]:
                continue
            f = frames[t]
            re = np.array([np.nansum(f[r, :]) / max(np.sum(rec.grid[r, :]), 1) for r in range(rec.rows)])
            ce = np.array([np.nansum(f[:, c]) / max(np.sum(rec.grid[:, c]), 1) for c in range(rec.cols)])
            ar_series[t] = int(np.sum(re > 0.25 * np.nanmax(re)))
            ac_series[t] = int(np.sum(ce > 0.25 * np.nanmax(ce)))
        multi = np.where((ar_series >= 2) & (ac_series >= 2))[0]
        print(f"  ▶ 多行多列帧={len(multi)} / 超阈帧={int(active.sum())}")
        if len(multi):
            segs = []
            s0 = prev = multi[0]
            for t in multi[1:]:
                if t - prev > 20:
                    segs.append((s0, prev)); s0 = t
                prev = t
            segs.append((s0, prev))
            print(f"     时间分段数={len(segs)}，最长段={max(b-a+1 for a,b in segs)}帧；"
                  f"前 10 段={segs[:10]}")
            # 形态分类：主点行最强的行 vs 次强行 的比值
            for t in [multi[len(multi)//2], multi[0], multi[-1]]:
                f = frames[t]
                re = np.array([np.nansum(f[r, :]) / max(np.sum(rec.grid[r, :]), 1) for r in range(rec.rows)])
                order = np.argsort(re)[::-1]
                r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
                print(f"     --- t={t} 峰值={P[t]:.0f} 主点=({r0},{c0})  "
                      f"最强两行={order[0]}({re[order[0]]:.0f}) / {order[1]}({re[order[1]]:.0f})  "
                      f"比={re[order[0]]/max(re[order[1]],1e-9):.2f}")
                print("     " + heat(rec, f, mx=P[t]).replace("\n", "\n     "))

        # ---- 4) rest 标定是否被污染 ----
        quiet = P <= np.percentile(P, 30)
        rest = np.median(d[quiet], axis=0) if quiet.any() else np.zeros(rec.n_active)
        print(f"  ▶ 静息帧(低峰值30%)上的逐通道响应中位数: "
              f"min={rest.min():.1f} p50={np.median(rest):.1f} p95={np.percentile(rest,95):.1f} max={rest.max():.1f}")
        top = np.argsort(rest)[::-1][:6]
        print("     最高的 6 个通道: " + ", ".join(
            f"({rec.index_map.flat[0] if False else ''}{np.unravel_index(np.argmax(idx == k), idx.shape)})={rest[k]:.1f}"
            for k in top))
        # 该通道持久占用程度：中位数占比
        med = np.median(d, axis=0)
        busy = med > 5 * max(np.median(rest), 1.0)
        if busy.any():
            print(f"     ⚠ 长期被占用(中位数 > 5×静息)的通道数={int(busy.sum())}: "
                  + ", ".join(f"{np.unravel_index(np.argmax(idx == k), idx.shape)}={med[k]:.0f}" for k in np.where(busy)[0][:12]))


if __name__ == "__main__":
    main()
