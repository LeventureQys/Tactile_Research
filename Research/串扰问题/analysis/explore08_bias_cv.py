# -*- coding: utf-8 -*-
"""第八步：串扰偏置的可预测性交叉验证。

这是判定"能否安全压制"的决定性实验：
    1) 用一半帧标定"每个主点位置 p 对应的其它位置的串扰偏置"（取低分位，
       避免把偶发真实触点算进偏置）
    2) 在另一半帧上检验：实际响应 y[q] 与预测偏置 bias 的残差
    3) 若残差 ~ 噪声底，说明该位置响应可由"主点位置"完全解释（= 串扰），
       压制安全；若残差显著大于噪声，说明存在真实第二触点，需要保留。

交叉验证按「事件」而非「帧」分割，避免同一按压事件同时进入标定与检验。
"""

import io
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def segments(P_all, thr, gap=5):
    on = np.where(np.isfinite(P_all) & (P_all > thr))[0]
    segs = []
    if not len(on):
        return segs
    s = prev = on[0]
    for t in on[1:]:
        if t - prev > gap:
            segs.append((s, prev))
            s = t
        prev = t
    segs.append((s, prev))
    return segs


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore08_bias_cv.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        idx = rec.index_map
        P_all = np.nanmax(frames, axis=(1, 2))
        thr = 0.25 * np.nanpercentile(P_all, 99.5)
        segs = segments(P_all, thr)
        # 以"事件"为单位划分：偶数为标定集，奇数为检验集
        cal_frames, test_frames = [], []
        for i, (a, b) in enumerate(segs):
            core = list(range(a, b + 1))
            (cal_frames if i % 2 == 0 else test_frames).extend(core)

        def pick(fs):
            """取事件内所有"响应超过阈值"的帧（保留足够样本量）。"""
            return [t for t in fs if np.isfinite(P_all[t]) and P_all[t] > thr]

        cal = pick(cal_frames)
        tst = pick(test_frames)
        # 噪声底：取"基本无触发"帧上的响应 99 分位
        low_frames = np.where(P_all < 0.15 * np.nanpercentile(P_all, 99.5))[0]
        noise = float(np.nanpercentile(frames[low_frames], 99)) if low_frames.size else 1.0

        print("=" * 112)
        print(f"{rec.label}: 事件数={len(segs)} 标定峰值帧={len(cal)} 检验峰值帧={len(tst)} "
              f"阈值={thr:.0f} 噪声99分位={noise:.1f}")

        # ---- 标定：按 (主点行, 主点列) 记录同行/同列其它位置的响应绝对值 ----
        # 偏置是"加法型"（与主点幅值弱相关），因此直接标定绝对值分布的低分位。
        bias_row = defaultdict(list)   # (r0, c0, c) -> values
        bias_col = defaultdict(list)   # (r0, c0, r) -> values
        for t in cal:
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            for c in range(rec.cols):
                if c == c0 or not np.isfinite(f[r0, c]):
                    continue
                bias_row[(r0, c0, c)].append(float(f[r0, c]))
            for r in range(rec.rows):
                if r == r0 or not np.isfinite(f[r, c0]):
                    continue
                bias_col[(r0, c0, r)].append(float(f[r, c0]))

        # ---- 检验 ----
        stat = defaultdict(lambda: dict(n=0, res_row=[], res_col=[], abs_row=[], abs_col=[],
                                        pred_row=[], pred_col=[]))
        detail = defaultdict(lambda: dict(n=0, res=[], raw=[]))
        for t in tst:
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            P = f[r0, c0]
            for c in range(rec.cols):
                if c == c0 or not np.isfinite(f[r0, c]):
                    continue
                v = bias_row.get((r0, c0, c))
                if not v or len(v) < 5:
                    continue
                pred = float(np.percentile(v, 40))
                residual = f[r0, c] - pred
                st = stat[(r0, c0)]
                st["n"] += 1
                st["res_row"].append(residual)
                st["abs_row"].append(f[r0, c])
                st["pred_row"].append(pred)
                d2 = detail[("row", r0, c0, c)]
                d2["n"] += 1
                d2["res"].append(residual)
                d2["raw"].append(f[r0, c])
            for r in range(rec.rows):
                if r == r0 or not np.isfinite(f[r, c0]):
                    continue
                v = bias_col.get((r0, c0, r))
                if not v or len(v) < 5:
                    continue
                pred = float(np.percentile(v, 40))
                residual = f[r, c0] - pred
                st = stat[(r0, c0)]
                st["n"] += 1
                st["res_col"].append(residual)
                st["abs_col"].append(f[r, c0])
                st["pred_col"].append(pred)
                d2 = detail[("col", r0, c0, r)]
                d2["n"] += 1
                d2["res"].append(residual)
                d2["raw"].append(f[r, c0])

        all_res_row = np.concatenate([np.array(v["res_row"]) for v in stat.values()]) if stat else np.array([])
        all_res_col = np.concatenate([np.array(v["res_col"]) for v in stat.values()]) if stat else np.array([])
        all_abs_row = np.concatenate([np.array(v["abs_row"]) for v in stat.values()]) if stat else np.array([])
        all_abs_col = np.concatenate([np.array(v["abs_col"]) for v in stat.values()]) if stat else np.array([])
        pred_row = np.concatenate([np.array(v["pred_row"]) for v in stat.values()]) if stat else np.array([])
        pred_col = np.concatenate([np.array(v["pred_col"]) for v in stat.values()]) if stat else np.array([])

        def desc(name, arr):
            if arr.size == 0:
                print(f"    {name}: 无样本")
                return
            print(f"    {name:26s} n={arr.size:7d} 均值={arr.mean():8.1f} "
                  f"p50={np.percentile(arr,50):7.1f} p90={np.percentile(arr,90):7.1f} "
                  f"|p90|={np.percentile(np.abs(arr),90):6.1f}")

        print("  同行(横向)方向:")
        desc("观测值 y_q", all_abs_row)
        desc("预测偏置 bias", pred_row)
        desc("残差 y_q - bias", all_res_row)
        print("  同列(纵向)方向:")
        desc("观测值 y_q", all_abs_col)
        desc("预测偏置 bias", pred_col)
        desc("残差 y_q - bias", all_res_col)

        # 残差是否可由噪声解释：比较残差的 |p90| 与噪声
        for nm, res, prd in (("横向", all_res_row, pred_row), ("纵向", all_res_col, pred_col)):
            if res.size == 0:
                continue
            r90 = np.percentile(np.abs(res), 90)
            ref = max(3 * noise, 0.05 * (np.percentile(prd, 90) if prd.size else 1.0))
            over = float(np.mean(res > ref))
            print(f"  → {nm}: 残差|p90|={r90:.1f} (噪声99分位={noise:.1f}, 比值={r90/max(noise,1e-9):.1f}) "
                  f"残差显著超出(>{ref:.0f})的占比={100*over:.1f}%")

        # 逐位置明细（取样本最多的几组）
        print("\n  [逐位置明细] 按样本数排序，取前 8 组")
        for key, v in sorted(detail.items(), key=lambda kv: -kv[1]["n"])[:8]:
            axis, r0, c0, j = key
            if v["n"] < 10:
                continue
            raw = np.array(v["raw"]); res = np.array(v["res"])
            print(f"    {axis} 主点({r0},{c0}) -> {j}: n={v['n']:4d} "
                  f"y_q p50={np.percentile(raw,50):6.1f} p90={np.percentile(raw,90):6.1f} | "
                  f"残差 p50={np.percentile(res,50):6.1f} p90={np.percentile(res,90):6.1f} "
                  f"超出占比={100*np.mean(res > 3*noise):5.1f}%")


if __name__ == "__main__":
    main()
