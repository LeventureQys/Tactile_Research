# -*- coding: utf-8 -*-
"""第七步：固定主点位置下的串扰律（乘法型比例 vs 加法型偏置）+ 时间过程观察。

关键问题：串扰值 y_q 与主点值 y_p 的关系
    * 乘法型: y_q = k * y_p          （比例泄漏）
    * 加法型: y_q = c                （固定偏置，与力值无关）
这决定压制策略是"减比例"还是"减常数"。
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


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore07.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        P_all = np.nanmax(frames, axis=(1, 2))
        idx = rec.index_map
        # 找连续响应区间
        thr = 0.25 * np.nanpercentile(P_all, 99.5)
        on = np.where(np.isfinite(P_all) & (P_all > thr))[0]
        segs = []
        if len(on):
            s = on[0]
            prev = on[0]
            for t in on[1:]:
                if t - prev > 5:
                    segs.append((s, prev))
                    s = t
                prev = t
            segs.append((s, prev))
        segs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        print("=" * 108)
        print(f"{rec.label}  响应区间数={len(segs)}  最长区间长度={segs[0][1]-segs[0][0]+1 if segs else 0} "
              f"阈值={thr:.0f}")
        for (a, b) in segs[:3]:
            seg = frames[a:b + 1]
            if seg.shape[0] < 8:
                continue
            # 区间主点：出现次数最多的 argmax
            am = np.nanargmax(seg.reshape(seg.shape[0], -1), axis=1)
            rr, cc = np.unravel_index(am, (rec.rows, rec.cols))
            rr, cc = np.asarray(rr), np.asarray(cc)
            pos, cnt = np.unique(np.stack([rr, cc], 1), axis=0, return_counts=True)
            r0, c0 = pos[np.argmax(cnt)]
            if idx[r0, c0] < 0:
                continue
            print(f"\n  [区间 {a}-{b} len={b-a+1}] 主点=({r0},{c0}) 占该区间 {cnt.max()}/{len(rr)} 帧")
            # 该区间时间序列（降采样）
            step = max(1, (b - a + 1) // 24)
            head = "    t      P"
            qs = [q for q in range(rec.cols) if np.isfinite(frames[a, r0, q]) and q != c0] \
                if True else []
            qs_col = [r for r in range(rec.rows) if np.isfinite(frames[a, r, c0]) and r != r0]
            print(head + "".join(f"|c{q}" for q in qs) + " ||" + "".join(f"|r{r}" for r in qs_col))
            for t in range(a, b + 1, step):
                f = frames[t]
                line = f"  {t:6d} {f[r0,c0]:6.0f}"
                line += "".join(f"|{f[r0,q]:5.0f}" for q in qs)
                line += " ||" + "".join(f"|{f[r,c0]:5.0f}" for r in qs_col)
                print(line)

        # 固定位置的比例稳定性
        print("\n  [固定位置] 主点值分箱 -> 串扰中位数（横向样本）")
        sel_pos = {}
        for t in range(frames.shape[0]):
            P = P_all[t]
            if not np.isfinite(P) or P < thr:
                continue
            r0, c0 = np.unravel_index(np.nanargmax(frames[t]), (rec.rows, rec.cols))
            sel_pos.setdefault((r0, c0), []).append(t)
        top = sorted(sel_pos.items(), key=lambda kv: -len(kv[1]))[:6]
        for (r0, c0), ts in top:
            ts = np.array(ts)
            yp = frames[ts, r0, c0]
            order = np.argsort(yp)
            yp_s, ts_s = yp[order], ts[order]
            nb = 5
            chunks = np.array_split(np.arange(len(ts_s)), nb)
            print(f"    主点({r0},{c0}) n={len(ts)}")
            for ch in chunks:
                if len(ch) == 0:
                    continue
                lo, hi = ts_s[ch[0]], ts_s[ch[-1]]
                plo, phi = yp_s[ch[0]], yp_s[ch[-1]]
                row = f"      P∈[{plo:5.0f},{phi:5.0f}] "
                for c in range(rec.cols):
                    if c == c0 or idx[r0, c] < 0:
                        continue
                    row += f" c{c}={np.median(frames[ts[ch], r0, c]):5.0f}"
                for r in range(rec.rows):
                    if r == r0 or idx[r, c0] < 0:
                        continue
                    row += f" r{r}={np.median(frames[ts[ch], r, c0]):5.0f}"
                print(row)


if __name__ == "__main__":
    main()
