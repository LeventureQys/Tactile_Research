# -*- coding: utf-8 -*-
"""压制算法验证：模式统计、抑制率、大面积保护、案例对比。"""

import io
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402
from crosstalk_suppress import CrosstalkSuppressor, SuppressConfig, grid_to_row  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def heat(rec, y, width=6, mx=None):
    lines = []
    if mx is None:
        vals = y[np.isfinite(y)]
        mx = float(np.nanmax(vals)) if vals.size else 1.0
    for r in range(rec.rows):
        row = []
        for c in range(rec.cols):
            v = y[r, c]
            if not np.isfinite(v):
                row.append(" " * width)
            else:
                f = v / mx if mx > 0 else 0
                ch = "." if f < 0.02 else "123456789"[min(8, int(f * 9))]
                row.append(f"{ch}{int(round(v)):>{width-1}}")
        lines.append("|".join(row))
    return "\n".join(lines)


def main():
    sys.stdout = io.open(os.path.join(OUT, "verify_suppress.txt"), "w", encoding="utf-8")
    summary = []
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        cfg = SuppressConfig(direction="auto", compensate="keep")
        sup = CrosstalkSuppressor(rec, cfg)
        sup.calibrate(d)
        y_out, infos = sup.process_batch(d)
        modes = Counter(i["mode"] for i in infos)
        dirs = Counter(i["direction"] for i in infos if i["mode"] == "single")

        # 单点帧的抑制效果：主方向线上非主点能量
        before, after = [], []
        for t, i in enumerate(infos):
            if i["mode"] != "single":
                continue
            before.append(i["line_others_sum"])
            n = i["direction"]
            if n == "row":
                yo = y_out[t, i["r0"], :].copy()
                yo[i["c0"]] = np.nan
                after.append(np.nansum(yo))
            else:
                yo = y_out[t, :, i["c0"]].copy()
                yo[i["r0"]] = np.nan
                after.append(np.nansum(yo))
        before = np.array(before); after = np.array(after)

        # 大面积帧：检查是否被改动（应零改动）
        area_changed = 0
        for t, i in enumerate(infos):
            if i["mode"] != "area":
                continue
            a, b = frames[t], y_out[t]
            m = np.isfinite(a) & np.isfinite(b)
            if np.any(np.abs(a[m] - b[m]) > 1e-9):
                area_changed += 1

        print("=" * 110)
        print(f"{rec.label}  帧数={rec.n_frames}  噪声底={sup.noise:.1f}  参考量程={sup.ref_scale:.0f}")
        print(f"  模式分布: {dict(modes)}   单点方向: {dict(dirs)}")
        print(f"  单点帧抑制: 线上非主点能量 压制前中位={np.median(before) if len(before) else 0:.0f} "
              f"→ 压制后中位={np.median(after) if len(after) else 0:.0f}  "
              f"(平均抑制率={100*(1-after.sum()/max(before.sum(),1e-9)):.1f}%)")
        print(f"  大面积帧被改动数 = {area_changed} (应为 0)")
        summary.append(dict(label=rec.label, frames=rec.n_frames,
                            single=modes.get("single", 0), area=modes.get("area", 0),
                            low=modes.get("low", 0), ambig=modes.get("ambiguous", 0),
                            dir=dict(dirs),
                            supp_ratio=100 * (1 - after.sum() / max(before.sum(), 1e-9)) if len(before) else 0,
                            area_changed=area_changed))

        # ---- 案例对比：最强的单点帧 & 一场大面积帧 ----
        strong = [t for t, i in enumerate(infos) if i["mode"] == "single"]
        if strong:
            t = max(strong, key=lambda tt: infos[tt]["peak"])
            i = infos[t]
            print(f"\n  [案例·单点] t={t} frame={rec.frame_index[t]} 主点=({i['r0']},{i['c0']}) "
                  f"方向={i['direction']} 峰值={i['peak']:.0f} 压制点数={i['suppressed']}")
            print("   --- 原始 ---")
            print("   " + heat(rec, frames[t], mx=i["peak"]).replace("\n", "\n   "))
            print("   --- 压制后 ---")
            print("   " + heat(rec, y_out[t], mx=i["peak"]).replace("\n", "\n   "))

        areas = [t for t, i in enumerate(infos) if i["mode"] == "area"]
        if areas:
            t = max(areas, key=lambda tt: infos[tt]["peak"])
            print(f"\n  [案例·大面积/多点] t={t} 峰值={infos[t]['peak']:.0f} "
                  f"活跃行={infos[t]['active_rows']} 活跃列={infos[t]['active_cols']} (原样放行)")
            print("   " + heat(rec, frames[t]).replace("\n", "\n   "))

    print("\n" + "=" * 110)
    print("汇总")
    for s in summary:
        print(f"  {s['label']:>18s} 帧={s['frames']:6d} 单点压制={s['single']:6d} 大面积放行={s['area']:6d} "
              f"形态不明={s['ambig']:6d} 静默={s['low']:6d} 抑制率={s['supp_ratio']:5.1f}% "
              f"大面积改动={s['area_changed']} 方向={s['dir']}")


if __name__ == "__main__":
    main()
