# -*- coding: utf-8 -*-
"""第九步：大面积保护诊断 —— 统计被放行的帧，确认放行是否合理。"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402
from crosstalk_suppress import CrosstalkSuppressor, SuppressConfig  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def show(rec, f, peak, out):
    idx = rec.index_map
    lines = []
    for r in range(rec.rows):
        cells = []
        for c in range(rec.cols):
            k = idx[r, c]
            cells.append("      ." if k < 0 else "%7.0f" % f[r, c])
        lines.append("   " + " ".join(cells))
    return "\n".join(lines)


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore09_area.txt"), "w", encoding="utf-8")
    for rec in iter_all_recordings(RAW):
        d = response(rec.data)
        frames = to_frames(rec, d)
        sup = CrosstalkSuppressor(rec, SuppressConfig(direction="auto")).calibrate(d)
        P_all = np.nanmax(frames, axis=(1, 2))
        thr = 0.25 * np.nanpercentile(P_all, 99.5)
        modes = {}
        areas = []
        for t in range(rec.n_frames):
            if not np.isfinite(P_all[t]) or P_all[t] < max(thr, 3 * sup.noise):
                modes["low"] = modes.get("low", 0) + 1
                continue
            y_out, info = sup.process(frames[t])
            modes[info["mode"]] = modes.get(info["mode"], 0) + 1
            if info["mode"] == "area":
                areas.append((t, info["peak"], info["active_rows"], info["active_cols"]))
        print("=" * 96, file=sys.stdout)
        print(f"{rec.label}  模式分布={modes}", file=sys.stdout)
        if areas:
            arr = np.array([a[1] for a in areas])
            print(f"  放行帧峰值: min={arr.min():.0f} p50={np.median(arr):.0f} max={arr.max():.0f}", file=sys.stdout)
            print("  活跃行数分布:", dict(zip(*[x.tolist() for x in np.unique([a[2] for a in areas], return_counts=True)])), file=sys.stdout)
            print("  活跃列数分布:", dict(zip(*[x.tolist() for x in np.unique([a[3] for a in areas], return_counts=True)])), file=sys.stdout)
            for t, pk, ar, ac in sorted(areas, key=lambda z: -z[1])[:2]:
                print(f"  --- t={t} 峰值={pk:.0f} 活跃行={ar} 活跃列={ac}", file=sys.stdout)
                print(show(rec, frames[t], pk, sys.stdout), file=sys.stdout)
        else:
            print("  （无放行帧）", file=sys.stdout)


if __name__ == "__main__":
    main()
