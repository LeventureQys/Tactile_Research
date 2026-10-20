# -*- coding: utf-8 -*-
"""第十三步：线内保护拦下的帧形态核查 + 合成边界用例生成。

1) 找出所有被 line_other_max / line_energy_max 拦下的帧，确认拦得对不对
2) 生成合成用例（整面按压、同列双触点、背景接触+主点），作为边界回归
"""

import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response_with_baseline, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402
from crosstalk_portable import (DIR_COL, DIR_ROW, MODE_AMBIG, MODE_AREA,  # noqa: E402
                                MODE_LOW, MODE_SINGLE, Params, Suppressor)
from crosstalk_suppress import CrosstalkSuppressor, SuppressConfig  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MODE_NAME = {MODE_LOW: "low", MODE_SINGLE: "single", MODE_AREA: "area", MODE_AMBIG: "ambiguous"}
DIR_NAME = {0: None, DIR_ROW: "row", DIR_COL: "col"}


def heat(rec, y, mx, width=7):
    idx = rec.index_map
    return "\n".join(" ".join(" " * width if not np.isfinite(y[r, c]) else f"{int(round(y[r, c])):>{width}}"
                              for c in range(rec.cols)) for r in range(rec.rows))


def main():
    sys.stdout = io.open(os.path.join(OUT, "explore13_bounds.txt"), "w", encoding="utf-8")

    # ---------- 1) 线内保护拦下的帧 ----------
    for rec in iter_all_recordings(RAW):
        d, _ = response_with_baseline(rec.data)
        frames = to_frames(rec, d)
        cfg = SuppressConfig(direction="auto", mode="zero")
        sup = CrosstalkSuppressor(rec, cfg).calibrate(d)
        P = np.nanmax(frames, axis=(1, 2))
        P = np.where(np.isfinite(P), P, 0.0)
        thr = max(0.25 * np.percentile(P, 99.5), 3 * sup.noise)
        hits = []
        for t in range(rec.n_frames):
            if P[t] < thr:
                continue
            y_out, info = sup.process(frames[t])
            if info["mode"] == "area" and info["active_rows"] < 3:
                hits.append(t)
        print("=" * 104)
        print(f"{rec.label}: 被『线内保护』拦下的帧 = {len(hits)}")
        for t in sorted(hits, key=lambda tt: -P[tt])[:3]:
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            e_row = float(np.nansum(np.delete(f[r0, :], c0)))
            e_col = float(np.nansum(np.delete(f[:, c0], r0)))
            direc = "row" if e_row >= e_col else "col"
            line = np.delete(f[r0, :] if direc == "row" else f[:, c0],
                             c0 if direc == "row" else r0)
            line = line[np.isfinite(line)]
            print(f"  --- t={t} 峰值={P[t]:.0f} 主点=({r0},{c0}) 方向={direc} "
                  f"line_other={np.max(line)/P[t]:.2f} line_energy={np.sum(line)/P[t]:.2f}")
            print("  " + heat(rec, f, P[t]).replace("\n", "\n  "))

    # ---------- 2) 合成边界用例 ----------
    print("\n" + "#" * 104)
    print("# 合成边界用例（用于移植回归）")
    synth = {}
    for rec in iter_all_recordings(RAW):
        d, _ = response_with_baseline(rec.data)
        cfg = SuppressConfig(direction="auto", mode="zero")
        sup = CrosstalkSuppressor(rec, cfg).calibrate(d)
        p = Params()
        p.area_min_rows, p.area_min_cols = cfg.area_min_rows, cfg.area_min_cols
        port = Suppressor(rec.rows, rec.cols, rec.mask, p)
        port.set_calibration(sup.ref_scale, sup.noise, sup.rest_ch)
        idx = rec.index_map
        n = rec.n_active
        cases = []

        def make(name, cells: dict, note):
            v = [0.0] * n
            for (r, c), val in cells.items():
                k = idx[r, c]
                if k >= 0:
                    v[k] = float(val)
            out = [0.0] * n
            info = port.process(list(v), out)
            cases.append(dict(name=name, note=note,
                              input_resp=[round(x, 6) for x in v],
                              expect_mode=MODE_NAME[info["mode"]],
                              expect_direction=DIR_NAME[info["direction"]],
                              expect_suppressed=info["suppressed"],
                              expect_out=[round(x, 6) for x in out]))
            print(f"  {rec.label} / {name}: mode={MODE_NAME[info['mode']]} "
                  f"dir={DIR_NAME[info['direction']]} suppressed={info['suppressed']}  ({note})")

        # ① 整面按压：所有有效格 500，主点 600
        all_cells = {(r, c): 500 for r in range(rec.rows) for c in range(rec.cols)
                     if idx[r, c] >= 0}
        r0, c0 = next(k for k in all_cells)
        all_cells[(r0, c0)] = 600
        make("synth_full_press", all_cells, "整面均匀按压，必须零改动放行")

        # ② 单点 + 强串扰（沿该设备的主方向铺一条衰减尾巴）
        is_row = (cfg.direction == "row" if cfg.direction else
                  (lambda: (lambda e: e[0] >= e[1])(
                      (np.nansum(d[:, idx[7, 1]]) if False else 0, 0)))())
        direction = "row" if (cfg.direction or "row") == "row" else "col"
        # 用标定表已判定的方向
        with io.open(os.path.join(OUT, "bias_table.json"), encoding="utf-8") as fh:
            tables = json.load(fh)
        key = next((k for k in tables
                    if k.split("/")[0] == rec.label.split("/")[0]
                    and k.split("/")[-1].split("-")[0] == rec.label.split("/")[-1].split("-")[0]), None)
        direction = tables[key]["direction"] if key else "row"
        single = {}
        if direction == "row":
            base = [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4)]
            vals = [200, 600, 400, 300, 200]
            for (r, c), v in zip(base, vals):
                if idx[r, c] >= 0:
                    single[(r, c)] = v
        else:
            base = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
            vals = [150, 600, 300, 200, 150]
            for (r, c), v in zip(base, vals):
                if idx[r, c] >= 0:
                    single[(r, c)] = v
        make("synth_single_with_tail", single, "单点+同轴串扰尾巴，必须被压制")

        # ③ 主点 + 另一轴上的独立接触（模拟新工况的背景接触）
        bg = dict(single)
        if direction == "row":
            for c in range(rec.cols):
                if idx[0, c] >= 0:
                    bg[(0, c)] = 80
        else:
            for r in range(rec.rows):
                if idx[r, 0] >= 0:
                    bg[(r, 0)] = 80
        make("synth_single_plus_orthogonal_contact", bg,
             "主点+正交轴独立接触：只压主点线，不得碰独立接触")

        synth[rec.label] = dict(rows=rec.rows, cols=rec.cols, layout_mask=rec.mask,
                                calibration=dict(ref_scale=float(sup.ref_scale),
                                                 noise=float(sup.noise),
                                                 rest=[round(float(x), 6) for x in sup.rest_ch]),
                                cases=cases)

    with io.open(os.path.join(OUT, "synthetic_cases.json"), "w", encoding="utf-8") as f:
        json.dump(synth, f, ensure_ascii=False, indent=1)
    print(f"\n合成用例已写入 {os.path.join(OUT, 'synthetic_cases.json')}")


if __name__ == "__main__":
    main()
