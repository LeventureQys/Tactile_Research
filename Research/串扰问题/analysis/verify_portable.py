# -*- coding: utf-8 -*-
"""移植蓝本的等价性验证 + 黄金测试用例导出。

1) 在 7 段录制的全部帧上，逐帧逐通道比对 Portable 版与 numpy 版的输出
2) 导出典型帧（单点横向/纵向、大面积、静默）作为黄金用例 JSON，
   供移植方在自己的平台上回归比对
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


def main():
    sys.stdout = io.open(os.path.join(OUT, "verify_portable.txt"), "w", encoding="utf-8")
    golden = {}
    for rec in iter_all_recordings(RAW):
        d, base = response_with_baseline(rec.data)
        frames = to_frames(rec, d)

        # ---- numpy 参考实现 ----
        ref_cfg = SuppressConfig(direction="auto", mode="zero")
        ref = CrosstalkSuppressor(rec, ref_cfg).calibrate(d)
        y_ref, infos_ref = ref.process_batch(d)

        # ---- 移植蓝本 ----
        p = Params()
        p.min_peak_abs = 0.0
        p.min_peak_ratio = 0.25
        port = Suppressor(rec.rows, rec.cols, rec.mask, p)
        out_ = [0.0] * port.n
        port.set_calibration(ref.ref_scale, ref.noise, ref.rest_ch)

        mism_out = 0
        mism_mode = 0
        mism_dir = 0
        infos_port = []
        for t in range(rec.n_frames):
            info = port.process(list(d[t]), out_)
            infos_port.append(info)
            # 输出比对
            yo = np.array(out_)
            idx = rec.index_map
            ref_vec = np.array([y_ref[t, r, c] for r in range(rec.rows)
                                for c in range(rec.cols) if idx[r, c] >= 0])
            if not np.allclose(yo, ref_vec, atol=1e-9, rtol=0):
                mism_out += 1
            ri = infos_ref[t]
            if MODE_NAME[info["mode"]] != ri["mode"]:
                mism_mode += 1
            if DIR_NAME[info["direction"]] != ri["direction"]:
                mism_dir += 1

        n_single = sum(1 for i in infos_port if i["mode"] == MODE_SINGLE)
        print(f"{rec.label}: 帧={rec.n_frames}  输出不一致={mism_out}  "
              f"模式不一致={mism_mode}  方向不一致={mism_dir}  单点帧={n_single}")

        # ---- 收集黄金用例：每种模式挑代表帧 ----
        cases = []
        picked = {"single_row": None, "single_col": None, "area": None, "low": None}
        best = {k: -1.0 for k in picked}
        for t in range(rec.n_frames):
            info = infos_port[t]
            pk = float(np.nanmax(frames[t]))
            if info["mode"] == MODE_SINGLE:
                key = "single_row" if info["direction"] == DIR_ROW else "single_col"
                if pk > best[key] and info["suppressed"] >= 3:
                    best[key] = pk
                    picked[key] = t
            elif info["mode"] == MODE_AREA and pk > best["area"]:
                best["area"] = pk
                picked["area"] = t
            elif info["mode"] == MODE_LOW and pk > best["low"]:
                best["low"] = pk
                picked["low"] = t
        for name, t in picked.items():
            if t is None:
                continue
            info = infos_port[t]
            out_ = [0.0] * port.n
            port.process(list(d[t]), out_)          # 重算，确保状态一致
            cases.append(dict(
                name=name, frame_index=int(rec.frame_index[t]),
                input_resp=[round(float(v), 6) for v in d[t]],
                expect_mode=MODE_NAME[info["mode"]],
                expect_direction=DIR_NAME[info["direction"]],
                expect_suppressed=int(info["suppressed"]),
                expect_out=[round(float(v), 6) for v in out_],
            ))
        golden[rec.label] = dict(
            rows=rec.rows, cols=rec.cols, layout_mask=rec.mask,
            calibration=dict(ref_scale=float(ref.ref_scale), noise=float(ref.noise),
                             rest=[round(float(v), 6) for v in ref.rest_ch]),
            cases=cases,
        )

    with io.open(os.path.join(OUT, "golden_cases.json"), "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
    print(f"\n黄金用例已写入 {os.path.join(OUT, 'golden_cases.json')}")


if __name__ == "__main__":
    main()
