# -*- coding: utf-8 -*-
"""黄金用例自检 —— 模拟移植方的回归流程。

覆盖两组用例：
  * golden_cases.json     来自真实录制的代表帧
  * synthetic_cases.json  构造的边界帧（整面按压 / 单点+串扰尾巴 / 主点+正交独立接触）

除逐位比对期望输出外，还校验一条**核心不变量**：
    压制只允许改动"主点所在的那一条线"，其余格子必须与输入逐位一致。
这条不变量是"不影响整面/多点触发"需求的形式化，任何移植实现都必须满足。
"""

import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_portable import (DIR_COL, DIR_ROW, MODE_AMBIG, MODE_AREA,  # noqa: E402
                                MODE_LOW, MODE_SINGLE, Params, Suppressor)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MODE_NAME = {MODE_LOW: "low", MODE_SINGLE: "single", MODE_AREA: "area", MODE_AMBIG: "ambiguous"}
DIR_NAME = {0: None, DIR_ROW: "row", DIR_COL: "col"}


def run_suite(path, title, extra_invariant=True):
    if not os.path.exists(path):
        print(f"[跳过] {title}: 文件不存在")
        return 0, 0
    g = json.load(io.open(path, encoding="utf-8"))
    total = passed = 0
    print(f"===== {title} =====")
    for label, dev in g.items():
        p = Params()
        p.area_min_rows, p.area_min_cols = 3, 3
        sup = Suppressor(dev["rows"], dev["cols"], dev["layout_mask"], p)
        cal = dev["calibration"]
        sup.set_calibration(cal["ref_scale"], cal["noise"], cal["rest"])
        cell_row = sup.cell_row
        cell_col = sup.cell_col
        print(f"{label}  (n_active={sup.n})")
        for case in dev["cases"]:
            total += 1
            inp = list(case["input_resp"])
            out = [0.0] * sup.n
            info = sup.process(inp, out)
            ok_out = all(abs(out[k] - case["expect_out"][k]) < 1e-6 for k in range(sup.n))
            ok_mode = MODE_NAME[info["mode"]] == case["expect_mode"]
            ok_dir = DIR_NAME[info["direction"]] == case["expect_direction"]
            ok_cnt = info["suppressed"] == case["expect_suppressed"]

            # 核心不变量：只有主点所在的那一条线允许被改动
            ok_inv = True
            if extra_invariant:
                k0 = int(np.argmax(inp))
                r0, c0 = cell_row[k0], cell_col[k0]
                touched = []
                for k in range(sup.n):
                    same_line = (cell_row[k] == r0) or (cell_col[k] == c0)
                    if not same_line and abs(out[k] - inp[k]) > 1e-9:
                        touched.append(k)
                ok_inv = not touched

            ok = ok_out and ok_mode and ok_dir and ok_cnt and ok_inv
            passed += ok
            flag = "PASS" if ok else "FAIL"
            note = case.get("note", "")
            print(f"   {flag}  {case['name']:38s} mode={MODE_NAME[info['mode']]:9s} "
                  f"dir={str(DIR_NAME[info['direction']]):5s} suppressed={info['suppressed']:3d}"
                  + (f"   {note}" if note else "")
                  + ("" if ok else f"   <-- out={ok_out} mode={ok_mode} dir={ok_dir} "
                                   f"cnt={ok_cnt} 不变量={ok_inv}"))
    print(f"{title}: {passed}/{total} 通过\n")
    return passed, total


def main():
    sys.stdout = io.open(os.path.join(OUT, "check_golden.txt"), "w", encoding="utf-8")
    p1, t1 = run_suite(os.path.join(OUT, "golden_cases.json"), "真实录制黄金用例")
    p2, t2 = run_suite(os.path.join(OUT, "synthetic_cases.json"), "合成边界用例")
    print(f"合计: {p1 + p2}/{t1 + t2} 通过")

    # 单帧耗时实测
    g = json.load(io.open(os.path.join(OUT, "golden_cases.json"), encoding="utf-8"))
    dev = max(g.values(), key=lambda d: len(d["calibration"]["rest"]))
    p = Params()
    sup = Suppressor(dev["rows"], dev["cols"], dev["layout_mask"], p)
    cal = dev["calibration"]
    sup.set_calibration(cal["ref_scale"], cal["noise"], cal["rest"])
    frames = [list(c["input_resp"]) for c in dev["cases"]]
    reps = 20000
    out = [0.0] * sup.n
    t0 = time.perf_counter()
    for i in range(reps):
        sup.process(frames[i % len(frames)], out)
    dt = (time.perf_counter() - t0) / reps
    print(f"单帧耗时（纯 Python {sup.n} 通道）: {dt*1e6:.1f} µs → 100 Hz 下约 {dt*100*100:.3f}% CPU")


if __name__ == "__main__":
    main()
