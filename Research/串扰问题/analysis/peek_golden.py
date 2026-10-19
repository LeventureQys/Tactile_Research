# -*- coding: utf-8 -*-
"""快速查看黄金用例结构。"""

import io
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

g = json.load(io.open(os.path.join(OUT, "golden_cases.json"), encoding="utf-8"))
print("设备数:", len(g))
for k, v in g.items():
    cal = v["calibration"]
    print(f"  {k}: rows={v['rows']} cols={v['cols']} n_active={len(cal['rest'])} "
          f"ref_scale={cal['ref_scale']:.1f} noise={cal['noise']:.1f} 用例数={len(v['cases'])}")
    for c in v["cases"]:
        print(f"     {c['name']:12s} frame={c['frame_index']:6d} mode={c['expect_mode']:9s} "
              f"dir={str(c['expect_direction']):5s} suppressed={c['expect_suppressed']}")
