# -*- coding: utf-8 -*-
"""合并 tune_grid 两片结果：排序 + 帕累托前沿。

判据：漂移侧越小越好（四份恒载 |首尾净变化| 的最大值），保留侧越大越好
（真实背景上分散 57×100 ADC 与集中 1×300 ADC 的配对保持率）。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ("tune_grid_part1.json", "tune_grid_part2.json")
DISPERSED, CONCENTRATED = "分散57×100", "集中1×300"


def main() -> int:
    rows = []
    for name in PARTS:
        path = HERE / name
        if not path.exists():
            print("缺少 %s（该分片还没跑完）" % name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["configs"])
    if not rows:
        return 1

    for row in rows:
        row["disp"] = row["injects"][DISPERSED]["hold_adc"]
        row["conc"] = row["injects"][CONCENTRATED]["hold_adc"]
        row["resid"] = row["injects"][DISPERSED]["release_7_12"]

    print("%-30s %10s %10s %10s %10s %8s" %
          ("配置", "漂移|净变|", "漂移|偏移|", "分散保持", "集中保持", "撤载残差"))
    for row in sorted(rows, key=lambda r: r["worst_abs_net"]):
        print("%-30s %10.1f %10.1f %10.1f %10.1f %8.1f"
              % (row["name"], row["worst_abs_net"], row["worst_abs_offset"],
                 row["disp"], row["conc"], row["resid"]))

    front = []
    for row in rows:
        dominated = any(
            other is not row
            and other["worst_abs_net"] <= row["worst_abs_net"]
            and other["disp"] >= row["disp"]
            and other["conc"] >= row["conc"]
            and (other["worst_abs_net"] < row["worst_abs_net"]
                 or other["disp"] > row["disp"] or other["conc"] > row["conc"])
            for other in rows)
        if not dominated:
            front.append(row)
    print("\n帕累托前沿（漂移更小 + 两个保留率都不更差，无人全面占优）：")
    for row in sorted(front, key=lambda r: r["worst_abs_net"]):
        print("  %-30s 漂移 %6.1f ｜ 分散 %5.1f ｜ 集中 %5.1f"
              % (row["name"], row["worst_abs_net"], row["disp"], row["conc"]))

    ref = {r["name"]: r for r in rows}
    print("\n相对参考点：")
    for name in ("现役 r=0.1", "r=0.4"):
        if name in ref:
            r = ref[name]
            print("  %-14s 漂移 %6.1f ｜ 分散 %5.1f（%.1f%%）｜ 集中 %5.1f（%.1f%%）"
                  % (name, r["worst_abs_net"], r["disp"], r["disp"] / 100.0,
                     r["conc"], r["conc"] / 300.0))
    (HERE / "tune_grid_merged.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> %s" % (HERE / "tune_grid_merged.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
