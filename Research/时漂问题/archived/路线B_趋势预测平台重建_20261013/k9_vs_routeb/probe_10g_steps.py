# -*- coding: utf-8 -*-
"""10 g 级台阶（≈100 ADC）在真实录制上到底能不能被保住？

需求口径的 10 g ≈ 100 ADC（1000 ADC/N 的粗略折算），远小于加载阶跃，也小于本底漂移的
全程幅度。本脚本用 1 秒窗口中值差在「未补偿输入」上检出 ±60 ADC 级台阶，
再比较各方法在同一时刻的对应台阶幅度，得到逐次保留率。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WINDOW_S = 1.0
GRID_S = 0.25
MIN_ADC = 60.0
REFRACTORY_S = 2.0
SOURCES = {
    "pre": "total_pre", "out_rec": "total_out_rec", "k9_rec": "total_k9_rec",
    "k9_0p4": "total_k9_0p4", "routeb": "total_routeb",
}
LABELS = ["设备1_测试1", "设备1_测试2", "设备1_10g反复增减",
          "设备2_测试1", "设备2_测试2", "设备2_10g反复增减"]
METHODS = ("out_rec", "k9_rec", "k9_0p4", "routeb")


def levels(t: np.ndarray, total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """网格上的「前 1 s 中值 / 后 1 s 中值」。"""
    grid = np.arange(t[0] + WINDOW_S, t[-1] - WINDOW_S, GRID_S)
    before = np.array([np.median(total[(t >= g - WINDOW_S) & (t < g)]) for g in grid])
    after = np.array([np.median(total[(t >= g) & (t < g + WINDOW_S)]) for g in grid])
    return grid, after - before


def detect(grid: np.ndarray, diff: np.ndarray) -> list[dict]:
    hits = np.flatnonzero(np.abs(diff) >= MIN_ADC)
    out: list[dict] = []
    for i in hits:
        if out and grid[i] - out[-1]["t"] < REFRACTORY_S:
            if abs(diff[i]) > abs(out[-1]["adc"]):
                out[-1].update(t=float(grid[i]), adc=float(diff[i]), i=int(i))
            continue
        out.append(dict(t=float(grid[i]), adc=float(diff[i]), i=int(i)))
    return out


def main() -> int:
    for label in LABELS:
        data = np.load(HERE / "series" / ("%s.npz" % label))
        t = data["t"]
        grid, diff_pre = levels(t, data["total_pre"])
        steps = detect(grid, diff_pre)
        per_method = {m: levels(t, data[SOURCES[m]])[1] for m in METHODS}
        print("\n=== %s ｜ 输入侧 ±%.0f ADC 级台阶共 %d 个（1 s 窗口中值差）"
              % (label, MIN_ADC, len(steps)))
        if not steps:
            continue
        print("  %-8s %9s | %s" % ("时刻s", "输入Δ", "  ".join("%-18s" % m for m in METHODS)))
        for step in steps:
            cells = []
            for m in METHODS:
                amp = float(per_method[m][step["i"]])
                ret = amp / step["adc"] if abs(step["adc"]) > 1e-9 else float("nan")
                cells.append("%+7.1f(%5.2f)" % (amp, ret))
            print("  %-8.2f %+9.1f | %s" % (step["t"], step["adc"], "  ".join(cells)))
        print("  -> 平均保留率：" + "  ".join(
            "%s %.2f" % (m, float(np.mean([per_method[m][s["i"]] / s["adc"] for s in steps])))
            for m in METHODS))
        print("  -> 保留 ≥0.7 的次数：" + "  ".join(
            "%s %d/%d" % (m, int(np.sum([per_method[m][s["i"]] / s["adc"] >= 0.7
                                         for s in steps])), len(steps))
            for m in METHODS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
