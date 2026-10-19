# -*- coding: utf-8 -*-
"""这批录制里究竟有没有可辨识的「卸载」动作？

在未补偿输入的总力上做 1 秒窗口中值差，分别统计正台阶与负台阶（多档阈值）。
若某份录制号称「反复增减」，却只有正台阶、没有负台阶，那它就无法用来验证逐次增减保留率。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = ["设备1_测试1", "设备1_测试2", "设备1_10g反复增减",
          "设备2_测试1", "设备2_测试2", "设备2_10g反复增减"]
WINDOW_S, GRID_S, REFRACTORY_S = 1.0, 0.25, 2.0
THRESHOLDS = (30.0, 60.0, 100.0, 300.0)


def main() -> int:
    for label in LABELS:
        data = np.load(HERE / "series" / ("%s.npz" % label))
        t, total = data["t"], data["total_pre"]
        grid = np.arange(t[0] + WINDOW_S, t[-1] - WINDOW_S, GRID_S)
        before = np.array([np.median(total[(t >= g - WINDOW_S) & (t < g)]) for g in grid])
        after = np.array([np.median(total[(t >= g) & (t < g + WINDOW_S)]) for g in grid])
        diff = after - before
        base = float(np.median(total[:200]))
        print("\n=== %s（%.1f s，输入总力 %.0f→%.0f，全程变化 %+.0f ADC）"
              % (label, t[-1], base, float(np.median(total[-200:])),
                 float(np.median(total[-200:]) - base)))
        print("  %8s %10s %10s %12s %12s" % ("阈值", "正台阶数", "负台阶数", "最大正台阶", "最大负台阶"))
        for th in THRESHOLDS:
            pos, neg = [], []
            for i in np.flatnonzero(np.abs(diff) >= th):
                if pos or neg:
                    last = (pos + neg)[-1]
                    if grid[i] - last[0] < REFRACTORY_S:
                        continue
                (pos if diff[i] > 0 else neg).append((float(grid[i]), float(diff[i])))
            print("  %8.0f %10d %10d %12s %12s"
                  % (th, len(pos), len(neg),
                     "%+.0f" % max((v for _, v in pos), default=float("nan"))
                     if pos else "—",
                     "%+.0f" % min((v for _, v in neg), default=float("nan"))
                     if neg else "—"))
            if th == 60.0:
                for kind, items in (("正", pos), ("负", neg)):
                    if items:
                        print("      %s台阶：%s" % (kind, ", ".join(
                            "%.1fs%+.0f" % (g, v) for g, v in items[:10])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
