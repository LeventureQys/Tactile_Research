# -*- coding: utf-8 -*-
"""真实加载在空间上是分散还是集中？

用加载前 3~0.5 s 与加载后 1~4 s 的逐通道中值差，量出增量的空间分布：
多少通道真的在承载、最大通道占总增量的比例、以及各通道 cap 与自身增量的比值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import SESSIONS, detect_steps, load_session, med  # noqa: E402
from creep_observer_k9 import Params, fields                                 # noqa: E402


def main() -> int:
    for label, device, name in SESSIONS:
        sess = load_session(device, name)
        t, values = sess["t"], sess["pre"].astype(np.float64)
        total = values.sum(1)
        steps = detect_steps(t, total, 2000.0)
        if not steps:
            continue
        load = steps[0]["t"]
        before = np.median(values[(t >= max(0.0, load - 3.0)) & (t < load - 0.5)], axis=0)
        after = np.median(values[(t >= load + 1.0) & (t < load + 4.0)], axis=0)
        delta = after - before
        positive = delta[delta > 0]
        total_delta = float(delta.sum())
        active = delta[delta >= 50.0]
        share = float(np.max(delta)) / total_delta if total_delta else float("nan")
        params = Params(**{k: float(v) for k, v in sess["session"]["algorithm"]["params"].items()
                           if k in {f.name for f in fields(Params)}})
        cap = params.slope_cap_frac * np.maximum(np.maximum(after, 1.0), 1.0)
        per_channel = delta / np.maximum(cap, 1e-9)
        print("\n=== %s ｜ 首次大加载 %.2f s，总增量 %+.0f ADC"
              % (label, load, total_delta))
        print("  有正增量的通道 %d/57 ｜ 增量 ≥50 ADC 的通道 %d ｜ 最大单通道 %+.0f ADC"
              % (int(np.sum(delta > 0)), int(active.size), float(np.max(delta))))
        print("  最大通道占总增量 %.1f%% ｜ 前 5 通道占 %.1f%% ｜ 增量中位（正值） %.1f ADC"
              % (100.0 * share,
                 100.0 * float(np.sort(delta)[-5:].sum() / total_delta) if total_delta else 0.0,
                 float(np.median(positive)) if positive.size else float("nan")))
        finite = per_channel[np.isfinite(per_channel) & (delta > 0)]
        print("  各通道「吸收自己那份所需时间」（cap 用加载后 e）：中位 %.2f s，p90 %.2f s，最大 %.2f s"
              % (float(np.median(finite)), float(np.percentile(finite, 90)), float(np.max(finite))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
