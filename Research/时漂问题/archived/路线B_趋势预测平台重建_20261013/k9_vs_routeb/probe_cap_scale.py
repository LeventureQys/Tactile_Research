# -*- coding: utf-8 -*-
"""为什么「集中加载保得住、分散加载保不住」—— 逐通道量出慢态吸收尺度的差别。

K9 的慢态积分速率上限是**逐通道**量：cap_k = slope_cap_frac · max(e_k, 1)，
其中 e_k 是该通道自己「未被解释掉的载荷」。分散注入把总增量摊到 57 个通道后，
每通道增量远小于该通道的 cap；集中注入则相反。本脚本把这两个数直接量出来。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import detect_steps, load_session, params_of  # noqa: E402
from creep_observer_k9 import CreepObserverK9                          # noqa: E402

R_SLOW_MAX = 0.6
START_OFFSET = 30.0
CASES = (("分散 57 × 100 ADC", 57, 100.0), ("集中 1 × 300 ADC", 1, 300.0))


def main() -> int:
    sess = load_session("设备1", "测试2-2026-10-13_16-43-59")
    t, values = sess["t"], sess["pre"].astype(np.float64)
    params = params_of(sess["session"], R_SLOW_MAX)
    load = detect_steps(t, values.sum(1), 2000.0)[0]["t"]
    start = load + START_OFFSET
    stop = int(np.searchsorted(t, start))

    obs = CreepObserverK9(params)
    obs._trace_frame = lambda *a: None
    for i in range(stop):
        obs.process(float(t[i]), values[i])

    y = values[stop] - obs.zero
    e = np.maximum(y - obs.x_fast - obs.x_slow, 0.0)
    cap = params.slope_cap_frac * np.maximum(e, 1.0)
    dt = float(np.median(np.diff(t)))
    print("注入起点 %.3f s ｜ 背景总力 %.0f ｜ 逐通道载荷 e：中位 %.1f，最大 %.1f（通道 %d）"
          % (start, values[stop].sum(), float(np.median(e)), float(np.max(e)),
             int(np.argmax(e))))
    print("该通道慢态吸收上限 cap = slope_cap_frac(%.3f) × max(e,1)：中位 %.3f，最大 %.2f ADC/s"
          % (params.slope_cap_frac, float(np.median(cap)), float(np.max(cap))))
    print("逐帧 dt = %.4f s\n" % dt)

    out = {"start_s": start, "background_total": float(values[stop].sum()),
           "e_median": float(np.median(e)), "e_max": float(np.max(e)),
           "cap_median": float(np.median(cap)), "cap_max": float(np.max(cap)),
           "cases": {}}
    for name, footprint, total_adc in CASES:
        per_channel = total_adc / footprint
        print("【%s】每通道增量 %.3f ADC" % (name, per_channel))
        if footprint == 1:
            idx = np.array([0])
            absorb = per_channel / cap[idx]
            print("  该通道 e=%.1f → cap=%.2f ADC/s → 吸收该增量需 %.1f s"
                  % (e[0], cap[0], absorb[0]))
            out["cases"][name] = {"per_channel_adc": per_channel,
                                  "cap": float(cap[0]), "absorb_s": float(absorb[0])}
        else:
            absorb = per_channel / cap
            print("  57 通道各自吸收该增量需：中位 %.2f s、最大 %.2f s、最小 %.2f s"
                  % (float(np.median(absorb)), float(np.max(absorb)), float(np.min(absorb))))
            print("  其中 1 s 内就吸完的通道占 %.0f%%"
                  % (100.0 * float(np.mean(absorb <= 1.0))))
            out["cases"][name] = {"per_channel_adc": per_channel,
                                  "absorb_median_s": float(np.median(absorb)),
                                  "absorb_max_s": float(np.max(absorb)),
                                  "frac_within_1s": float(np.mean(absorb <= 1.0))}
        print()
    (HERE / "cap_scale.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-> %s" % (HERE / "cap_scale.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
