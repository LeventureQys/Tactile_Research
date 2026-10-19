# -*- coding: utf-8 -*-
"""路线B 的中间层在「一次事件都不触发」时是否仍然改变 K9 的补偿轨迹？

做法：把候选阈值抬到不可达（threshold_adc / noise_multiplier 设成 1e9），
使事件通路永不确认；此时路线B 只保留逐帧透传 + 大跳变时的状态重置（_reset_uncertain/_reanchor）。
与纯 K9（r_slow_max=0.4）逐帧对比，即可分离「事件锁存」与「状态重置」两种影响。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import (METHODS, SESSIONS, detect_steps, load_session,  # noqa: E402
                                  med, metrics, params_of, run_k9)
from route_b import CONFIG, route_b                                            # noqa: E402

DEAD = dict(CONFIG)
DEAD.update(threshold_adc=1e9, noise_multiplier=1e9)


def main() -> int:
    print("%-22s %12s %12s %11s %10s %10s %10s"
          % ("会话", "max|B0-K9.4|", "非零帧占比", "事件数",
             "K9.4净变", "B0净变", "路线B净变"))
    for label, device, name in SESSIONS:
        sess = load_session(device, name)
        t, pre = sess["t"], sess["pre"]
        raw_params = dict(sess["session"]["algorithm"]["params"])
        params = params_of(sess["session"], 0.4)
        k9 = run_k9(pre, t, params)

        values = {k: v for k, v in raw_params.items()}
        values["r_slow_max"] = 0.4
        out_b0, trace0 = route_b(pre, t, values, dict(DEAD))
        out_b, trace = route_b(pre, t, values)

        raw_total = pre.sum(1)
        steps = detect_steps(t, raw_total, 2000.0)
        load = steps[0]["t"] if steps else float(t[0])
        diff = out_b0.sum(1) - k9.sum(1)
        active = np.abs(diff) > 1.0
        nz = np.flatnonzero(active)
        print("%-22s %12.1f %11.2f%% %11d %+10.1f %+10.1f %+10.1f"
              % (label, float(np.max(np.abs(diff))), 100.0 * float(np.mean(active)),
                 len(trace0["events"]),
                 metrics(t, k9.sum(1), load)["net_change_adc"],
                 metrics(t, out_b0.sum(1), load)["net_change_adc"],
                 metrics(t, out_b.sum(1), load)["net_change_adc"]))
        if nz.size:
            print("%-22s   差异区间 %.2f~%.2f s（%d 帧）；加载时刻 %.2f s，"
                  "加载后 8 s 内占 %d 帧"
                  % ("", t[nz[0]], t[nz[-1]], nz.size, load,
                     int(np.sum(nz < np.searchsorted(t, load + 8.0)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
