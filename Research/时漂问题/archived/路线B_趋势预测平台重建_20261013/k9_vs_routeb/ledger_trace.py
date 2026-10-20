# -*- coding: utf-8 -*-
"""路线B 平台账本（ledger）时间轨迹 + 真实台阶时刻对照。

回答一个问题：真实录制里的 10 g 级台阶发生时，事件通路有没有把增量锁进账本？
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "02_现场调参" / "09-21_09-26_工程A_v3.4"))

from compare_k9_vs_routeb import (JUMP_ADC, SESSIONS, detect_steps,  # noqa: E402
                                  load_session, params_of)
from route_b import CONFIG, OnlineObserver                            # noqa: E402

TARGETS = ("设备1/10g反复增减", "设备2/10g反复增减", "设备1/测试1", "设备2/测试1")


def main() -> int:
    for label, device, name in SESSIONS:
        if label not in TARGETS:
            continue
        sess = load_session(device, name)
        t, values = sess["t"], sess["pre"]
        raw_params = dict(sess["session"]["algorithm"]["params"])
        params = params_of(sess["session"], 0.4)
        obs = OnlineObserver(params, dict(CONFIG))
        obs.diag = deque(maxlen=10 ** 7)
        ledger = np.empty(t.size)
        for i in range(t.size):
            obs.process(float(t[i]), values[i])
            ledger[i] = float(np.sum(obs.ledger))
        steps = detect_steps(t, values.sum(1), JUMP_ADC)
        small = detect_small(t, values.sum(1), 150.0)
        print("\n=== %s（%d 帧）" % (label, t.size))
        print("  账本 |ledger|: max=%.1f  非零帧占比=%.2f%%  末值=%.1f"
              % (float(np.max(np.abs(ledger))), 100.0 * float(np.mean(np.abs(ledger) > 0.5)),
                 float(ledger[-1])))
        nz = np.flatnonzero(np.abs(ledger) > 0.5)
        if nz.size:
            runs = []
            start = nz[0]
            for a, b in zip(nz[:-1], nz[1:]):
                if b - a > 1:
                    runs.append((start, a))
                    start = b
            runs.append((start, nz[-1]))
            for a, b in runs[:12]:
                print("    ledger 非零区间 %.2f~%.2f s（%.0f 帧），区间内 |ledger| 最大 %.1f"
                      % (t[a], t[b], b - a + 1, float(np.max(np.abs(ledger[a:b + 1])))))
        print("  事件:")
        for e in obs.events:
            print("    %.2f s  %-8s Δ=%+.1f ADC  平台合计=%+.1f"
                  % (e["time_s"], e["action"], e["delta_total_adc"], e["platform_total_adc"]))
        print("  真实台阶（首次大加载 %.2f s，阈值 %.0f ADC）:" % (steps[0]["t"] if steps else -1, JUMP_ADC))
        for s in steps:
            k = int(np.searchsorted(t, s["t"]))
            print("    %.2f s 输入台阶 %+.0f ADC → 该帧 ledger %+.1f，其后 3 s ledger 最大 %.1f"
                  % (s["t"], s["adc"], ledger[k],
                     float(np.max(np.abs(ledger[k:k + 300]))) if k + 1 < ledger.size else 0.0))
        print("  小台阶（阈值 150 ADC，前 12 个）:")
        for s in small[:12]:
            k = int(np.searchsorted(t, s["t"]))
            print("    %.2f s 输入台阶 %+.0f ADC → 该帧 ledger %+.1f，其后 3 s ledger 最大 %.1f"
                  % (s["t"], s["adc"], ledger[k],
                     float(np.max(np.abs(ledger[k:k + 300]))) if k + 1 < ledger.size else 0.0))
    return 0


def detect_small(t: np.ndarray, total: np.ndarray, min_adc: float,
                 refractory_s: float = 2.0) -> list[dict]:
    d = np.diff(total)
    hits = np.flatnonzero(np.abs(d) >= min_adc)
    out: list[dict] = []
    for i in hits:
        if out and t[i] - out[-1]["t"] < refractory_s:
            out[-1]["t"] = float(t[i])
            out[-1]["i"] = int(i)
            continue
        out.append(dict(i=int(i), t=float(t[i]), adc=float(d[i])))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
