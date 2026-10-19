# -*- coding: utf-8 -*-
"""小幅度真实卸载测试：防护会不会"卡住"基线跟随？

构造长时间恒载（让 y_floor 爬到接近工作电平），在下探幅度极小的情况下施加
**真实卸载（降到新电平并保持）**，比较 关闭防护 / 开启防护 的输出。

判据：卸载后稳态差应≈0；若防护让 y_floor 卡在高位导致欠补偿，稳态差会显著。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from guarded_k9 import compensate  # noqa: E402

FPS = 100.0
CH = 52
BASELINE = 6.0
LOAD = 834.0
RUN_S = 1200.0
UNLOAD_AT = 1100.0
MAGNITUDES = (20.0, 50.0, 150.0, 400.0)   # 每通道减小的 ADC
RAMP_S = 0.2                               # 卸载很快完成


def build(per_channel_drop: float):
    time = np.arange(int(RUN_S * FPS)) / FPS
    values = np.full((time.size, CH), BASELINE)
    values[time >= 1.0] += LOAD
    if per_channel_drop > 0:
        ramp = np.clip((time - UNLOAD_AT) / RAMP_S, 0.0, 1.0)
        values -= ramp[:, None] * per_channel_drop
    return time, values


def main() -> None:
    params = dict(r_fast=0.03, tau_c_fast_s=2.0, tau_r_fast_s=0.1, r_slow_max=0.1,
                  slope_cap_frac=0.02, slow_confirm_s=3.0, soft_unfreeze_s=2.0,
                  tau_r_slow_idle_s=0.2, tau_r_slow_s=150.0, tau_slope_s=1.0,
                  slope_gate_frac=0.05, tau_zero_s=8.0, idle_frac=0.05,
                  y_max_tau_s=600.0, edge_slope_thres=60.0, edge_refract_s=2.0,
                  edge_boost_s=2.0, tau_c_fast_boost_s=2.0, hold_eps=2.0,
                  hold_tau_s=0.5, y_floor_tau_s=300.0, bypass_release_frac=1.15,
                  bypass_engage_frac=1.25, bypass_noise_sigma=3.0,
                  bypass_base_tau_s=10.0, ramp_slope_min=0.5, ramp_full_s=4.0)

    time, clean_values = build(0.0)
    clean_off = np.asarray(compensate(clean_values, time, params, dict(enabled=False))[0],
                           dtype=float).sum(axis=1)

    rows = []
    for magnitude in MAGNITUDES:
        _, values = build(magnitude)
        off = np.asarray(compensate(values, time, params, dict(enabled=False))[0],
                         dtype=float).sum(axis=1)
        on = np.asarray(compensate(values, time, params, dict(enabled=True))[0],
                        dtype=float).sum(axis=1)

        def win(a, b, sig):
            mask = (time >= a) & (time < b)
            return float(np.median(sig[mask]))

        rows.append(dict(
            per_channel_drop_adc=magnitude,
            total_drop_adc=round(magnitude * CH, 1),
            # 卸载后各阶段的 防护-无防护 输出差
            diff_0_1s=round(win(UNLOAD_AT, UNLOAD_AT + 1.0, np.abs(on - off)), 2),
            diff_1_5s=round(win(UNLOAD_AT + 1.0, UNLOAD_AT + 5.0, np.abs(on - off)), 2),
            diff_5_30s=round(win(UNLOAD_AT + 5.0, UNLOAD_AT + 30.0, np.abs(on - off)), 2),
            diff_steady_30_90s=round(win(UNLOAD_AT + 30.0, UNLOAD_AT + 90.0, np.abs(on - off)), 3),
            off_shift=round(win(UNLOAD_AT + 30.0, UNLOAD_AT + 90.0, off)
                            - win(UNLOAD_AT - 30.0, UNLOAD_AT - 1.0, off), 2),
            on_shift=round(win(UNLOAD_AT + 30.0, UNLOAD_AT + 90.0, on)
                           - win(UNLOAD_AT - 30.0, UNLOAD_AT - 1.0, on), 2),
        ))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    # 参照：不做卸载时两臂的固有差异（应为 0）
    same_off = np.asarray(compensate(clean_values, time, params, dict(enabled=False))[0],
                          dtype=float).sum(axis=1)
    same_on = np.asarray(compensate(clean_values, time, params, dict(enabled=True))[0],
                         dtype=float).sum(axis=1)
    control = dict(control_max_abs_diff=round(float(np.max(np.abs(same_on - same_off))), 6))

    report = dict(config=dict(RUN_S=RUN_S, UNLOAD_AT=UNLOAD_AT, RAMP_S=RAMP_S,
                              channels=CH, load_per_channel=LOAD),
                  control=control, cases=rows)
    (HERE / "small_unload_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(control, ensure_ascii=False))


if __name__ == "__main__":
    main()
