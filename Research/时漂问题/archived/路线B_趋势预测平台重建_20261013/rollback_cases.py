# -*- coding: utf-8 -*-
"""路线B回撤/撤载因果合成案例。"""
from __future__ import annotations

import numpy as np

FPS = 100
CHANNELS = 57
BASELINE = 8.0


def _windows(*pairs):
    return [{"start_s": float(start), "end_s": float(end)} for start, end in pairs]


def _timeline(duration_s=80.0):
    return np.arange(0.0, duration_s, 1.0 / FPS, dtype=float)


def _profile(time, name, load, windows, acceptance):
    reference = np.full((time.size, CHANNELS), BASELINE, dtype=float)
    values = reference + load
    return {
        "name": name,
        "kind": "rollback",
        "time": time,
        "reference": reference,
        "values": values,
        "truth_total": load.sum(axis=1),
        "event_windows": windows,
        "acceptance": acceptance,
    }


def _dispersed_hold_withdrawal(time):
    channels = np.arange(0, CHANNELS, 3)
    fraction_up = np.clip((time - 10.0) / 2.0, 0.0, 1.0)
    fraction_down = 1.0 - np.clip((time - 27.0) / 2.0, 0.0, 1.0)
    level = np.where(time < 12.0, fraction_up, 1.0)
    level = np.where(time >= 27.0, fraction_down, level)
    load = np.zeros((time.size, CHANNELS), dtype=float)
    load[:, channels] = (100.0 / channels.size) * level[:, None]
    return load


def _three_rounds(time):
    load = np.zeros((time.size, CHANNELS), dtype=float)
    channels = np.arange(0, CHANNELS, 3)
    for start in (10.0, 30.0, 50.0):
        add = (time >= start) & (time < start + 5.0)
        remove = (time >= start + 5.0) & (time < start + 10.0)
        load[np.ix_(add, channels)] = 100.0 / channels.size
        load[np.ix_(remove, channels)] = -100.0 / channels.size
    return load


def _partial_unload(time):
    level = np.where(time < 10.0, 0.0, 100.0)
    level = np.where(time >= 30.0, 50.0, level)
    level = np.where(time >= 50.0, 0.0, level)
    return np.repeat((level / CHANNELS)[:, None], CHANNELS, axis=1)


def _slow_drift_small_step(time):
    drift = np.maximum(time - 5.0, 0.0) * (12.0 / CHANNELS)
    step = (time >= 35.0) * (20.0 / CHANNELS)
    return np.repeat((drift + step)[:, None], CHANNELS, axis=1)


def generate_cases():
    time = _timeline()
    cases = [
        _profile(
            time,
            "rollback_dispersed_100_hold15_withdraw",
            _dispersed_hold_withdrawal(time),
            _windows((10, 12), (12, 27), (27, 29), (29, 35)),
            {"hold_target_adc": 100.0, "withdraw_target_adc": 0.0, "hold_tolerance_adc": 10.0, "withdraw_residual_tolerance_adc": 10.0},
        ),
        _profile(
            time,
            "rollback_continuous_plusminus100_three_rounds",
            _three_rounds(time),
            _windows((10, 15), (15, 30), (30, 35), (35, 50), (50, 55), (55, 65)),
            {"cycle_abs_target_adc": 100.0, "cycle_tolerance_adc": 15.0, "residual_tolerance_adc": 15.0},
        ),
        _profile(
            time,
            "rollback_partial_unload_50_percent",
            _partial_unload(time),
            _windows((10, 30), (30, 50), (50, 60)),
            {"loaded_target_adc": 100.0, "partial_target_adc": 50.0, "partial_tolerance_adc": 10.0, "final_residual_tolerance_adc": 10.0},
        ),
        _profile(
            time,
            "rollback_slow_drift_small_step",
            _slow_drift_small_step(time),
            _windows((5, 35), (35, 40), (55, 65)),
            {"step_target_adc": 20.0, "step_tolerance_adc": 12.0, "drift_residual_tolerance_adc": 15.0},
        ),
    ]
    for case in cases:
        if not np.allclose(case["truth_total"], (case["values"] - case["reference"]).sum(axis=1)):
            raise AssertionError(case["name"])
    return cases


if __name__ == "__main__":
    for case in generate_cases():
        print(case["name"], float(case["truth_total"].max()), case["event_windows"])
