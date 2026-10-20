# -*- coding: utf-8 -*-
"""路线B趋势预测平台的合成验证案例。"""
from __future__ import annotations

import numpy as np


FPS = 100
CHANNELS = 57
BASELINE = 8.0


def _timeline(duration_s: float = 120.0):
    return np.arange(0.0, duration_s, 1.0 / FPS, dtype=float)


def _case(name, time, load, kind, reference=None, noise_std=0.0, seed=0,
         measure_windows=None):
    time = np.asarray(time, dtype=float)
    load = np.asarray(load, dtype=float)
    if load.shape != (time.size, CHANNELS):
        raise ValueError("load must have shape (n_samples, 57)")
    rng = np.random.default_rng(seed)
    reference = np.full_like(load, BASELINE) if reference is None else np.asarray(reference, dtype=float)
    values = reference + load
    if noise_std:
        values = values + rng.normal(0.0, noise_std, values.shape)
    truth_total = np.sum(load, axis=1)
    return {
        "name": name,
        "time": time,
        "reference": reference,
        "values": values,
        "truth_total": truth_total,
        "measure_windows": measure_windows or [],
        "kind": kind,
    }


def _windows(*pairs):
    return [{"start_s": float(start), "end_s": float(end)} for start, end in pairs]


def _step(time, start, amount, channels=None):
    channels = np.arange(CHANNELS) if channels is None else np.asarray(channels)
    out = np.zeros((time.size, CHANNELS), dtype=float)
    active = np.flatnonzero(time >= start)
    out[np.ix_(active, channels)] = amount
    return out


def _ramp(time, start, duration, amount, channels=None):
    channels = np.arange(CHANNELS) if channels is None else np.asarray(channels)
    fraction = np.clip((time - start) / duration, 0.0, 1.0)
    out = np.zeros((time.size, CHANNELS), dtype=float)
    out[:, channels] = amount * fraction[:, None]
    return out


def _pulse_train(time, events, channels=None):
    channels = np.arange(CHANNELS) if channels is None else np.asarray(channels)
    out = np.zeros((time.size, CHANNELS), dtype=float)
    for start, amount, duration in events:
        active = (time >= start) & (time < start + duration)
        out[np.ix_(active, channels)] += amount
    return out


def _base_cases():
    time = _timeline()
    concentrated = np.array([0])
    dispersed = np.arange(0, CHANNELS, 3)
    cases = []
    cases.append(_case("positive_each_channel_500adc_step2s", time, _step(time, 2.0, 500.0), "positive_all_channel_calibration", measure_windows=_windows((0, 2), (2, 4))))
    cases.append(_case("positive_100adc_concentrated_step15s", time, _step(time, 15.0, 100.0, concentrated), "positive_concentrated", measure_windows=_windows((0, 2), (15, 20))))
    cases.append(_case("positive_100adc_dispersed_step15s", time, _step(time, 15.0, 100.0 / dispersed.size, dispersed), "positive_dispersed", measure_windows=_windows((0, 2), (15, 20))))
    cases.append(_case("positive_100adc_signed_split_step15s", time, _step(time, 15.0, 100.0 / 2, [0, 1]) + _step(time, 15.0, -100.0 / 2, [2, 3]) + _step(time, 15.0, 100.0, [4]), "positive_signed_split", measure_windows=_windows((0, 2), (15, 20))))
    events = [(15.0, 500.0, 5.0), (35.0, 500.0, 5.0), (55.0, 500.0, 5.0), (75.0, -500.0, 5.0), (95.0, -500.0, 5.0), (110.0, -500.0, 5.0)]
    cases.append(_case("positive_repeated_add_remove_three_cycles", time, _pulse_train(time, events, [0]), "positive_repeated", measure_windows=_windows((15, 20), (35, 40), (55, 60), (75, 80), (95, 100), (110, 115))))
    for duration in (0.2, 1.0, 3.0, 5.0):
        cases.append(_case(f"positive_500adc_ramp_{duration:g}s", time, _ramp(time, 15.0, duration, 500.0, dispersed), "positive_ramp", measure_windows=_windows((0, 2), (15, 15 + duration), (15 + duration + 1, 15 + duration + 3))))
    return cases


def _negative_cases():
    time = _timeline()
    total = time.size
    cases = []
    linear = np.full((total, CHANNELS), 5.7 / CHANNELS) * time[:, None]
    cases.append(_case("negative_linear_drift_5p7adc_per_s", time, linear, "negative_linear_drift"))
    linear = np.full((total, CHANNELS), 20.0 / CHANNELS) * time[:, None]
    cases.append(_case("negative_linear_drift_20adc_per_s", time, linear, "negative_linear_drift"))
    for tau in (20.0, 80.0):
        creep = (1000.0 * (1.0 - np.exp(-time / tau)))[:, None] / CHANNELS
        cases.append(_case(f"negative_exponential_creep_1000adc_tau{tau:g}s", time, np.repeat(creep, CHANNELS, axis=1), "negative_exponential_creep"))
    rate = 2.0 + 8.0 * (1.0 + np.sin(2.0 * np.pi * time / 45.0)) / 2.0
    smooth = np.cumsum(rate)[:, None] / FPS / CHANNELS
    cases.append(_case("negative_smooth_drift_changing_rate", time, np.repeat(smooth, CHANNELS, axis=1), "negative_smooth_drift"))
    rng = np.random.default_rng(41)
    white = rng.normal(0.0, 2.0, (total, CHANNELS))
    cases.append(_case("negative_white_noise", time, white, "negative_white_noise"))
    common = _step(time, 15.0, 100.0 / CHANNELS)
    cases.append(_case("negative_common_mode_step_57x100adc", time, common, "negative_common_mode_step", measure_windows=_windows((0, 2), (15, 20))))
    return cases


def generate_cases():
    """返回全部合成验证案例，每项包含统一字段。"""
    cases = _base_cases() + _negative_cases()
    _self_check(cases)
    return cases


def _self_check(cases):
    for case in cases:
        values = case["values"]
        reference = case["reference"]
        truth = case["truth_total"]
        if values.shape != (case["time"].size, CHANNELS):
            raise AssertionError(case["name"])
        if not np.allclose(truth, np.sum(values - reference, axis=1)):
            raise AssertionError(f"truth_total mismatch: {case['name']}")
    common = next(c for c in cases if c["kind"] == "negative_common_mode_step")
    dispersed = next(c for c in cases if c["kind"] == "positive_dispersed")
    if not np.allclose(common["truth_total"], dispersed["truth_total"]):
        raise AssertionError("common-mode and dispersed loads must share total truth")


if __name__ == "__main__":
    generated = generate_cases()
    print(f"validated {len(generated)} cases")
