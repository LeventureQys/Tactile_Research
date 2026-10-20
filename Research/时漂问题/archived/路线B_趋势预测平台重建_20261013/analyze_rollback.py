# -*- coding: utf-8 -*-
"""使用当前 route_b 接口分析回撤合成案例。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from route_b import CONFIG, route_b
from rollback_cases import generate_cases

HERE = Path(__file__).resolve().parent
PARAMS = dict(
    r_fast=0.03,
    tau_c_fast_s=2,
    tau_r_fast_s=0.1,
    r_slow_max=0.4,
    slope_cap_frac=0.02,
    slow_confirm_s=3,
    soft_unfreeze_s=2,
    tau_r_slow_idle_s=0.2,
)


def _median_window(time, signal, start, end):
    values = signal[(time >= start) & (time < end)]
    return float(np.median(values)) if values.size else float("nan")


def _metrics(case, routed, trace):
    time = case["time"]
    truth = case["truth_total"]
    baseline = _median_window(time, routed, 0.0, 5.0)
    delta = routed - baseline
    windows = case["event_windows"]
    window_outputs = []
    for window in windows:
        start, end = window["start_s"], window["end_s"]
        window_outputs.append(
            {
                "start_s": start,
                "end_s": end,
                "truth_total_adc": _median_window(time, truth, start, end),
                "output_delta_adc": _median_window(time, delta, start, end),
                "residual_adc": _median_window(time, delta - truth, start, end),
            }
        )
    is_hold_withdraw = case["name"] == "rollback_dispersed_100_hold15_withdraw"
    hold = _median_window(time, delta, 15.0, 27.0) if is_hold_withdraw else "N/A"
    withdrawal = _median_window(time, delta, 29.0, 35.0) if is_hold_withdraw else "N/A"
    retention = float(hold / 100.0) if is_hold_withdraw else "N/A"
    return {
        "baseline_adc": baseline,
        "hold_output_adc": hold,
        "hold_truth_adc": 100.0 if is_hold_withdraw else "N/A",
        "hold_retention": retention,
        "withdrawal_output_adc": withdrawal,
        "withdrawal_residual_adc": withdrawal,
        "truth_peak_abs_adc": float(np.max(np.abs(truth))),
        "output_peak_abs_adc": float(np.max(np.abs(delta))),
        "event_count": len(trace["events"]),
        "event_times_s": [round(event["time_s"], 4) for event in trace["events"]],
        "window_metrics": window_outputs,
        "acceptance": case["acceptance"],
    }


def analyze(case):
    routed_channels, trace = route_b(case["values"], case["time"], PARAMS)
    routed = routed_channels.sum(axis=1)
    return {
        "name": case["name"],
        "kind": case["kind"],
        "truth_total": [float(value) for value in case["truth_total"]],
        "event_windows": case["event_windows"],
        "acceptance_metrics": _metrics(case, routed, trace),
    }


def main():
    rows = [analyze(case) for case in generate_cases()]
    output = {
        "config": CONFIG,
        "params": PARAMS,
        "algorithm": "route_b_current_interface_rollback_analysis",
        "cases": rows,
        "notes": [
            "hold_retention = hold_output_adc / 100 ADC；withdrawal_residual_adc为撤载窗口输出相对初始基线的残差。",
            "连续+100/-100案例总量真值为0，另保留truth_peak_abs_adc揭示逐路变化。",
        ],
    }
    path = HERE / "route_b_rollback_results.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        metrics = row["acceptance_metrics"]
        retention = metrics["hold_retention"]
        residual = metrics["withdrawal_residual_adc"]
        retention_text = "N/A" if retention == "N/A" else "%.4f" % retention
        residual_text = "N/A" if residual == "N/A" else "%.4f" % residual
        print(
            row["name"],
            "hold_retention=" + retention_text,
            "withdrawal_residual_adc=" + residual_text,
            "events=%d" % metrics["event_count"],
        )


if __name__ == "__main__":
    main()
