# -*- coding: utf-8 -*-
"""带瞬态保护的 K9 研究实现。

与 toolbox plugins/k9_creep_observer._compensate 的关系：
    - guard=None（或 enabled=False）时逐帧等价于工具箱实现（用于对拍）。
    - guard 启用时，只增加一层「瞬态帧状态隔离」：

污染链路（cpp creep_observer.cpp）：
    104-105  y_floor 下行瞬时跟随
    111-113  idle 成立 -> load_dwell 归零、zero 更新
    174-180  idle_k -> x_slow 按 tau_r_slow_idle_s 快泄放

防护规则（逐通道、因果、无未来帧）：
    trigger = |slope| > spike_slope_thres          （slope 用原始输入计算）
    保护期 = 自触发起 spike_max_s 之内
    保护期内该通道用「上一帧有效输入」推进所有内部状态，输出仍用真实输入
    持续偏离超过 spike_max_s -> 判定为真实变化，恢复正常响应
"""
from __future__ import annotations

import math

import numpy as np

_DENOM_FLOOR = 1e-9

DEFAULT_PARAMS = {
    "r_slow_max": 0.2, "tau_r_slow_s": 150.0, "tau_slope_s": 1.0, "slope_gate_frac": 0.05,
    "slope_cap_frac": 0.05, "tau_zero_s": 8.0, "idle_frac": 0.05, "y_max_tau_s": 600.0,
    "r_fast": 0.01, "tau_c_fast_s": 40.0, "tau_r_fast_s": 6.0, "edge_slope_thres": 60.0,
    "edge_refract_s": 2.0, "edge_boost_s": 2.0, "tau_c_fast_boost_s": 2.0,
    "soft_unfreeze_s": 8.0, "hold_eps": 2.0, "hold_tau_s": 0.5, "slow_confirm_s": 5.0,
    "y_floor_tau_s": 300.0, "tau_r_slow_idle_s": 0.5, "bypass_release_frac": 1.15,
    "bypass_engage_frac": 1.25, "bypass_noise_sigma": 3.0, "bypass_base_tau_s": 10.0,
    "ramp_slope_min": 0.5, "ramp_full_s": 4.0,
}

GUARD_DEFAULTS = dict(enabled=True, spike_slope_thres=60.0, spike_max_s=0.30,
                      floor_confirm_s=0.50, idle_confirm_s=0.30, floor_deep_adc=200.0,
                      floor_guard=True, idle_guard=True, spike_guard=False)


def _coerce(params) -> dict:
    merged = dict(DEFAULT_PARAMS)
    if isinstance(params, dict):
        for key, value in params.items():
            if key in merged:
                merged[key] = value
    return {key: (float(value) if key != "param_set" else value) for key, value in merged.items()}


def _pos(value) -> float:
    number = float(value)
    return number if number > _DENOM_FLOOR else _DENOM_FLOOR


def compensate(values: np.ndarray, time_axis: np.ndarray, params: dict, guard: dict | None = None):
    p = _coerce(params)
    g = dict(GUARD_DEFAULTS)
    if guard:
        g.update(guard)
    guard_on = bool(g["enabled"])

    r_fast = p["r_fast"]; tau_c_fast_s = _pos(p["tau_c_fast_s"]); tau_r_fast_s = _pos(p["tau_r_fast_s"])
    r_slow_max = p["r_slow_max"]; tau_r_slow_s = _pos(p["tau_r_slow_s"]); tau_slope_s = _pos(p["tau_slope_s"])
    slope_gate_frac = p["slope_gate_frac"]; slope_cap_frac = p["slope_cap_frac"]
    tau_zero_s = _pos(p["tau_zero_s"]); idle_frac = p["idle_frac"]; y_max_tau_s = _pos(p["y_max_tau_s"])
    edge_slope_thres = p["edge_slope_thres"]; edge_refract_s = p["edge_refract_s"]
    edge_boost_s = p["edge_boost_s"]; tau_c_fast_boost_s = _pos(p["tau_c_fast_boost_s"])
    soft_unfreeze_s = p["soft_unfreeze_s"]; hold_eps = p["hold_eps"]; hold_tau_s = _pos(p["hold_tau_s"])
    slow_confirm_s = p["slow_confirm_s"]; y_floor_tau_s = _pos(p["y_floor_tau_s"])
    tau_r_slow_idle_s = p["tau_r_slow_idle_s"]; bypass_release_frac = p["bypass_release_frac"]
    bypass_engage_frac = p["bypass_engage_frac"]; bypass_noise_sigma = p["bypass_noise_sigma"]
    bypass_base_tau_s = _pos(p["bypass_base_tau_s"]); ramp_slope_min = p["ramp_slope_min"]
    ramp_full_s = p["ramp_full_s"]
    spike_thres = float(g["spike_slope_thres"]); spike_max_s = float(g["spike_max_s"])

    n_rows, n_ch = values.shape
    out = values.copy()
    if n_rows == 0 or n_ch == 0:
        return out, {"guard_events": np.zeros(0, dtype=int), "bypass_mask": np.zeros(0, dtype=bool)}

    first = np.where(np.isfinite(values[0]), values[0], 0.0)
    zero = first.copy(); y_max = np.zeros(n_ch); v_lp = first.copy(); v_fast_lp = first.copy()
    y_floor = first.copy(); load_dwell = np.zeros(n_ch); ramp_dwell = np.zeros(n_ch)
    t_edge = np.full(n_ch, 1e6); applied = np.zeros(n_ch); x_fast = np.zeros(n_ch); x_slow = np.zeros(n_ch)
    total_baseline = float(first.sum()); total_noise = 0.0
    last_ts = float(time_axis[0])
    refract = max(edge_refract_s, edge_boost_s)

    v_used = first.copy()
    spike_elapsed = np.zeros(n_ch)
    below_elapsed = np.zeros(n_ch)
    idle_elapsed = np.zeros(n_ch)
    guarded_frames = np.zeros(n_rows, dtype=int)
    guard_frames_total = 0
    fast_release_events = 0

    for index in range(1, n_rows):
        raw = values[index]
        row_finite = bool(np.isfinite(raw).all())
        v = raw if row_finite else np.where(np.isfinite(raw), raw, 0.0)
        dt = float(time_axis[index]) - last_ts
        last_ts = float(time_axis[index])
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.0
        else:
            dt = min(dt, 0.1)
        if not row_finite:
            dt = 0.0

        bypass_frame = False
        if dt > 0.0:
            total = float(v.sum())
            if np.isfinite(total):
                alpha = min(dt / bypass_base_tau_s, 1.0)
                release = max(bypass_release_frac * total_baseline, total_baseline + bypass_noise_sigma * total_noise)
                engage = max(bypass_engage_frac * total_baseline, total_baseline + (bypass_noise_sigma + 1.0) * total_noise)
                if total < engage:
                    total_baseline += alpha * (total - total_baseline)
                    total_noise += alpha * (abs(total - total_baseline) - total_noise)
                    if total_noise < 0.0:
                        total_noise = 0.0
                if total < release:
                    bypass_frame = True

        if dt > 0.0:
            slope_real = (v - v_lp) / tau_slope_s
            if guard_on and g.get("spike_guard"):
                trigger = np.abs(slope_real) > spike_thres
                spike_elapsed = np.where(trigger, spike_elapsed + dt, 0.0)
                protect = trigger & (spike_elapsed <= spike_max_s)
                if protect.any():
                    guarded_frames[index] = 1
                    guard_frames_total += 1
                v = np.where(protect, v_used, v)
            v_used = v.copy()

            v_eff = v
            y_decay = math.exp(-dt / y_max_tau_s)
            y0 = v_eff - zero
            below = y0 < y_floor
            if guard_on and g.get("floor_guard"):
                # y_floor 下行跟随需持续确认：短时浅跌破（扰动）不改变基线；
                # 深跌破（真实卸载）立即跟随，避免卸载响应被延迟
                below_elapsed = np.where(below, below_elapsed + dt, 0.0)
                deep = (y_floor - y0) > float(g["floor_deep_adc"])
                follow = below & (deep | (below_elapsed >= float(g["floor_confirm_s"])))
                y_floor = np.where(below, np.where(follow, y0, y_floor),
                                   y_floor + (dt / y_floor_tau_s) * (y0 - y_floor))
            else:
                y_floor = np.where(below, y0, y_floor + (dt / y_floor_tau_s) * (y0 - y_floor))
            y_max = np.maximum(y_max * y_decay, np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            idle = (y0 - y_floor) < idle_frac * span
            load_dwell = np.where(idle, 0.0, load_dwell + dt)
            zero = np.where(idle, zero + (dt / tau_zero_s) * (v_eff - zero), zero)
            y = v_eff - zero
            slope = (v_eff - v_lp) / tau_slope_s
            e_now = np.maximum(y - x_fast - x_slow, 0.0)
            t_edge = t_edge + dt
            fire = (slope > edge_slope_thres) & (t_edge > refract)
            t_edge = np.where(fire, 0.0, t_edge)
            ramp_on = (slope > ramp_slope_min) & (slope <= edge_slope_thres)
            ramp_dwell = np.where(ramp_on, ramp_dwell + dt, 0.0)
            load_dwell = np.where(np.abs(slope) > edge_slope_thres, 0.0, load_dwell)
            if edge_boost_s > 0.0:
                tc1 = np.where(t_edge < edge_boost_s, tau_c_fast_boost_s, tau_c_fast_s)
            else:
                tc1 = np.full(n_ch, tau_c_fast_s)
            if ramp_full_s > 0.0:
                weight = np.minimum(ramp_dwell / ramp_full_s, 1.0)
                tc1 = tc1 * (1.0 - weight) + tau_c_fast_boost_s * weight
            dx1_rate = np.where(e_now > 0.0, (r_fast * e_now - x_fast) / tc1, -x_fast / tau_r_fast_s)
            x_fast = np.maximum(x_fast + dx1_rate * dt, 0.0)
            v_lp = v_lp + (dt / tau_slope_s) * (v_eff - v_lp)

            e = np.maximum(y - x_fast - x_slow, 0.0)
            span_k = np.maximum(y_max - y_floor, 1.0)
            idle_k = (y - y_floor) < idle_frac * span_k
            e_pos = e > 0.0
            base = np.maximum(e, 1.0)
            cap = slope_cap_frac * base
            if soft_unfreeze_s > 0.0:
                soft_ok = (1.0 - np.exp(-t_edge / soft_unfreeze_s)) > 0.5
            else:
                soft_ok = t_edge > 0.0
            dwell_ok = load_dwell >= slow_confirm_s
            gate_ok = np.abs(slope) < slope_gate_frac * base
            dx2 = np.clip(slope - dx1_rate, -cap, cap)
            x_slow = x_slow + np.where(e_pos & dwell_ok & soft_ok & gate_ok, dt * dx2, 0.0)
            x_slow = np.where(e_pos, np.minimum(np.maximum(x_slow, 0.0), r_slow_max * base), x_slow)
            release_mask = (~e_pos) | idle_k
            if guard_on and g.get("idle_guard"):
                # 快速泄放需 idle 持续确认：瞬时 idle 只用慢泄放，避免被瞬态骗走补偿量
                idle_elapsed = np.where(idle_k, idle_elapsed + dt, 0.0)
                idle_fast_ok = idle_k & (idle_elapsed >= float(g["idle_confirm_s"]))
            else:
                idle_fast_ok = idle_k
            if idle_fast_ok.any():
                fast_release_events += 1
            tau_r2 = np.where(idle_fast_ok & (tau_r_slow_idle_s > 0.0), tau_r_slow_idle_s, tau_r_slow_s)
            tau_r2 = np.where(tau_r2 > _DENOM_FLOOR, tau_r2, _DENOM_FLOOR)
            x_slow = np.where(release_mask, np.maximum(x_slow - (dt / tau_r2) * x_slow, 0.0), x_slow)

            if hold_eps > 0.0:
                slope_a = (v_eff - v_fast_lp) / hold_tau_s
                v_fast_lp = v_fast_lp + (dt / hold_tau_s) * (v_eff - v_fast_lp)
                d_des = x_fast + x_slow
                a_up = applied + dt * np.maximum(slope_a + hold_eps, 0.0)
                a_dn = applied + dt * (slope_a - hold_eps)
                a = np.where(d_des > applied, np.minimum(d_des, a_up), np.maximum(d_des, a_dn))
                applied = np.where(e_pos, np.minimum(np.maximum(a, 0.0), np.maximum(d_des, 0.0)), d_des)

        if bypass_frame:
            applied = x_fast + x_slow
            out[index] = raw
        elif hold_eps > 0.0:
            out[index] = raw - applied
        else:
            out[index] = raw - x_fast - x_slow

    return out, {"guard_frames": int(guard_frames_total), "fast_release_frames": int(fast_release_events)}


__all__ = ["compensate", "DEFAULT_PARAMS", "GUARD_DEFAULTS"]
