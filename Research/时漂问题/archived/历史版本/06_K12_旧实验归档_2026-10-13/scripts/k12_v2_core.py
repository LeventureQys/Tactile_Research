"""Experimental causal total-domain event ledger around the unchanged K9 core."""
from collections import deque
import numpy as np

DEFAULTS = dict(enabled=True, window_s=.25, separation_s=.5,
                threshold_total_adc=45., noise_multiplier=4., drift_tau_s=8., confirm_s=.35,
                large_step_adc=2000., large_freeze_s=8., refractory_s=2.)


def compensate(values, time, params, k9_core, options=None):
    cfg = dict(DEFAULTS)
    cfg.update(options or {})
    if not cfg['enabled']:
        out, _ = k9_core(values, time, params)
        return out, dict(offset_total=np.zeros(len(time)), guard=np.zeros(len(time)),
                         large_guard=np.zeros(len(time)), events=[])
    recent = deque()
    history = deque()
    ledger = np.zeros(values.shape[1])
    drift = np.zeros(values.shape[1])
    previous_mean = None
    previous_t = float(time[0])
    previous_input = values[0].copy()
    blocked_until = float(time[0])
    refractory_until = float(time[0])
    adjusted = np.empty_like(values)
    offsets = np.empty_like(values)
    guard = np.zeros(len(time))
    large_guard = np.zeros(len(time))
    events = []
    pending = None
    for index, (stamp, row) in enumerate(zip(time, values)):
        stamp = float(stamp)
        dt = max(0., min(stamp - previous_t, .1))
        if not np.isfinite(row).all():
            adjusted[index] = row - ledger
            offsets[index] = ledger
            continue
        if abs(float((row - previous_input).sum())) >= cfg['large_step_adc']:
            ledger[:] = 0.
            drift[:] = 0.
            recent.clear()
            history.clear()
            previous_mean = None
            blocked_until = stamp + cfg['large_freeze_s']
            pending = None
        recent.append((stamp, row.copy()))
        while recent and recent[0][0] < stamp - cfg['window_s']:
            recent.popleft()
        mean = np.mean([item[1] for item in recent], axis=0)
        history.append((stamp, mean.copy()))
        while len(history) > 1 and history[1][0] <= stamp - cfg['separation_s']:
            history.popleft()
        ready = history and stamp - history[0][0] >= cfg['separation_s'] * .95
        if stamp < blocked_until:
            large_guard[index] = 1.
        elif ready:
            span = stamp - history[0][0]
            delta = mean - history[0][1]
            innovation = delta - drift * span
            totals = np.array([item[1].sum() for item in recent])
            noise = np.std(totals) / np.sqrt(max(len(totals), 1))
            threshold = max(cfg['threshold_total_adc'], cfg['noise_multiplier'] * noise)
            event_size = float(innovation.sum())
            if pending is not None:
                started, anchor_t, anchor, direction, slope_before = pending
                candidate = mean - anchor - slope_before * (stamp - anchor_t)
                if stamp - started >= cfg['confirm_s']:
                    size = float(candidate.sum())
                    if size * direction >= threshold:
                        ledger += candidate
                        refractory_until = stamp + cfg['refractory_s']
                        history.clear()
                        history.append((stamp, mean.copy()))
                        events.append(dict(time_s=stamp, delta_total_adc=size,
                                           threshold_adc=float(threshold), channels=candidate.tolist()))
                    pending = None
            elif stamp >= refractory_until and abs(event_size) >= threshold:
                pending = (stamp, history[0][0], history[0][1].copy(),
                           np.sign(event_size), drift.copy())
            elif stamp >= refractory_until and previous_mean is not None and dt > 0:
                slope = (mean - previous_mean) / dt
                weight = 1. - np.exp(-dt / cfg['drift_tau_s'])
                drift += weight * (slope - drift)
        if stamp < refractory_until:
            guard[index] = 1.
        adjusted[index] = row - ledger
        offsets[index] = ledger
        previous_mean = mean.copy()
        previous_t = stamp
        previous_input = row.copy()
    background, _ = k9_core(adjusted, time, params)
    return background + offsets, dict(offset_total=offsets.sum(axis=1), guard=guard,
                                     large_guard=large_guard, events=events)
