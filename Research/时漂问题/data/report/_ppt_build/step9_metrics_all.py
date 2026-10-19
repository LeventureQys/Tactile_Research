"""Metrics for all six datasets, using one shared definition set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import save_json  # noqa: E402

HERE = Path(__file__).parent
ORDER = ["palm1", "palm2", "fing1", "fing2", "four1", "four2"]
ALL = json.loads((HERE / "all_sessions.json").read_text(encoding="utf-8"))

metrics: dict = {}
for key in ORDER:
    d = np.load(HERE / f"{key}_curves.npz")
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    t1 = float(t[-1])

    # load completion: first crossing of 50 % of the settled median
    settled_med = float(np.median(raw[int(0.3 * raw.size):]))
    idx = int(np.argmax(raw > 0.5 * settled_med))
    t_load = float(t[idx]) if idx > 0 else 0.0
    t0 = t_load + 15.0
    m = t >= t0

    def plateau(a, b, y):
        mm = (t >= a) & (t <= b)
        return float(np.mean(y[mm])) if mm.any() else float("nan")

    raw_start, raw_end = plateau(t0, t0 + 6, raw), plateau(t1 - 6, t1, raw)
    algo_start, algo_end = plateau(t0, t0 + 6, algo), plateau(t1 - 6, t1, algo)

    A = np.vstack([t[m] - t0, np.ones(m.sum())]).T
    kr = float(np.linalg.lstsq(A, raw[m], rcond=None)[0][0])
    ka = float(np.linalg.lstsq(A, algo[m], rcond=None)[0][0])

    def detrend_std(y, from_s, win_s=5.0):
        """Short-window standard deviation over the settled hold phase only."""
        fs = 1.0 / float(np.mean(np.diff(t)))
        w = max(int(win_s * fs), 2)
        mask = t >= from_s
        seg = y[mask]
        n = (seg.size // w) * w
        if n < w:
            return float("nan")
        blk = seg[:n].reshape(-1, w)
        return float(np.mean(np.std(blk, axis=1)))

    def max_rise(y, lo, hi):
        best = (-1e9, 0.0)
        for x in np.arange(lo, hi - 1.0, 0.1):
            dv = float(np.interp(x + 1.0, t, y) - np.interp(x, t, y))
            if dv > best[0]:
                best = (dv, x)
        return {"rise_mN": best[0], "at_s": best[1]}

    lo_scan = max(t0 + 8.0, 25.0)
    ev_raw = max_rise(raw, lo_scan, t1 - 5)
    ev_algo = max_rise(algo, lo_scan, t1 - 5)

    # step at the raw event location, measured on both signals
    at = ev_raw["at_s"]
    if key == "palm1":          # fixed, documented window around the 61.5 s step
        pre, post = (61.05, 61.45), (61.55, 62.05)
    else:
        pre = (at - 1.0, at - 0.1)
        post = (at + 0.1, at + 1.5)
    step_raw = plateau(*post, raw) - plateau(*pre, raw)
    step_algo = plateau(*post, algo) - plateau(*pre, algo)

    # for the transient page: pre-event baseline and the level 30 s after the event
    if key == "palm1":
        base_win, after_probe = (58.0, 61.4), 90.0
        base_raw, base_algo = plateau(*base_win, raw), plateau(*base_win, algo)
        after_raw = plateau(after_probe - 2, after_probe + 2, raw)
        after_algo = plateau(after_probe - 2, after_probe + 2, algo)
    else:
        base_win, after_probe = pre, None
        base_raw, base_algo, after_raw, after_algo = (
            plateau(*pre, raw), plateau(*pre, algo), None, None)

    # late-window drift after the event (baseline poisoning check)
    tail_a, tail_b = t1 - 60.0, t1 - 5.0

    metrics[key] = {
        "key": key, "part": ALL[key]["part"], "folder": ALL[key]["folder"],
        "cells": ALL[key]["cells"], "duration_s": t1, "frames": ALL[key]["frames"],
        "fps": ALL[key]["fps"], "load_done_s": t_load, "settled_from_s": t0,
        "raw": {"start_mN": raw_start, "end_mN": raw_end, "drift_mN": raw_end - raw_start,
                "slope_mN_per_min": kr * 60, "lin_total_mN": kr * (t1 - t0),
                "detrend_std_mN": detrend_std(raw, t0), "mean_mN": float(np.mean(raw[m]))},
        "algo": {"start_mN": algo_start, "end_mN": algo_end, "drift_mN": algo_end - algo_start,
                 "slope_mN_per_min": ka * 60, "lin_total_mN": ka * (t1 - t0),
                 "detrend_std_mN": detrend_std(algo, t0), "mean_mN": float(np.mean(algo[m]))},
        "event": {"at_s": at, "raw_1s_mN": ev_raw["rise_mN"], "algo_1s_mN": ev_algo["rise_mN"],
                  "step_pre_s": list(pre), "step_post_s": list(post),
                  "step_raw_mN": step_raw, "step_algo_mN": step_algo,
                  "base_window_s": list(base_win), "base_raw_mN": base_raw, "base_algo_mN": base_algo,
                  "after_at_s": after_probe, "after_raw_mN": after_raw, "after_algo_mN": after_algo},
        "tail": {"window_s": [tail_a, tail_b],
                 "raw_mN": plateau(tail_a, tail_b, raw), "algo_mN": plateau(tail_a, tail_b, algo)},
    }
    it = metrics[key]
    it["suppression"] = {
        "drift_pct": float(100 * (1 - abs(it["algo"]["drift_mN"]) / abs(it["raw"]["drift_mN"])))
        if it["raw"]["drift_mN"] else None,
        "std_pct": float(100 * (1 - it["algo"]["detrend_std_mN"] / it["raw"]["detrend_std_mN"])),
    }

save_json(metrics, HERE / "metrics_all.json")

hdr = f"{'key':6} {'part':8} {'dur_s':>7} {'cells':>5} {'raw_start':>9} {'raw_end':>9} {'raw_drift':>9} {'algo_drift':>10} {'sup%':>6} {'raw_std':>8} {'algo_std':>8} {'ev_raw':>7} {'ev_algo':>7}"
print(hdr)
print("-" * len(hdr))
for key in ORDER:
    it = metrics[key]
    sup = f"{it['suppression']['drift_pct']:.0f}" if it['suppression']['drift_pct'] is not None else "n/a"
    print(f"{key:6} {it['part']:8} {it['duration_s']:7.1f} {it['cells']:5d} "
          f"{it['raw']['start_mN']:9.0f} {it['raw']['end_mN']:9.0f} {it['raw']['drift_mN']:9.0f} "
          f"{it['algo']['drift_mN']:10.0f} {sup:>6} {it['raw']['detrend_std_mN']:8.1f} "
          f"{it['algo']['detrend_std_mN']:8.1f} {it['event']['raw_1s_mN']:7.0f} {it['event']['algo_1s_mN']:7.0f}")
print("\nper-dataset 30 s blocks (mN) — raw | algo")
for key in ORDER:
    d = np.load(HERE / f"{key}_curves.npz")
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    print(f"\n--- {key} ({metrics[key]['part']}) ---")
    block = 30.0
    for start in np.arange(0, t[-1], block):
        mm = (t >= start) & (t < start + block)
        if mm.sum() == 0:
            continue
        print(f"  {start:6.0f}-{min(start+block, t[-1]):6.0f}s  raw {raw[mm].mean():8.0f}  algo {algo[mm].mean():8.0f}"
              f"  raw_min {raw[mm].min():8.0f}  raw_max {raw[mm].max():8.0f}")
