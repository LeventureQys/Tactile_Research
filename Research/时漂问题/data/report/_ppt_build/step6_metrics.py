"""Compute the client-facing metrics for both sessions and dump them to JSON."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import save_json  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "metrics.json"

SESSIONS = {
    "s1": {"key": "dev1", "label": "数据一（第 1 组录制）", "t_load_end": 20.0},
    "s2": {"key": "dev2", "label": "数据二（第 2 组录制）", "t_load_end": 20.0},
}

metrics: dict = {}
for sid, cfg in SESSIONS.items():
    d = np.load(HERE / f"{cfg['key']}_curves.npz")
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    t0 = cfg["t_load_end"]
    t1 = float(t[-1])
    m = t >= t0

    def val(y, x):
        return float(np.mean(y[(t >= x - 1.0) & (t <= x + 1.0)]))

    raw_start, raw_end = val(raw, t0 + 3), val(raw, t1 - 3)
    algo_start, algo_end = val(algo, t0 + 3), val(algo, t1 - 3)

    # slope over the settled window
    A = np.vstack([t[m] - t0, np.ones(m.sum())]).T
    kr = np.linalg.lstsq(A, raw[m], rcond=None)[0][0]
    ka = np.linalg.lstsq(A, algo[m], rcond=None)[0][0]

    def detrend_std(y, win_s=5.0):
        fs = 1.0 / np.mean(np.diff(t))
        w = int(win_s * fs)
        n = (y.size // w) * w
        blk = y[:n].reshape(-1, w)
        return float(np.mean(np.std(blk, axis=1)))

    # transient: maximal 1 s rise after the load phase, plus the 61 s event
    def max_rise(y, lo, hi):
        best = (-1e9, 0.0)
        for x in np.arange(lo, hi - 1.0, 0.1):
            dv = np.interp(x + 1.0, t, y) - np.interp(x, t, y)
            if dv > best[0]:
                best = (dv, x)
        return {"rise_mN": float(best[0]), "at_s": float(best[1])}

    ev_raw = max_rise(raw, max(25, 2.5 * t0), t1 - 5)
    ev_algo = max_rise(algo, max(25, 2.5 * t0), t1 - 5)

    # transient step = mean of the post-event plateau minus mean of the pre-event plateau
    def plateau(a, b, y):
        mm = (t >= a) & (t <= b)
        return float(np.mean(y[mm])) if mm.any() else float("nan")

    if sid == "s1":
        pre_win, post_win = (61.05, 61.45), (61.55, 62.05)
        base_win = (58.0, 61.4)
        pre_r, post_r = plateau(*pre_win, raw), plateau(*post_win, raw)
        pre_a, post_a = plateau(*pre_win, algo), plateau(*post_win, algo)
        step_raw, step_algo = post_r - pre_r, post_a - pre_a
        after_win = 90.0
    else:
        # session 2 has no isolated step; use its largest 1 s excursion
        at = ev_raw["at_s"]
        pre_win, post_win = (at - 5.0, at - 0.5), (at + 0.5, at + 5.0)
        base_win = pre_win
        step_raw, step_algo = ev_raw["rise_mN"], ev_algo["rise_mN"]
        after_win = None

    metrics[sid] = {
        "label": cfg["label"],
        "duration_s": t1,
        "frames": int(t.size),
        "settled_window": [t0, t1],
        "raw": {"start_mN": raw_start, "end_mN": raw_end, "drift_mN": raw_end - raw_start,
                "slope_mN_per_min": float(kr * 60), "lin_total_mN": float(kr * (t1 - t0)),
                "detrend_std_mN": detrend_std(raw), "mean_mN": float(np.mean(raw[m]))},
        "algo": {"start_mN": algo_start, "end_mN": algo_end, "drift_mN": algo_end - algo_start,
                 "slope_mN_per_min": float(ka * 60), "lin_total_mN": float(ka * (t1 - t0)),
                 "detrend_std_mN": detrend_std(algo), "mean_mN": float(np.mean(algo[m]))},
        "max_1s_rise": {"raw": ev_raw, "algo": ev_algo},
        "event_step": {
            "pre_window_s": list(pre_win), "post_window_s": list(post_win),
            "base_window_s": list(base_win),
            "base_raw_mN": plateau(*base_win, raw), "base_algo_mN": plateau(*base_win, algo),
            "raw_mN": float(step_raw), "algo_mN": float(step_algo),
            "after_check_s": after_win,
            "after_raw_mN": float(plateau(after_win - 2, after_win + 2, raw)) if after_win else None,
            "after_algo_mN": float(plateau(after_win - 2, after_win + 2, algo)) if after_win else None,
        },
    }
    it = metrics[sid]
    it["suppression"] = {
        "drift_pct": float(100 * (1 - abs(it["algo"]["drift_mN"]) / abs(it["raw"]["drift_mN"]))),
        "lin_pct": float(100 * (1 - abs(it["algo"]["lin_total_mN"]) / abs(it["raw"]["lin_total_mN"]))),
        "std_pct": float(100 * (1 - it["algo"]["detrend_std_mN"] / it["raw"]["detrend_std_mN"])),
        "transient_pct": float(100 * (1 - ev_algo["rise_mN"] / ev_raw["rise_mN"])),
    }

save_json(metrics, OUT)

for sid, it in metrics.items():
    print(f"=== {sid} {it['label']} ===  {it['duration_s']:.0f}s  {it['frames']} frames")
    for k in ("raw", "algo"):
        v = it[k]
        print(f"  {k:5s} start {v['start_mN']:9.0f} end {v['end_mN']:9.0f} "
              f"drift {v['drift_mN']:+8.0f} mN  slope {v['slope_mN_per_min']:+7.2f} mN/min  "
              f"std(detrend) {v['detrend_std_mN']:6.1f} mN")
    print(f"  suppression: drift {it['suppression']['drift_pct']:.1f}%  lin {it['suppression']['lin_pct']:.1f}% "
          f"std {it['suppression']['std_pct']:.1f}%  transient {it['suppression']['transient_pct']:.1f}%")
    print(f"  max 1s rise raw {it['max_1s_rise']['raw']['rise_mN']:+.0f} mN @ {it['max_1s_rise']['raw']['at_s']:.1f}s | "
          f"algo {it['max_1s_rise']['algo']['rise_mN']:+.0f} mN @ {it['max_1s_rise']['algo']['at_s']:.1f}s")
    print(f"  event step pre{it['event_step']['pre_window_s']} -> post{it['event_step']['post_window_s']}: "
          f"raw {it['event_step']['raw_mN']:+.0f} mN  algo {it['event_step']['algo_mN']:+.0f} mN"
          + (f" | @{it['event_step']['after_check_s']}s: raw "
             f"{it['event_step']['after_raw_mN'] - it['event_step']['base_raw_mN']:+.0f} mN vs base, "
             f"algo {it['event_step']['after_algo_mN'] - it['event_step']['base_algo_mN']:+.0f} mN vs base"
             if it['event_step']['after_check_s'] else ""))
print("\nsaved", OUT)
