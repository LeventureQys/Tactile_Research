"""Human-readable roll-up of metrics_all.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
M = json.loads((HERE / "metrics_all.json").read_text(encoding="utf-8"))
A = json.loads((HERE / "all_sessions.json").read_text(encoding="utf-8"))
ORDER = ["palm1", "palm2", "fing1", "fing2", "four1", "four2"]

print(f"{'key':6} {'part':8} {'load_done':>9} {'settled':>8} {'ev@':>8} {'ev_raw1s':>8} {'ev_algo1s':>9} "
      f"{'step_raw':>8} {'step_algo':>9} {'tail_raw':>9} {'tail_algo':>9}")
for k in ORDER:
    v = M[k]
    print(f"{k:6} {v['part']:8} {v['load_done_s']:9.1f} {v['settled_from_s']:8.1f} {v['event']['at_s']:8.1f} "
          f"{v['event']['raw_1s_mN']:8.0f} {v['event']['algo_1s_mN']:9.0f} {v['event']['step_raw_mN']:8.0f} "
          f"{v['event']['step_algo_mN']:9.0f} {v['tail']['raw_mN']:9.0f} {v['tail']['algo_mN']:9.0f}")

print("\nrelative drift (raw | algo), % of the settled start level:")
for k in ORDER:
    v = M[k]
    rr = 100 * v["raw"]["drift_mN"] / v["raw"]["start_mN"]
    aa = 100 * v["algo"]["drift_mN"] / v["algo"]["start_mN"]
    print(f"  {k:6} {v['part']:8} raw {rr:+7.2f}%   algo {aa:+7.2f}%   "
          f"std {v['raw']['detrend_std_mN']:6.1f} -> {v['algo']['detrend_std_mN']:6.1f} mN")

print("\nsession table:")
for k in ORDER:
    a, v = A[k], M[k]
    print(f"  {k:6} {v['part']:8} {v['duration_s']/60:5.1f} min  {v['frames']:6d} frames  "
          f"{v['fps']:6.1f} fps  {v['cells']:3d} cells")
