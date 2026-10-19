"""Extract per-session curves (total force, raw sum, algo sum) and segment stats."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import load_matrix, load_stream, save_json, summarize  # noqa: E402

REPORT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\report\右手掌-new_param")
OUT = Path(__file__).parent
SESSIONS = {
    "dev1": REPORT / "设备1-2026-10-15_15-58-21-61s左右数据剧烈上抬",
    "dev2": REPORT / "设备2-2026-10-15_16-31-43",
}

summary: dict = {}
for key, folder in SESSIONS.items():
    print(f"=== {key} ===", flush=True)
    total = load_stream(folder / "device_001_forcetotal_seg0.csv")
    t = total.iloc[:, 1].to_numpy(dtype=np.float64)
    f_total = total.iloc[:, 4].to_numpy(dtype=np.float64)

    t_raw, _fr, v_raw, _m1 = load_matrix(folder / "device_001_mapforce_raw_seg0.csv")
    t_algo, _fa, v_algo, _m2 = load_matrix(folder / "device_001_mapforce_algo_seg0.csv")
    sum_raw = v_raw.sum(axis=1)
    sum_algo = v_algo.sum(axis=1)

    np.savez_compressed(
        OUT / f"{key}_curves.npz",
        t=t, force_total=f_total, raw_sum=sum_raw, algo_sum=sum_algo,
        raw_max=v_raw.max(axis=1), algo_max=v_algo.max(axis=1),
        raw_pos=(v_raw > 0).sum(axis=1), algo_pos=(v_algo > 0).sum(axis=1),
    )

    # 60 s block table
    blocks = []
    for start in np.arange(0, t[-1], 60.0):
        m = (t >= start) & (t < start + 60.0)
        if m.sum() == 0:
            continue
        blocks.append({
            "t0": float(start), "t1": float(min(start + 60.0, t[-1])),
            "raw": float(np.mean(sum_raw[m])),
            "algo": float(np.mean(sum_algo[m])),
            "total": float(np.mean(f_total[m])),
            "raw_max": float(np.max(sum_raw[m])),
            "algo_max": float(np.max(sum_algo[m])),
            "raw_min": float(np.min(sum_raw[m])),
            "algo_min": float(np.min(sum_algo[m])),
        })
    info = {
        "folder": folder.name,
        "time": summarize(t),
        "force_total_final": float(f_total[-1]),
        "sum_raw_final": float(sum_raw[-1]),
        "sum_algo_final": float(sum_algo[-1]),
        "sum_raw_max": float(sum_raw.max()),
        "sum_algo_max": float(sum_algo.max()),
        "blocks_60s": blocks,
    }
    summary[key] = info
    print(f"{key}: frames={t.size} dur={t[-1]:.1f}s raw_final={sum_raw[-1]:.0f} algo_final={sum_algo[-1]:.0f} "
          f"total_final={f_total[-1]:.0f}")

save_json(summary, OUT / "curves_summary.json")
for key, info in summary.items():
    print(f"\n--- {key} 60s blocks (mN) ---")
    print(f"{'window':>13} {'raw_mean':>10} {'algo_mean':>10} {'raw_min':>9} {'raw_max':>9} {'algo_min':>9} {'algo_max':>9}")
    for b in info["blocks_60s"]:
        print(f"{b['t0']:5.0f}-{b['t1']:5.0f}s {b['raw']:10.0f} {b['algo']:10.0f} "
              f"{b['raw_min']:9.0f} {b['raw_max']:9.0f} {b['algo_min']:9.0f} {b['algo_max']:9.0f}")
