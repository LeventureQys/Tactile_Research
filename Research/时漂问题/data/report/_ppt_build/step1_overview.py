"""Overview pass: per-session totals, raw-vs-algo force curves, saved as npz for later plotting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import block_stats, load_matrix, save_json, summarize  # noqa: E402

REPORT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\report\右手掌-new_param")
OUT = Path(__file__).parent
SESSIONS = {
    "dev1": REPORT / "设备1-2026-10-15_15-58-21-61s左右数据剧烈上抬",
    "dev2": REPORT / "设备2-2026-10-15_16-31-43",
}

summary: dict = {}
for key, folder in SESSIONS.items():
    print(f"=== {key}: {folder.name} ===", flush=True)
    t_total, f_total, _v_total, meta_total = load_matrix(folder / "device_001_forcetotal_seg0.csv")
    t_raw, f_raw, v_raw, meta_raw = load_matrix(folder / "device_001_mapforce_raw_seg0.csv")
    t_algo, f_algo, v_algo, meta_algo = load_matrix(folder / "device_001_mapforce_algo_seg0.csv")

    sum_raw = v_raw.sum(axis=1)
    sum_algo = v_algo.sum(axis=1)
    print("frames", t_total.size, t_raw.size, t_algo.size)
    print("meta roles:", meta_total.get("流角色"), meta_raw.get("流角色"), meta_algo.get("流角色"))

    np.savez_compressed(
        OUT / f"{key}_curves.npz",
        t=t_total,
        force_total=f_total,
        raw_sum=sum_raw,
        algo_sum=sum_algo,
        raw_max=v_raw.max(axis=1),
        algo_max=v_algo.max(axis=1),
        raw_pos=(v_raw > 0).sum(axis=1),
        algo_pos=(v_algo > 0).sum(axis=1),
    )

    info = {
        "folder": folder.name,
        "time": summarize(t_total),
        "force_total_mN": {
            "first": float(f_total[0]),
            "last": float(f_total[-1]),
            "min": float(f_total.min()),
            "max": float(f_total.max()),
            "median": float(np.median(f_total)),
        },
        "raw_sum_mN": {
            "first": float(sum_raw[0]),
            "last": float(sum_raw[-1]),
            "min": float(sum_raw.min()),
            "max": float(sum_raw.max()),
            "median": float(np.median(sum_raw)),
        },
        "algo_sum_mN": {
            "first": float(sum_algo[0]),
            "last": float(sum_algo[-1]),
            "min": float(sum_algo.min()),
            "max": float(sum_algo.max()),
            "median": float(np.median(sum_algo)),
        },
        "raw_minus_total_absmax": float(np.max(np.abs(sum_raw - f_total))),
        "algo_minus_total_absmax": float(np.max(np.abs(sum_algo - f_total))),
        "zero_tail_raw": block_stats(t_raw, sum_raw, t_raw[-1] - 60, t_raw[-1]),
        "zero_tail_algo": block_stats(t_algo, sum_algo, t_algo[-1] - 60, t_algo[-1]),
    }
    summary[key] = info
    print(json.dumps(info, ensure_ascii=False, indent=2))

save_json(summary, OUT / "overview.json")
print("saved", OUT / "overview.json")
