"""Extract curves + metrics for every dataset under data/report (all parts)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import load_matrix, load_stream, save_json, summarize  # noqa: E402

HERE = Path(__file__).parent
REPORT = HERE.parent

DATASETS = [
    ("palm1", "右手掌", "右手掌-new_param/设备1-2026-10-15_15-58-21-61s左右数据剧烈上抬"),
    ("palm2", "右手掌", "右手掌-new_param/设备2-2026-10-15_16-31-43"),
    ("fing1", "右指腹", "右指腹-new_param/设备1-2026-10-15_16-55-27"),
    ("fing2", "右指腹", "右指腹-new_param/设备2-2026-10-15_17-05-47"),
    ("four1", "四指指腹", "四指指腹-new_param/设备1-2026-10-15_17-19-21"),
    ("four2", "四指指腹", "四指指腹-new_param/设备2-2026-10-15_17-25-50"),
]

out: dict = {}
for key, part, rel in DATASETS:
    folder = REPORT / rel
    if not folder.exists():
        print(f"!! missing {rel}")
        continue
    print(f"=== {key} ({part}) {folder.name} ===", flush=True)
    total = load_stream(folder / "device_001_forcetotal_seg0.csv")
    t = total.iloc[:, 1].to_numpy(dtype=np.float64)
    f_total = total.iloc[:, 4].to_numpy(dtype=np.float64)
    _t_raw, _fr, v_raw, _m1 = load_matrix(folder / "device_001_mapforce_raw_seg0.csv")
    _t_algo, _fa, v_algo, _m2 = load_matrix(folder / "device_001_mapforce_algo_seg0.csv")
    raw = v_raw.sum(axis=1)
    algo = v_algo.sum(axis=1)
    np.savez_compressed(HERE / f"{key}_curves.npz", t=t, force_total=f_total,
                        raw_sum=raw, algo_sum=algo,
                        raw_max=v_raw.max(axis=1), algo_max=v_algo.max(axis=1),
                        raw_pos=(v_raw > 0).sum(axis=1), algo_pos=(v_algo > 0).sum(axis=1))
    meta = summarize(t)
    info = {
        "key": key, "part": part, "folder": rel, "cells": int(v_raw.shape[1]),
        "duration_s": meta["duration_s"], "frames": meta["frames"], "fps": meta["fps"],
        "raw_first": float(raw[0]), "raw_last": float(raw[-1]),
        "algo_first": float(algo[0]), "algo_last": float(algo[-1]),
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "algo_min": float(algo.min()), "algo_max": float(algo.max()),
        "raw_median_settled": float(np.median(raw[len(raw) // 3:])),
        "algo_median_settled": float(np.median(algo[len(algo) // 3:])),
    }
    out[key] = info
    print(json.dumps(info, ensure_ascii=False))
    print(f"   head raw {raw[:50].mean():.0f} / tail raw {raw[-500:].mean():.0f}")

save_json(out, HERE / "all_sessions.json")
print("\nsaved all_sessions.json")
