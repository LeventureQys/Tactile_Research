# -*- coding: utf-8 -*-
"""追踪 194~196s 下探震荡对 K9 内部状态的影响。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
OUT = Path(__file__).resolve().parent
K9_IMPL = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\01_当前工作_K9\02_现场调参\09-21_09-26_工程A_v3.4")
sys.path.insert(0, str(K9_IMPL))
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

LIMIT_S = 250.0
TRACE_FROM, TRACE_TO = 190.0, 200.0


def load_stream(name: str) -> pd.DataFrame:
    path = FOLDER / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def main() -> None:
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    raw_params = session["algorithm"]["params"]
    names = {"r_slow_max", "slope_cap_frac", "slope_gate_frac", "r_fast", "tau_c_fast_s",
             "tau_r_fast_s", "tau_slope_s", "tau_zero_s", "idle_frac", "y_max_tau_s",
             "edge_slope_thres", "edge_refract_s", "edge_boost_s", "tau_c_fast_boost_s",
             "soft_unfreeze_s", "hold_eps", "hold_tau_s", "slow_confirm_s", "y_floor_tau_s",
             "tau_r_slow_idle_s", "bypass_release_frac", "bypass_engage_frac",
             "bypass_noise_sigma", "bypass_base_tau_s", "ramp_slope_min", "ramp_full_s",
             "tau_r_slow_s"}
    params = Params(**{k: float(v) for k, v in raw_params.items() if k in names})

    pre = load_stream("device_001_pre_seg0.csv")
    seg = load_stream("device_001_seg000.csv")
    values_all = np.asarray(pre.iloc[:, 3:], dtype=float)
    seg_all = np.asarray(seg.iloc[:, 3:], dtype=float)
    frame_index = np.asarray(pre.iloc[:, 2], dtype=float)
    time_all = (frame_index - frame_index[0]) / fps
    limit = int(np.searchsorted(time_all, LIMIT_S))

    observer = CreepObserverK9(params)
    observer._trace_frame = lambda *args: None
    rows = []
    n_ch = values_all.shape[1]
    for index in range(limit):
        stamp = float(time_all[index])
        row = values_all[index]
        out = observer.process(stamp, row)
        if TRACE_FROM <= stamp <= TRACE_TO:
            y = row - observer.zero
            span = np.maximum(observer.y_max - observer.y_floor, 1.0)
            idle = (y - observer.y_floor) < params.idle_frac * span
            e_now = np.maximum(y - observer.x_fast - observer.x_slow, 0.0)
            slope = observer._slope
            rows.append(dict(
                t=round(stamp, 4),
                pre_total=round(float(row.sum()), 1),
                out_total=round(float(out.sum()), 1),
                fw_total=round(float(seg_all[index].sum()), 1),
                x_fast=round(float(observer.x_fast.sum()), 2),
                x_slow=round(float(observer.x_slow.sum()), 2),
                applied=round(float(observer.applied.sum()), 2),
                y_floor_sum=round(float(observer.y_floor.sum()), 1),
                y_max_sum=round(float(observer.y_max.sum()), 1),
                zero_sum=round(float(observer.zero.sum()), 1),
                idle_channels=int(idle.sum()),
                e_zero_channels=int((e_now <= 0).sum()),
                edge_fire_channels=int((np.abs(slope) > params.edge_slope_thres).sum()),
                dwell_zero_channels=int((observer.load_dwell == 0).sum()),
                v_lp_sum=round(float(observer.v_lp.sum()), 1),
            ))
    (OUT / "state_result.json").write_text(json.dumps(dict(params=raw_params, rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"traced {len(rows)} frames in [{TRACE_FROM},{TRACE_TO}]s; N={n_ch}")


if __name__ == "__main__":
    main()
