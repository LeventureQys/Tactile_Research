# -*- coding: utf-8 -*-
"""K10 在「右手掌-2026-10-15_15-22-01」上的调参扫描。

阶段：
  diag  —— 修正版瞬态下探检测（0.2 s 中值滤波后取差分，通道级深度），用于标定 guard_floor_deep / guard_floor_confirm_s
  sweep —— K10 候选扫描：guard 4 参对照 + K9 现场 8 参档位对照

口径：
  - 输入流 device_001_seg000.csv（ADC 显示域，本会话 algorithm.enabled=false，无算法输出可对照）
  - 时间轴按 observed_fps 重建（录制 elapsed 列 38% 重复）
  - 恒载指标：分析窗自 t=30 s 起（跳过 0~5 s 空载→加载过渡），参考值取窗内首个 10 s 中值，
    指标 = 10 s 窗口中值相对参考值的最大绝对偏移 / 末段漂移 / 末 120 s 斜率
  - 压缩比 = 末段补偿后总值 / 末段输入总值（防"靠多扣把漂移做平"）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from guarded_k9 import DEFAULT_PARAMS, GUARD_DEFAULTS, compensate  # noqa: E402

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右手掌-2026-10-15_15-22-01")
OUT = HERE / "palm_tune_20261015"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 100.389239040211
ANALYSIS_START = 30.0
BIN_S = 10.0
TAIL_S = 120.0
SMOOTH_S = 0.2
DIP_THRESHOLD = 150.0
DIP_EXCLUDE_S = 30.0

# 现场可改写的 8 个寄存器参数（其余 19 项固定为工作树默认值）
PALM_R_FIRST_ARTICLE = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
                            slope_cap_frac=0.09, r_slow_max=0.2, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)
PALM_R_PALM9 = dict(PALM_R_FIRST_ARTICLE, slope_cap_frac=0.012, r_slow_max=0.25)
PLAN_V4_K9 = dict(r_fast=0.03, tau_c_fast_s=2.0, tau_r_fast_s=0.1, r_slow_max=0.1, slope_cap_frac=0.02,
                  soft_unfreeze_s=2.0, slow_confirm_s=3.0, tau_r_slow_idle_s=0.2)
PALM_FORCE_B = dict(r_fast=0.05, tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=2.0,
                    slope_cap_frac=0.009, r_slow_max=0.35, tau_r_fast_s=6.0, tau_r_slow_idle_s=0.5)


def load_main() -> tuple[np.ndarray, np.ndarray]:
    path = FOLDER / "device_001_seg000.csv"
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    frame = pd.read_csv(path, skiprows=skip, header=0, engine="c")
    x = np.asarray(frame.iloc[:, 3:], dtype=float)
    t = np.arange(x.shape[0], dtype=float) / FPS
    return t, x


def smooth(sig: np.ndarray, win_s: float = SMOOTH_S) -> np.ndarray:
    win = max(3, int(round(win_s * FPS)))
    return pd.Series(sig).rolling(win, center=True, min_periods=1).median().to_numpy()


def med_in(t: np.ndarray, sig: np.ndarray, a: float, b: float):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def hold_metrics(t: np.ndarray, total: np.ndarray, start: float = ANALYSIS_START) -> dict:
    ref = med_in(t, total, start, start + BIN_S)
    wins = []
    a = start
    while a + BIN_S <= t[-1]:
        value = med_in(t, total, a, a + BIN_S)
        if value is not None:
            wins.append((a + BIN_S / 2, value))
        a += BIN_S
    if not wins:
        return {"valid": False}
    arr = np.asarray(wins, dtype=float)
    tail = med_in(t, total, t[-1] - BIN_S, t[-1])
    late = arr[arr[:, 0] > t[-1] - TAIL_S]
    slope = float(np.polyfit(late[:, 0], late[:, 1], 1)[0]) if late.shape[0] > 1 else None
    return {"valid": True, "ref_adc": round(ref, 1), "end_adc": round(tail, 1),
            "endpoint_drift_adc": round(tail - ref, 1),
            "max_abs_offset_adc": round(float(np.max(np.abs(arr[:, 1] - ref))), 1),
            "tail_slope_adc_s": round(slope, 4) if slope is not None else None,
            "windows": arr}


def stage_diag(t: np.ndarray, x: np.ndarray) -> dict:
    total = x.sum(1)
    sm_total = smooth(total)
    diff = np.diff(sm_total)
    resid = total - sm_total
    sigma_total = float(np.median([np.std(resid[int(a * FPS):int(a * FPS) + int(FPS)])
                                   for a in np.arange(0.0, t[-1] - 1.0, 1.0)]))
    candidates = [i for i in np.flatnonzero(diff < -DIP_THRESHOLD) if t[i] >= DIP_EXCLUDE_S]
    merged = []
    for idx in candidates:
        if merged and t[idx] - merged[-1]["t_drop"] < 0.5:
            event = merged[-1]
            if sm_total[idx + 1] < event["trough"]:
                event.update(trough=float(sm_total[idx + 1]), t_trough=float(t[idx + 1]))
            continue
        merged.append(dict(t_drop=float(t[idx]), start_level=float(sm_total[idx]),
                           trough=float(sm_total[idx + 1]), t_trough=float(t[idx + 1])))
    rows = []
    sm_chan = np.column_stack([smooth(x[:, k]) for k in range(x.shape[1])])
    for event in merged:
        before = med_in(t, sm_total, max(0.0, event["t_drop"] - 10.0), event["t_drop"])
        back = med_in(t, sm_total, event["t_trough"] + 2.0, event["t_trough"] + 4.0)
        if before is None or back is None:
            continue
        depth = before - event["trough"]
        lo = int(max(0.0, event["t_drop"] - 5.0) * FPS)
        hi = int(min(t[-1], event["t_trough"] + 1.0) * FPS)
        chan_drop = np.clip(np.median(sm_chan[lo:max(lo + 1, int(event["t_drop"] * FPS))], axis=0)
                            - sm_chan[lo:max(lo + 1, hi)].min(axis=0), 0.0, None)
        rows.append(dict(t_drop=round(event["t_drop"], 2), depth_total_adc=round(float(depth), 1),
                         back_offset_adc=round(float(back - before), 1),
                         recovered=bool(abs(back - before) < max(30.0, 3 * sigma_total)),
                         chan_depth_max=round(float(chan_drop.max()), 1),
                         chan_depth_p99=round(float(np.percentile(chan_drop, 99)), 1),
                         chan_depth_p95=round(float(np.percentile(chan_drop, 95)), 1),
                         chan_drop_top=sorted([round(float(v), 1) for v in np.sort(chan_drop)[-5:]], reverse=True)))
    transient = [r for r in rows if r["recovered"]]
    info = dict(fps=FPS, frames=int(x.shape[0]), channels=int(x.shape[1]), duration_s=round(float(t[-1]), 2),
                smooth_window_s=SMOOTH_S, threshold_adc=DIP_THRESHOLD, exclude_before_s=DIP_EXCLUDE_S,
                noise_total_sigma_adc=round(sigma_total, 2),
                events=len(rows), transient_events=len(transient),
                max_chan_depth_transient=max([r["chan_depth_max"] for r in transient], default=0.0),
                max_depth_total_transient=max([r["depth_total_adc"] for r in transient], default=0.0))
    (OUT / "diag_dips.json").write_text(json.dumps(dict(info=info, events=rows), ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    for row in rows[:30]:
        print("   ", row)
    return info


def candidates() -> list[dict]:
    guard_off = dict(GUARD_DEFAULTS, enabled=False)
    guard_on = dict(GUARD_DEFAULTS, enabled=True)
    return [
        dict(tag="G00_guard_off", params=PALM_R_FIRST_ARTICLE, guard=guard_off),
        dict(tag="G01_guard_default", params=PALM_R_FIRST_ARTICLE, guard=guard_on),
        dict(tag="G02_floor_confirm_0.2", params=PALM_R_FIRST_ARTICLE, guard=dict(guard_on, floor_confirm_s=0.2)),
        dict(tag="G03_floor_confirm_1.0", params=PALM_R_FIRST_ARTICLE, guard=dict(guard_on, floor_confirm_s=1.0)),
        dict(tag="G04_idle_confirm_0.1", params=PALM_R_FIRST_ARTICLE, guard=dict(guard_on, idle_confirm_s=0.1)),
        dict(tag="G05_idle_confirm_1.0", params=PALM_R_FIRST_ARTICLE, guard=dict(guard_on, idle_confirm_s=1.0)),
        dict(tag="G06_floor_deep_100", params=PALM_R_FIRST_ARTICLE, guard=dict(guard_on, floor_deep_adc=100.0)),
        dict(tag="G07_floor_deep_800", params=PALM_R_FIRST_ARTICLE, guard=dict(guard_on, floor_deep_adc=800.0)),
        dict(tag="G08_spike_guard_on", params=PALM_R_FIRST_ARTICLE,
             guard=dict(guard_on, spike_guard=True, spike_slope_thres=60.0, spike_max_s=0.30)),
        dict(tag="P00_worktree_default", params={}, guard=guard_on),
        dict(tag="P01_plan_v4_k9", params=PLAN_V4_K9, guard=guard_on),
        dict(tag="P02_palm_first_article", params=PALM_R_FIRST_ARTICLE, guard=guard_on),
        dict(tag="P03_palm9_second", params=PALM_R_PALM9, guard=guard_on),
        dict(tag="P04_palm_force_B", params=PALM_FORCE_B, guard=guard_on),
        dict(tag="P05_rf_005_cap009", params=dict(PALM_R_FIRST_ARTICLE, r_fast=0.05, slope_cap_frac=0.009),
             guard=guard_on),
        dict(tag="P06_rf_002_cap002", params=dict(PALM_R_FIRST_ARTICLE, slope_cap_frac=0.02), guard=guard_on),
    ]


def stage_sweep(t: np.ndarray, x: np.ndarray) -> dict:
    input_total = x.sum(1)
    base = hold_metrics(t, input_total, ANALYSIS_START)
    rows = [dict(tag="INPUT_reference", params="input", guard="n/a",
                 max_abs_offset_adc=base["max_abs_offset_adc"],
                 endpoint_drift_adc=base["endpoint_drift_adc"], tail_slope_adc_s=base["tail_slope_adc_s"],
                 end_total_adc=base["end_adc"], compression_ratio=1.0)]
    for candidate in candidates():
        started = time.perf_counter()
        out, info = compensate(x.copy(), t, candidate["params"], candidate["guard"])
        elapsed = time.perf_counter() - started
        metrics = hold_metrics(t, out.sum(1), ANALYSIS_START)
        row = dict(tag=candidate["tag"],
                   guard_enabled=bool(candidate["guard"].get("enabled")),
                   guard=json.dumps(candidate["guard"], ensure_ascii=False),
                   max_abs_offset_adc=metrics["max_abs_offset_adc"],
                   endpoint_drift_adc=metrics["endpoint_drift_adc"],
                   tail_slope_adc_s=metrics["tail_slope_adc_s"],
                   end_total_adc=metrics["end_adc"],
                   compression_ratio=round(metrics["end_adc"] / base["end_adc"], 4),
                   guard_frames=info.get("guard_frames"), fast_release_frames=info.get("fast_release_frames"),
                   seconds=round(elapsed, 2))
        rows.append(row)
        print(f"{candidate['tag']:24s} offset={row['max_abs_offset_adc']:8.1f} "
              f"drift={row['endpoint_drift_adc']:8.1f} slope={row['tail_slope_adc_s']:+.3f} "
              f"ratio={row['compression_ratio']:.4f} ({elapsed:.1f}s)", flush=True)

    payload = dict(source=str(FOLDER), analysis_start_s=ANALYSIS_START, bin_s=BIN_S,
                   input_baseline=dict(ref_adc=base["ref_adc"], end_adc=base["end_adc"]),
                   candidates=rows)
    (OUT / "sweep_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT / 'sweep_result.json'}")
    return payload


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "diag"
    t, x = load_main()
    print(f"载入: {x.shape[0]} 帧 × {x.shape[1]} 通道, {t[-1]:.1f} s", flush=True)
    if stage == "diag":
        stage_diag(t, x)
    else:
        stage_sweep(t, x)


if __name__ == "__main__":
    main()
