# -*- coding: utf-8 -*-
"""K10 调参前置诊断：右手掌-2026-10-15_15-22-01 精细定标（第二步）。

输出：
  - 起步段形态（0~20 s，0.5 s 网格）
  - 漂移率随时间（20 s 窗口线性拟合），分早/中/末段统计
  - 噪声尺度：总值与逐通道的高频残差 σ
  - 通道级下探事件（深度 / 持续 / 是否恢复），用于标定 guard_floor_deep 与 guard_floor_confirm_s
口径对齐：tune_right_thumb.py 的 10 s 窗口恒载指标；exp6_dip_events.py 的下探检测框架（单流版）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右手掌-2026-10-15_15-22-01")
OUT = Path(__file__).resolve().parent / "palm_diag_20261015"
OUT.mkdir(parents=True, exist_ok=True)

DROP_ADC = 60.0
MERGE_S = 0.5
RECOVER_S = 2.0


def load_stream(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def med_in(t, sig, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def main() -> None:
    meta = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0]["observed_fps"])
    frame = load_stream(FOLDER, "device_001_seg000.csv")
    x = np.asarray(frame.iloc[:, 3:], dtype=float)
    n = x.shape[0]
    t = np.arange(n, dtype=float) / fps
    total = x.sum(1)

    # ── 起步段（0~20 s，0.5 s 网格）──
    head = []
    a = 0.0
    while a < 20.0:
        head.append(dict(t=round(a, 2), med=round(float(np.median(total[(t >= a) & (t < a + 0.5)])), 1)))
        a += 0.5

    # ── 10 s 窗口恒载指标（口径同 tune_right_thumb.hold_metric，参考窗取首 10 s）──
    ref_start = 10.0
    ref = med_in(t, total, ref_start, ref_start + 10.0)
    wins = []
    a = ref_start
    while a + 10.0 <= t[-1]:
        wins.append((a + 5.0, med_in(t, total, a, a + 10.0)))
        a += 10.0
    offsets = [abs(v - ref) for _, v in wins]
    tail_ref = med_in(t, total, t[-1] - 10.0, t[-1])
    late = np.array([(q, v) for q, v in wins if q > t[-1] - 120.0])
    slope_tail = float(np.polyfit(late[:, 0], late[:, 1], 1)[0]) if late.size else None

    # ── 漂移率：20 s 窗口线性拟合 ──
    rates = []
    a = 0.0
    while a + 20.0 <= t[-1]:
        mask = (t >= a) & (t < a + 20.0)
        if mask.sum() > 10:
            coef = np.polyfit(t[mask], total[mask], 1)
            rates.append((round(a + 10.0, 1), round(float(coef[0]), 3)))
        a += 10.0

    def rate_stat(lo, hi):
        vals = [v for q, v in rates if lo <= q < hi]
        if not vals:
            return None
        return dict(median=round(float(np.median(vals)), 3), min=round(float(np.min(vals)), 3),
                    max=round(float(np.max(vals)), 3), n=len(vals))

    # ── 噪声：0.5 s 中值滤波后的高频残差 ──
    def noise_sigma(sig, win_s=1.0, grid_s=0.25):
        resid = []
        a = 0.0
        while a + win_s <= t[-1]:
            mask = (t >= a) & (t < a + win_s)
            if mask.any():
                seg = sig[mask]
                resid.append(np.std(seg - np.median(seg)))
            a += grid_s
        return float(np.median(resid)) if resid else None

    chan_noise = np.array([noise_sigma(x[:, k]) for k in range(x.shape[1])], dtype=float)
    total_noise = noise_sigma(total)

    # ── 通道级下探事件（单流版）──
    diff = np.diff(total)
    cand = np.flatnonzero(diff < -DROP_ADC)
    events = []
    for idx in cand:
        if events and t[idx] - events[-1]["t_drop"] < MERGE_S:
            ev = events[-1]
            if total[idx + 1] < ev["trough"]:
                ev["trough"] = float(total[idx + 1])
                ev["t_trough"] = float(t[idx + 1])
                ev["t_drop"] = float(t[idx])
            continue
        events.append(dict(t_drop=float(t[idx]), start_level=float(total[idx]),
                           trough=float(total[idx + 1]), t_trough=float(t[idx + 1])))
    rows = []
    for ev in events:
        back = med_in(t, total, ev["t_trough"] + RECOVER_S, ev["t_trough"] + RECOVER_S + 2.0)
        before = med_in(t, total, max(0.0, ev["t_drop"] - 10.0), ev["t_drop"])
        if back is None or before is None:
            continue
        depth = before - ev["trough"]
        recovered_frac = (back - ev["trough"]) / max(depth, 1e-9)
        lo_i, hi_i = int(ev["t_drop"] * fps), int(min(t[-1], ev["t_trough"] + 1.0) * fps)
        seg = x[lo_i:hi_i] if hi_i > lo_i else x[lo_i:lo_i + 1]
        chan_drop = (np.median(x[max(0, lo_i - int(5 * fps)):lo_i], axis=0) - seg.min(axis=0)) if seg.size else np.zeros(x.shape[1])
        chan_drop = np.clip(chan_drop, 0.0, None)
        rows.append(dict(t_drop=round(ev["t_drop"], 2), depth_total_adc=round(float(depth), 1),
                         recovered_fraction=round(float(recovered_frac), 3),
                         duration_s=round(float(ev["t_trough"] - ev["t_drop"]), 3),
                         transient=bool(recovered_frac > 0.5),
                         chan_depth_max=round(float(chan_drop.max()), 1),
                         chan_depth_p95=round(float(np.percentile(chan_drop, 95)), 1),
                         chan_over_deep_100=round(float((chan_drop > 100.0).mean()), 4),
                         chan_over_deep_200=round(float((chan_drop > 200.0).mean()), 4)))
    transient_rows = [r for r in rows if r["transient"]]

    info = dict(
        fps=round(fps, 4), frames=int(n), channels=int(x.shape[1]), duration_s=round(float(t[-1]), 2),
        analysis_start_s=ref_start, ref_total_adc=round(ref, 1), end_total_adc=round(tail_ref, 1),
        endpoint_drift_adc=round(tail_ref - ref, 1),
        max_abs_offset_10s_adc=round(float(max(offsets)), 1),
        tail_slope_adc_s=round(slope_tail, 4) if slope_tail is not None else None,
        creep_rate_adc_s=dict(early_0_60=rate_stat(0, 60), mid_60_300=rate_stat(60, 300),
                              late_300_670=rate_stat(300, 700), all=rate_stat(0, 700)),
        noise_total_sigma_adc=round(float(total_noise), 2) if total_noise else None,
        noise_per_channel_median=round(float(np.median(chan_noise)), 3),
        noise_per_channel_p95=round(float(np.percentile(chan_noise, 95)), 3),
        dip_events=dict(threshold_adc=DROP_ADC, count=len(rows), transient_count=len(transient_rows),
                        real_unload_count=len(rows) - len(transient_rows)),
    )

    payload = dict(info=info, head_0_20s=head, rate_series=rates, dip_events=rows,
                   window_offsets=[{"t_mid": round(q, 1), "med": round(v, 1)} for q, v in wins])
    (OUT / "diag_step2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(info, ensure_ascii=False, indent=1))
    print("--- 0~20s 0.5s 网格总值 ---")
    print("   " + "  ".join(f"{h['t']:.1f}:{h['med']:.0f}" for h in head))
    print("--- 漂移率(每10s开出20s窗, ADC/s) ---")
    line = []
    for q, v in rates:
        line.append(f"{q:.0f}s:{v:+.2f}")
        if len(line) == 10:
            print("   " + " ".join(line))
            line = []
    if line:
        print("   " + " ".join(line))
    print(f"--- 下探事件 {len(rows)} 个 / 瞬态 {len(transient_rows)} 个 ---")
    for r in rows[:25]:
        print("   ", r)
    print(f"输出: {OUT / 'diag_step2.json'}")


if __name__ == "__main__":
    main()
