# -*- coding: utf-8 -*-
"""K10 调参 · 第二轮（定向精扫）：右手掌-2026-10-15_15-22-01。

第一轮已确认：
  - guard 层（floor/idle/spike）在本次数据上几乎零差异 ⇒ 无瞬态下探，防护不是本数据的杠杆
  - K9 本体是关键：工作树默认 227 ADC / palm9 档 223 ADC，输入漂移 1525 ADC
第二轮目的：把残余漂移继续压下去，并量化"准度代价"（压缩比）。

新增口径：offset 拆成带符号的正偏移（上漂）与负偏移（下坠），避免绝对值掩盖方向。
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
from guarded_k9 import GUARD_DEFAULTS, compensate  # noqa: E402

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右手掌-2026-10-15_15-22-01")
OUT = HERE / "palm_tune_20261015"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 100.389239040211
ANALYSIS_START = 30.0
BIN_S = 10.0
TAIL_S = 120.0

GUARD_ON = dict(GUARD_DEFAULTS, enabled=True)

# 161747（429 s 手掌长保压）两套处方
G161747_REC = dict(r_fast=0.06, tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=1.5,
                   slope_cap_frac=0.005, r_slow_max=0.15, tau_r_fast_s=6.0, tau_r_slow_idle_s=0.5)
G161747_MIN = dict(slope_cap_frac=0.005, r_slow_max=0.15)
PALM_R_PALM9 = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
                    slope_cap_frac=0.012, r_slow_max=0.25, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)


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
    return np.arange(x.shape[0], dtype=float) / FPS, x


def med_in(t: np.ndarray, sig: np.ndarray, a: float, b: float):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def metrics(t: np.ndarray, total: np.ndarray) -> dict:
    start = ANALYSIS_START
    ref = med_in(t, total, start, start + BIN_S)
    wins = []
    a = start
    while a + BIN_S <= t[-1]:
        value = med_in(t, total, a, a + BIN_S)
        if value is not None:
            wins.append((a + BIN_S / 2, value))
        a += BIN_S
    arr = np.asarray(wins, dtype=float)
    delta = arr[:, 1] - ref
    tail = med_in(t, total, t[-1] - BIN_S, t[-1])
    late = arr[arr[:, 0] > t[-1] - TAIL_S]
    slope = float(np.polyfit(late[:, 0], late[:, 1], 1)[0]) if late.shape[0] > 1 else None
    return dict(ref_adc=round(ref, 1), end_adc=round(tail, 1),
                endpoint_drift_adc=round(tail - ref, 1),
                up_offset_adc=round(float(np.max(delta)), 1),
                down_offset_adc=round(float(np.min(delta)), 1),
                max_abs_offset_adc=round(float(np.max(np.abs(delta))), 1),
                tail_slope_adc_s=round(slope, 4) if slope is not None else None,
                settle_s=next((round(q - start, 1) for q, _ in wins if abs(_ - ref) > 50.0), 0.0))


def hold_settle(t: np.ndarray, total: np.ndarray, tol: float = 50.0) -> float:
    """最后一次偏离参考值超过 tol 的时刻（相对分析起点）。"""
    ref = med_in(t, total, ANALYSIS_START, ANALYSIS_START + BIN_S)
    last = ANALYSIS_START
    a = ANALYSIS_START
    while a + BIN_S <= t[-1]:
        value = med_in(t, total, a, a + BIN_S)
        if value is not None and abs(value - ref) > tol:
            last = a + BIN_S
        a += BIN_S
    return round(last - ANALYSIS_START, 1)


def build_candidates() -> list[dict]:
    rows = [
        dict(tag="A1_161747_recommended", params=G161747_REC),
        dict(tag="A2_161747_minimal", params=G161747_MIN),
        dict(tag="A3_palm9_second", params=PALM_R_PALM9),
        dict(tag="A4_palm9_cap005", params=dict(PALM_R_PALM9, slope_cap_frac=0.005)),
        dict(tag="A5_palm9_rsm015", params=dict(PALM_R_PALM9, r_slow_max=0.15)),
    ]
    for cap in (0.02, 0.03, 0.04, 0.05, 0.07):
        for rsm in (0.15, 0.25, 0.35, 0.50):
            rows.append(dict(tag=f"C_cap{cap:.2f}_rsm{rsm:.2f}", params=dict(slope_cap_frac=cap, r_slow_max=rsm)))
    for rf in (0.02, 0.03, 0.05, 0.08):
        rows.append(dict(tag=f"D_rf{rf:.2f}_cap050_rsm020", params=dict(r_fast=rf, slope_cap_frac=0.05, r_slow_max=0.2)))
    return rows


def main() -> None:
    t, x = load_main()
    input_total = x.sum(1)
    base = metrics(t, input_total)
    base["settle_s"] = hold_settle(t, input_total)
    print(f"输入: {x.shape[0]} 帧 × {x.shape[1]} 通道, {t[-1]:.1f} s | "
          f"漂移 {base['endpoint_drift_adc']} ADC, 最大偏移 {base['max_abs_offset_adc']} ADC, "
          f"末斜率 {base['tail_slope_adc_s']} ADC/s", flush=True)

    rows = [dict(tag="INPUT_reference", **base, compression_ratio=1.0)]
    header = f"{'tag':26s} {'maxabs':>8s} {'up':>8s} {'down':>8s} {'drift':>8s} {'slope':>8s} {'ratio':>7s} {'settle':>7s}"
    print(header, flush=True)
    for candidate in build_candidates():
        started = time.perf_counter()
        out, _info = compensate(x.copy(), t, candidate["params"], GUARD_ON)
        elapsed = time.perf_counter() - started
        out_total = out.sum(1)
        m = metrics(t, out_total)
        m["settle_s"] = hold_settle(t, out_total)
        row = dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False),
                   **m, compression_ratio=round(m["end_adc"] / base["end_adc"], 4),
                   seconds=round(elapsed, 2))
        rows.append(row)
        print(f"{row['tag']:26s} {row['max_abs_offset_adc']:8.1f} {row['up_offset_adc']:8.1f} "
              f"{row['down_offset_adc']:8.1f} {row['endpoint_drift_adc']:8.1f} "
              f"{row['tail_slope_adc_s']:+8.3f} {row['compression_ratio']:7.4f} {row['settle_s']:7.1f}",
              flush=True)

    (OUT / "sweep_round2.json").write_text(
        json.dumps(dict(source=str(FOLDER), analysis_start_s=ANALYSIS_START, rows=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"输出: {OUT / 'sweep_round2.json'}")


if __name__ == "__main__":
    main()
