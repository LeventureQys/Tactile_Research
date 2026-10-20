# -*- coding: utf-8 -*-
"""K10 调参 · 第三轮：用户体感口径（加载 10 s 后读数还涨多少）+ 落点。

前两轮口径（t=30 s 起的窗口最大偏移）对 cap 大的档不公平：30 s 时刻各档已补的量不同。
本轮改用与既有手掌报告一致的口径：
  E        = 加载沿结束后 [t0+0.35, t0+1.0] 的输入最小值（弹性电平，口径同 161747 报告）
  rise     = 输出(末 10 s 中值) − 输出(10~20 s 中值)   ← 主指标：读数还在涨多少
  tail     = 输出末 120 s 拟合斜率                      ← 是否已冻结
  landing  = 输出(末 10 s 中值) − E                     ← 准度（落点）
  ratio    = 输出末 / 输入末
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
GUARD_ON = dict(GUARD_DEFAULTS, enabled=True)


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


def med_in(t, sig, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def evaluate(t: np.ndarray, total: np.ndarray, onset: float, e_level: float) -> dict:
    early = med_in(t, total, max(onset + 8.0, 10.0), max(onset + 18.0, 20.0))
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    late_mask = t > t[-1] - 120.0
    grid = np.arange(t[-1] - 120.0, t[-1] - 10.0, 10.0)
    late = [(g, med_in(t, total, g, g + 10.0)) for g in grid]
    late = [(q, v) for q, v in late if v is not None]
    slope = float(np.polyfit([q for q, _ in late], [v for _, v in late], 1)[0]) if len(late) > 1 else None
    return dict(early_adc=round(early, 1), end_adc=round(end, 1),
                rise_adc=round(end - early, 1), tail_slope_adc_s=round(slope, 4) if slope else None,
                landing_adc=round(end - e_level, 1))


def main() -> None:
    t, x = load_main()
    input_total = x.sum(1)

    # 加载沿定位：总值从空载跳到受载
    diff = np.diff(input_total)
    onset = float(t[int(np.argmax(diff > 500.0))])
    e_level = float(np.min(input_total[(t >= onset + 0.35) & (t <= onset + 1.0)]))
    print(f"加载沿 t0 = {onset:.2f} s；弹性电平 E = {e_level:.1f} ADC", flush=True)
    print("0~3 s (0.1 s 网格):", flush=True)
    print("   " + " ".join(f"{v:.1f}:{med_in(t, input_total, v, v + 0.1):.0f}"
                           for v in np.arange(0.0, 3.0, 0.1)), flush=True)

    base = evaluate(t, input_total, onset, e_level)
    print(f"输入对照: 末段 {base['end_adc']:.0f} | rise {base['rise_adc']:.0f} | "
          f"尾斜率 {base['tail_slope_adc_s']:+.3f} | 落点 {base['landing_adc']:+.0f}", flush=True)
    print(f"{'tag':30s} {'rise':>8s} {'end':>9s} {'landing':>9s} {'slope':>8s} {'ratio':>7s}", flush=True)

    rows = [dict(tag="INPUT_reference", params="input", **base, compression_ratio=1.0)]
    for cap in (0.02, 0.03, 0.05):
        for rsm in (0.25, 0.35, 0.50):
            for rf in (0.02, 0.05, 0.08):
                tag = f"cap{cap:.2f}_rsm{rsm:.2f}_rf{rf:.2f}"
                params = dict(r_fast=rf, slope_cap_frac=cap, r_slow_max=rsm,
                              tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=1.5,
                              tau_r_fast_s=6.0, tau_r_slow_idle_s=0.5)
                started = time.perf_counter()
                out, _ = compensate(x.copy(), t, params, GUARD_ON)
                m = evaluate(t, out.sum(1), onset, e_level)
                row = dict(tag=tag, params=json.dumps(params, ensure_ascii=False), **m,
                           compression_ratio=round(m["end_adc"] / base["end_adc"], 4),
                           seconds=round(time.perf_counter() - started, 2))
                rows.append(row)
                print(f"{tag:30s} {row['rise_adc']:8.1f} {row['end_adc']:9.1f} {row['landing_adc']:9.1f} "
                      f"{row['tail_slope_adc_s']:+8.3f} {row['compression_ratio']:7.4f}", flush=True)

    (OUT / "sweep_round3.json").write_text(
        json.dumps(dict(source=str(FOLDER), onset_s=onset, E_adc=e_level, rows=rows),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT / 'sweep_round3.json'}")


if __name__ == "__main__":
    main()
