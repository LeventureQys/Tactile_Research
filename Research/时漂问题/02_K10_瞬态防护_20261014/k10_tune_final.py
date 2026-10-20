# -*- coding: utf-8 -*-
"""K10 调参 · 最终折中轮：在两份手掌长保压上同时评同一批候选。

数据：
  palm_152201 = 右手掌-2026-10-15_15-22-01（670 s，强蠕变：输入 rise 1526 ADC，尾斜率 +0.78 ADC/s）
  b2d896      = 20260927_161747（429 s，弱蠕变：输入 rise 580 ADC，尾斜率 +0.30 ADC/s）

目标：找一套在「强蠕变跟得上」与「弱蠕变不过扣」之间平衡的现场 8 参。
主判据 rise（加载 10~20 s 平台 → 末段，正=还涨 / 负=下坠），辅以 landing（落点）与 offset30（窗口最大偏移）。
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

GUARD_ON = dict(GUARD_DEFAULTS, enabled=True)
OUT = HERE / "palm_tune_20261015"
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "palm_152201": Path(r"D:\$RECYCLE.BIN\S-1-5-21-3129820652-4156655561-3743858824-1001\$RKJ92E8"),
    "b2d896": Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\archived\手掌数据"
                   r"\20260927_161747_single_device_b2d896"),
}

BASE = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
            slope_cap_frac=0.009, r_slow_max=0.2, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)


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


def metrics(t, total, onset, e_level):
    lo = max(onset + 8.0, 10.0)
    early = med_in(t, total, lo, lo + 10.0)
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    late = [(g, med_in(t, total, g, g + 10.0)) for g in np.arange(t[-1] - 120.0, t[-1] - 10.0, 10.0)]
    late = [(q, v) for q, v in late if v is not None]
    slope = float(np.polyfit([q for q, _ in late], [v for _, v in late], 1)[0]) if len(late) > 1 else None
    ref = med_in(t, total, 30.0, 40.0)
    wins = [med_in(t, total, a, a + 10.0) for a in np.arange(30.0, t[-1] - 10.0, 10.0)]
    wins = [v for v in wins if v is not None]
    return dict(rise_adc=round(end - early, 1), landing_adc=round(end - e_level, 1),
                tail_slope_adc_s=round(slope, 4) if slope is not None else None,
                offset30_adc=round(float(np.max(np.abs(np.asarray(wins) - ref))), 1) if wins else None,
                end_adc=round(end, 1))


def build_candidates() -> list[dict]:
    rows = [dict(tag="F00_live_cap009_base", params=dict(BASE))]
    for cap, rsm in ((0.02, 0.35), (0.035, 0.35)):
        for rf in (0.02, 0.03, 0.04, 0.05):
            rows.append(dict(tag=f"F_cap{cap:.3f}_rsm{rsm:.2f}_rf{rf:.2f}",
                             params=dict(BASE, slope_cap_frac=cap, r_slow_max=rsm, r_fast=rf)))
    rows.append(dict(tag="F_161747_struct_cap035_rsm35",
                     params=dict(r_fast=0.06, tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=1.5,
                                 slope_cap_frac=0.035, r_slow_max=0.35, tau_r_fast_s=6.0,
                                 tau_r_slow_idle_s=0.5)))
    return rows


def main() -> None:
    datasets = []
    for tag, folder in PRESETS.items():
        meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        fps = float(meta["devices"][0]["observed_fps"])
        frame = load_stream(folder, "device_001_seg000.csv")
        x = np.asarray(frame.iloc[:, 3:], dtype=float)
        t = np.arange(x.shape[0], dtype=float) / fps
        total = x.sum(1)
        onset = float(t[int(np.argmax(np.diff(total) > 500.0))])
        e_level = float(np.min(total[(t >= onset + 0.35) & (t <= onset + 1.0)]))
        base = metrics(t, total, onset, e_level)
        datasets.append(dict(tag=tag, folder=folder, t=t, x=x, onset=onset, e_level=e_level,
                             input_metrics=base))
        print(f"[{tag}] {x.shape[0]} 帧 × {x.shape[1]} 通道 {t[-1]:.0f} s | E={e_level:.0f} | "
              f"输入 rise {base['rise_adc']:.0f} landing {base['landing_adc']:.0f} "
              f"tail {base['tail_slope_adc_s']:+.3f} offset30 {base['offset30_adc']:.0f}", flush=True)

    candidates = build_candidates()
    results = {}
    for dataset in datasets:
        rows = [dict(tag="INPUT", **dataset["input_metrics"], ratio=1.0)]
        print(f"--- {dataset['tag']} ---", flush=True)
        print(f"{'tag':30s} {'rise':>8s} {'landing':>9s} {'tail':>8s} {'offset30':>9s} {'ratio':>7s}", flush=True)
        for candidate in candidates:
            started = time.perf_counter()
            out, _ = compensate(dataset["x"].copy(), dataset["t"], candidate["params"], GUARD_ON)
            m = metrics(dataset["t"], out.sum(1), dataset["onset"], dataset["e_level"])
            row = dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False),
                       **m, ratio=round(m["end_adc"] / dataset["input_metrics"]["end_adc"], 4),
                       seconds=round(time.perf_counter() - started, 2))
            rows.append(row)
            print(f"{row['tag']:30s} {row['rise_adc']:8.1f} {row['landing_adc']:9.1f} "
                  f"{row['tail_slope_adc_s']:+8.3f} {row['offset30_adc']:9.1f} {row['ratio']:7.4f}", flush=True)
        results[dataset["tag"]] = rows

    (OUT / "tune_final_two_dataset.json").write_text(
        json.dumps(dict(presets={k: str(v) for k, v in PRESETS.items()}, results=results),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT / 'tune_final_two_dataset.json'}")


if __name__ == "__main__":
    main()
