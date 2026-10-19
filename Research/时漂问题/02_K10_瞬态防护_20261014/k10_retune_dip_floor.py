# -*- coding: utf-8 -*-
"""K10 重扫终轮：把「过峰段下探」压到最低，同时看代价（尾端读数还涨多少）。

口径：
    dip_med = 输出 0.5 s 中值序列在 [onset+1, onset+8] 的最高点 − 在 [onset+8, onset+60] 的最低点
              （中值消噪，贴近肉眼观感）
    rise    = 末段 10 s 中值 − 10~20 s 中值（读数还涨多少）
    landing = 末段 10 s 中值 − 弹性电平 E
两条路线比：① 降 cap（慢态速率上限）② 推迟慢态开门（conf / soft）
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

OUT = HERE / "palm_tune_20261015"
GUARD_ON = dict(GUARD_DEFAULTS, enabled=True)
FOLDER = Path(r"D:\$RECYCLE.BIN\S-1-5-21-3129820652-4156655561-3743858824-1001\$RKJ92E8")

LIVE = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
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


def median_curve(t, sig, onset, span=60.0, step=0.5):
    grid = np.arange(onset, onset + span, step)
    return np.array([med_in(t, sig, a, a + step) if med_in(t, sig, a, a + step) is not None else np.nan
                     for a in grid])


def analyze(t, total, onset, e_level):
    curve = median_curve(t, total, onset, span=60.0)
    peak_i = int(np.nanargmax(curve[: int(8.0 / 0.5)]))          # 0~8 s 段内峰
    after = curve[peak_i:]
    dip_med = float(curve[peak_i] - np.nanmin(after))
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    early = med_in(t, total, max(onset + 8.0, 10.0), max(onset + 18.0, 20.0))
    return dict(dip_med=round(dip_med, 1), t_peak_s=round(onset + peak_i * 0.5, 2),
                peak_med=round(float(curve[peak_i]), 1),
                rise_adc=round(end - early, 1), landing_adc=round(end - e_level, 1))


def build_candidates() -> list[dict]:
    rows = []
    for cap in (0.0, 0.002, 0.003, 0.005, 0.007):
        for rsm in (0.10, 0.15, 0.20):
            rows.append(dict(tag=f"A_cap{cap:.3f}_rsm{rsm:.2f}_rf02",
                             params=dict(LIVE, slope_cap_frac=cap, r_slow_max=rsm, r_fast=0.02)))
    for conf in (1.5, 3.0, 5.0):
        for soft in (1.5, 8.0):
            rows.append(dict(tag=f"B_cap009_conf{conf:.1f}_soft{soft:.1f}",
                             params=dict(LIVE, slow_confirm_s=conf, soft_unfreeze_s=soft)))
    for cap in (0.012, 0.02):
        rows.append(dict(tag=f"C_cap{cap:.3f}_conf5_soft8",
                         params=dict(LIVE, slope_cap_frac=cap, slow_confirm_s=5.0, soft_unfreeze_s=8.0)))
    rows.append(dict(tag="D_LIVE", params=dict(LIVE)))
    rows.append(dict(tag="D_OLD_RECO_cap035_rsm35_rf03",
                     params=dict(LIVE, slope_cap_frac=0.035, r_slow_max=0.35, r_fast=0.03)))
    rows.append(dict(tag="D_worksheet_cap05_conf5_soft8",
                     params=dict(slow_confirm_s=5.0, soft_unfreeze_s=8.0, slope_cap_frac=0.05,
                                 r_slow_max=0.2)))
    return rows


def main() -> None:
    meta = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0]["observed_fps"])
    frame = load_stream(FOLDER, "device_001_seg000.csv")
    x = np.asarray(frame.iloc[:, 3:], dtype=float)
    t = np.arange(x.shape[0], dtype=float) / fps
    total = x.sum(1)
    onset = float(t[int(np.argmax(np.diff(total) > 500.0))])
    e_level = float(np.min(total[(t >= onset + 0.35) & (t <= onset + 1.0)]))
    in_curve = median_curve(t, total, onset, span=6.0, step=0.5)
    print(f"onset {onset:.2f}s  E {e_level:.0f} | 输入 0~6 s 中值: "
          f"{' '.join(f'{v:.0f}' for v in in_curve)}", flush=True)
    print(f"{'候选':34s} {'dip_med':>8s} {'t_peak':>7s} {'peak':>8s} {'rise':>8s} {'landing':>9s}", flush=True)

    rows = []
    for candidate in build_candidates():
        started = time.perf_counter()
        out, _ = compensate(x.copy(), t, candidate["params"], GUARD_ON)
        m = analyze(t, out.sum(1), onset, e_level)
        rows.append(dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False),
                         **m, seconds=round(time.perf_counter() - started, 2)))
        print(f"{candidate['tag']:34s} {m['dip_med']:8.1f} {m['t_peak_s']:7.2f} {m['peak_med']:8.1f} "
              f"{m['rise_adc']:8.1f} {m['landing_adc']:9.1f}", flush=True)

    (OUT / "retune_dip_floor.json").write_text(json.dumps(dict(onset_s=onset, E_adc=e_level, rows=rows),
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT / 'retune_dip_floor.json'}")


if __name__ == "__main__":
    main()
