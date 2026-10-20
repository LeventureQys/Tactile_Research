# -*- coding: utf-8 -*-
"""K10 重扫：以「过峰下探」为硬约束重选现场 8 参。

新增判据（上一轮遗漏的区间）：
    dip_early  = max(输出[onset+1, onset+8]) − min(输出[onset+8, onset+30])   ← 过峰段下探
    dip_all    = 全段峰值 − 其后的最小值
    dip_to_end = 全段峰值 − 末段 10 s 中值
约束顺序：dip_early / dip_all 先看，再看 rise（读数还涨多少）与 landing（落点）。
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


def analyze(t, total, onset, e_level):
    lo = onset + 0.5
    seg = t >= lo
    ts, ys = t[seg], total[seg]
    peak_i = int(np.argmax(ys))
    peak = float(ys[peak_i])
    after = ys[peak_i:]
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    early = med_in(t, total, max(onset + 8.0, 10.0), max(onset + 18.0, 20.0))
    # 过峰段：加载后 1~8 s 的最高点 → 8~30 s 的最低点
    win_a = med_in(t, total, onset + 1.0, onset + 8.0)
    hi_a = float(np.max(total[(t >= onset + 1.0) & (t <= onset + 8.0)]))
    lo_b = float(np.min(total[(t >= onset + 8.0) & (t <= onset + 30.0)]))
    return dict(peak_adc=round(peak, 1), t_peak_s=round(float(ts[peak_i]), 2),
                dip_all=round(peak - float(after.min()), 1),
                dip_to_end=round(peak - end, 1),
                dip_early=round(hi_a - lo_b, 1), dip_early_med=round(win_a - lo_b, 1),
                rise_adc=round(end - early, 1), landing_adc=round(end - e_level, 1),
                off30=round(float(np.max(np.abs(
                    np.asarray([v for v in (med_in(t, total, a, a + 10.0)
                                            for a in np.arange(30.0, t[-1] - 10.0, 10.0)) if v is not None])
                    - med_in(t, total, 30.0, 40.0)))), 1))


def build_candidates() -> list[dict]:
    rows = [dict(tag="LIVE_cap009_rsm20_rf02", params=dict(LIVE)),
            dict(tag="LIVE+rf03", params=dict(LIVE, r_fast=0.03)),
            dict(tag="LIVE+rf04", params=dict(LIVE, r_fast=0.04)),
            dict(tag="LIVE+rsm25", params=dict(LIVE, r_slow_max=0.25)),
            dict(tag="OLD_RECO_cap035_rsm35_rf03", params=dict(LIVE, slope_cap_frac=0.035,
                                                              r_slow_max=0.35, r_fast=0.03))]
    for cap in (0.005, 0.007, 0.012, 0.015):
        for rsm in (0.15, 0.20, 0.25):
            for rf in (0.02, 0.03):
                rows.append(dict(tag=f"cap{cap:.3f}_rsm{rsm:.2f}_rf{rf:.2f}",
                                 params=dict(LIVE, slope_cap_frac=cap, r_slow_max=rsm, r_fast=rf)))
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
    print(f"onset {onset:.2f}s  E {e_level:.0f}", flush=True)
    print(f"{'候选':32s} {'dipEarly':>9s} {'dipAll':>8s} {'dipEnd':>8s} {'rise':>8s} {'landing':>9s} {'off30':>7s}",
          flush=True)

    rows = []
    for candidate in build_candidates():
        started = time.perf_counter()
        out, _ = compensate(x.copy(), t, candidate["params"], GUARD_ON)
        m = analyze(t, out.sum(1), onset, e_level)
        rows.append(dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False),
                         **m, seconds=round(time.perf_counter() - started, 2)))
        print(f"{candidate['tag']:32s} {m['dip_early']:9.1f} {m['dip_all']:8.1f} {m['dip_to_end']:8.1f} "
              f"{m['rise_adc']:8.1f} {m['landing_adc']:9.1f} {m['off30']:7.1f}", flush=True)

    (OUT / "retune_no_dip.json").write_text(json.dumps(dict(onset_s=onset, E_adc=e_level, rows=rows),
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT / 'retune_no_dip.json'}")


if __name__ == "__main__":
    main()
