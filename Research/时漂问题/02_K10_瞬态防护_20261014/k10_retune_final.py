# -*- coding: utf-8 -*-
"""K10 重扫收官：以 soft_unfreeze_s / slow_confirm_s 为主旋钮（错开过峰段），精定推荐值。"""
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
CACHE = OUT / "palm_152201_cache.npz"

LIVE = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
            slope_cap_frac=0.009, r_slow_max=0.2, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)


def load_main():
    if CACHE.exists():
        data = np.load(CACHE)
        return data["t"], data["x"], float(data["onset"]), float(data["E"])
    path = FOLDER / "device_001_seg000.csv"
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(path)
    frame = pd.read_csv(path, skiprows=skip, header=0, engine="c")
    x = np.asarray(frame.iloc[:, 3:], dtype=float)
    meta = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0]["observed_fps"])
    t = np.arange(x.shape[0], dtype=float) / fps
    total = x.sum(1)
    onset = float(t[int(np.argmax(np.diff(total) > 500.0))])
    e_level = float(np.min(total[(t >= onset + 0.35) & (t <= onset + 1.0)]))
    np.savez_compressed(CACHE, t=t, x=x, onset=onset, E=e_level)
    return t, x, onset, e_level


def med_in(t, sig, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def analyze(t, total, onset, e_level):
    grid = np.arange(onset, onset + 60.0, 0.5)
    curve = np.array([med_in(t, total, a, a + 0.5) for a in grid], dtype=float)
    head = curve[:16]                      # 0~8 s
    peak_i = int(np.nanargmax(head))
    dip = float(curve[peak_i] - np.nanmin(curve[peak_i:]))
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    early = med_in(t, total, max(onset + 8.0, 10.0), max(onset + 18.0, 20.0))
    return dict(dip_med=round(dip, 1), peak_med=round(float(curve[peak_i]), 1),
                rise_adc=round(end - early, 1), landing_adc=round(end - e_level, 1),
                dip_end=round(float(curve[peak_i]) - end, 1))


def build_candidates() -> list[dict]:
    rows = []
    for cap in (0.009, 0.012, 0.02):
        for soft in (8.0, 15.0, 30.0):
            for conf in (5.0, 8.0):
                rows.append(dict(tag=f"cap{cap:.3f}_conf{conf:.0f}_soft{soft:.0f}_rf02",
                                 params=dict(LIVE, slope_cap_frac=cap, slow_confirm_s=conf,
                                             soft_unfreeze_s=soft, r_fast=0.02)))
    for cap in (0.012, 0.02):
        for rf in (0.03, 0.04):
            rows.append(dict(tag=f"cap{cap:.3f}_conf5_soft8_rf{rf:.2f}",
                             params=dict(LIVE, slope_cap_frac=cap, slow_confirm_s=5.0,
                                         soft_unfreeze_s=8.0, r_fast=rf)))
    rows.append(dict(tag="live_cap009_conf1.5_soft1.5", params=dict(LIVE)))
    return rows


def main() -> None:
    t, x, onset, e_level = load_main()
    total = x.sum(1)
    print(f"onset {onset:.2f}s  E {e_level:.0f}  frames {x.shape[0]}", flush=True)
    print(f"{'候选':32s} {'dip_med':>8s} {'peak':>8s} {'rise':>8s} {'landing':>9s}", flush=True)
    rows = []
    for candidate in build_candidates():
        started = time.perf_counter()
        out, _ = compensate(x.copy(), t, candidate["params"], GUARD_ON)
        m = analyze(t, out.sum(1), onset, e_level)
        rows.append(dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False),
                         **m, seconds=round(time.perf_counter() - started, 2)))
        print(f"{candidate['tag']:32s} {m['dip_med']:8.1f} {m['peak_med']:8.1f} {m['rise_adc']:8.1f} "
              f"{m['landing_adc']:9.1f}", flush=True)

    rows.sort(key=lambda r: (r["dip_med"], r["rise_adc"]))
    (OUT / "retune_final_dip_first.json").write_text(
        json.dumps(dict(onset_s=onset, E_adc=e_level, rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n按 dip 升序（前 6）:")
    for row in rows[:6]:
        print(f"   {row['tag']:32s} dip {row['dip_med']:7.1f}  rise {row['rise_adc']:7.1f}  landing {row['landing_adc']:8.1f}")
    print(f"输出: {OUT / 'retune_final_dip_first.json'}")


if __name__ == "__main__":
    main()
