# -*- coding: utf-8 -*-
"""K10 定版轮：conf/soft 错开过峰段（dip 触底）后，用 rf 回收落点，双数据交叉验证。

推荐候选统一：slow_confirm_s = 8.0、soft_unfreeze_s = 8.0、r_slow_max = 0.2；
在 cap ∈ {0.009, 0.012, 0.02} 与 rf ∈ {0.03, 0.04, 0.05} 上取点。
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
PRESETS = {
    "palm_152201": Path(r"D:\$RECYCLE.BIN\S-1-5-21-3129820652-4156655561-3743858824-1001\$RKJ92E8"),
    "b2d896": Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\archived\手掌数据"
                   r"\20260927_161747_single_device_b2d896"),
}
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
    grid = np.arange(onset, onset + 60.0, 0.5)
    curve = np.array([med_in(t, total, a, a + 0.5) for a in grid], dtype=float)
    peak_i = int(np.nanargmax(curve[:16]))
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    early = med_in(t, total, max(onset + 8.0, 10.0), max(onset + 18.0, 20.0))
    return dict(dip_med=round(float(curve[peak_i] - np.nanmin(curve[peak_i:])), 1),
                rise_adc=round(end - early, 1), landing_adc=round(end - e_level, 1),
                end_adc=round(end, 1))


def build_candidates() -> list[dict]:
    rows = [dict(tag="REF_live_cap009_conf1.5_soft1.5", params=dict(LIVE)),
            dict(tag="REF_old_reco_cap035_rsm35_rf03",
                 params=dict(LIVE, slope_cap_frac=0.035, r_slow_max=0.35, r_fast=0.03)),
            dict(tag="REF_161747_cap005_rsm15_rf06_conf2_soft1.5",
                 params=dict(r_fast=0.06, tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=1.5,
                             slope_cap_frac=0.005, r_slow_max=0.15, tau_r_fast_s=6.0,
                             tau_r_slow_idle_s=0.5))]
    for cap in (0.009, 0.012, 0.02):
        for rf in (0.03, 0.04, 0.05):
            rows.append(dict(tag=f"NEW_cap{cap:.3f}_conf8_soft8_rf{rf:.2f}",
                             params=dict(LIVE, slope_cap_frac=cap, slow_confirm_s=8.0,
                                         soft_unfreeze_s=8.0, r_fast=rf)))
    return rows


def main() -> None:
    report = {}
    for tag, folder in PRESETS.items():
        meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        fps = float(meta["devices"][0]["observed_fps"])
        frame = load_stream(folder, "device_001_seg000.csv")
        x = np.asarray(frame.iloc[:, 3:], dtype=float)
        t = np.arange(x.shape[0], dtype=float) / fps
        total = x.sum(1)
        onset = float(t[int(np.argmax(np.diff(total) > 500.0))])
        e_level = float(np.min(total[(t >= onset + 0.35) & (t <= onset + 1.0)]))
        print(f"\n=== {tag}  onset {onset:.2f}s  E {e_level:.0f}  {x.shape[0]} 帧 × {x.shape[1]} 通道 ===", flush=True)
        print(f"{'候选':40s} {'dip_med':>8s} {'rise':>8s} {'landing':>9s}", flush=True)
        rows = []
        for candidate in build_candidates():
            out, _ = compensate(x.copy(), t, candidate["params"], GUARD_ON)
            m = analyze(t, out.sum(1), onset, e_level)
            rows.append(dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False), **m))
            print(f"{candidate['tag']:40s} {m['dip_med']:8.1f} {m['rise_adc']:8.1f} {m['landing_adc']:9.1f}",
                  flush=True)
        report[tag] = dict(onset_s=onset, E_adc=e_level, rows=rows)

    (OUT / "retune_define.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {OUT / 'retune_define.json'}")


if __name__ == "__main__":
    main()
