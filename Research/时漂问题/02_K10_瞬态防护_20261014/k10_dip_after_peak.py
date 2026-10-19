# -*- coding: utf-8 -*-
"""过峰下探专项：加载快相峰值之后输出下沉了多少（前几轮指标漏测的区间）。

口径（对齐 161747 报告「下坠 = 段内峰值 − 段末」）：
    peak      = 输出在 [onset+0.5 s, 末] 的最大值
    trough    = 该峰值之后的最小值
    dip       = peak − trough          ← 过峰下探量
    dip_end   = peak − 末段 10 s 中值
    t_peak    = 峰值时刻；t_trough = 下探最低点时刻
"""
from __future__ import annotations

import json
import sys
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
MINE = dict(LIVE, r_fast=0.03, slope_cap_frac=0.035, r_slow_max=0.35)
WORKSHEET = {}


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


def analyze(t: np.ndarray, total: np.ndarray, onset: float) -> dict:
    lo = onset + 0.5
    seg = t >= lo
    ts, ys = t[seg], total[seg]
    peak_i = int(np.argmax(ys))
    peak = float(ys[peak_i])
    after = ys[peak_i:]
    trough = float(after.min())
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    early = med_in(t, total, max(onset + 8.0, 10.0), max(onset + 18.0, 20.0))
    ref = med_in(t, total, 30.0, 40.0)
    wins = [v for v in (med_in(t, total, a, a + 10.0) for a in np.arange(30.0, t[-1] - 10.0, 10.0)) if v is not None]
    return dict(peak_adc=round(peak, 1), t_peak_s=round(float(ts[peak_i]), 2),
                trough_adc=round(trough, 1), t_trough_s=round(float(ts[peak_i + int(np.argmin(after))]), 2),
                dip_adc=round(peak - trough, 1), dip_end_adc=round(peak - end, 1),
                rise_adc=round(end - early, 1),
                offset30_adc=round(float(np.max(np.abs(np.asarray(wins) - ref))), 1))


def build_candidates() -> list[dict]:
    rows = [
        dict(tag="设备现值 (cap.009 rsm.2 rf.02)", params=LIVE),
        dict(tag="**上一轮推荐 (cap.035 rsm.35 rf.03)**", params=MINE),
        dict(tag="工作树默认 (cap.05 rsm.2 rf.01)", params=WORKSHEET),
    ]
    for cap in (0.005, 0.009, 0.015, 0.02, 0.035):
        for rf in (0.01, 0.02, 0.03):
            rows.append(dict(tag=f"cap{cap:.3f}_rsm.35_rf{rf:.2f}",
                             params=dict(LIVE, slope_cap_frac=cap, r_slow_max=0.35, r_fast=rf)))
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
    print(f"onset {onset:.2f}s  E {e_level:.0f}  |  输入 {json.dumps(analyze(t, total, onset), ensure_ascii=False)}",
          flush=True)

    print(f"\n{'候选':40s} {'peak':>8s} {'t_peak':>7s} {'trough':>9s} {'dip':>7s} {'dip→末':>8s} {'rise':>8s} {'off30':>7s}", flush=True)
    rows = []
    curves = {}
    for candidate in build_candidates():
        out, _ = compensate(x.copy(), t, candidate["params"], GUARD_ON)
        total_o = out.sum(1)
        m = analyze(t, total_o, onset)
        rows.append(dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False), **m))
        print(f"{candidate['tag']:40s} {m['peak_adc']:8.1f} {m['t_peak_s']:7.2f} {m['trough_adc']:9.1f} "
              f"{m['dip_adc']:7.1f} {m['dip_end_adc']:8.1f} {m['rise_adc']:8.1f} {m['offset30_adc']:7.1f}", flush=True)
        if candidate["tag"] in ("设备现值 (cap.009 rsm.2 rf.02)", "**上一轮推荐 (cap.035 rsm.35 rf.03)**"):
            curves[candidate["tag"]] = [round(float(med_in(t, total_o, a, a + 0.5)), 1)
                                        for a in np.arange(onset, onset + 40.0, 0.5)]

    print("\n加载后 0~40 s 输出（0.5 s 网格，第一行=设备现值，第二行=上一轮推荐）:")
    for tag, values in curves.items():
        print(f"  {tag}")
        for i in range(0, len(values), 10):
            print("     " + " ".join(f"{v:8.1f}" for v in values[i:i + 10]))
    print("  输入同段:")
    ins = [round(float(med_in(t, total, a, a + 0.5)), 1) for a in np.arange(onset, onset + 40.0, 0.5)]
    for i in range(0, len(ins), 10):
        print("     " + " ".join(f"{v:8.1f}" for v in ins[i:i + 10]))

    (OUT / "dip_after_peak.json").write_text(json.dumps(dict(onset_s=onset, E_adc=e_level, rows=rows,
                                                            curves_0_40s=curves),
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {OUT / 'dip_after_peak.json'}")


if __name__ == "__main__":
    main()
