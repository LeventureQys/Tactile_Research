# -*- coding: utf-8 -*-
"""推荐档闭环验证：guard 开/关差异 + 与设备现值逐项对照（两份手掌长保压）。"""
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
PRESETS = {
    "palm_152201": Path(r"D:\$RECYCLE.BIN\S-1-5-21-3129820652-4156655561-3743858824-1001\$RKJ92E8"),
    "b2d896": Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\archived\手掌数据"
                   r"\20260927_161747_single_device_b2d896"),
}
LIVE = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
            slope_cap_frac=0.009, r_slow_max=0.2, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)
RECOMMENDED = dict(LIVE, r_fast=0.03, slope_cap_frac=0.035, r_slow_max=0.35)


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
    wins = [v for v in (med_in(t, total, a, a + 10.0) for a in np.arange(30.0, t[-1] - 10.0, 10.0)) if v is not None]
    return dict(rise_adc=round(end - early, 1), landing_adc=round(end - e_level, 1),
                tail_slope_adc_s=round(slope, 4) if slope is not None else None,
                offset30_adc=round(float(np.max(np.abs(np.asarray(wins) - ref))), 1))


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

        arms = {}
        for label, params, guard in (
            ("live_cap009_guard_on", LIVE, dict(GUARD_DEFAULTS, enabled=True)),
            ("recommended_guard_on", RECOMMENDED, dict(GUARD_DEFAULTS, enabled=True)),
            ("recommended_guard_off", RECOMMENDED, dict(GUARD_DEFAULTS, enabled=False)),
        ):
            out, info = compensate(x.copy(), t, params, guard)
            arms[label] = dict(**metrics(t, out.sum(1), onset, e_level), guard_frames=info.get("guard_frames"))
        on, off = arms["recommended_guard_on"], arms["recommended_guard_off"]
        arms["guard_switch_delta"] = dict(
            rise_adc=round(on["rise_adc"] - off["rise_adc"], 2),
            offset30_adc=round(on["offset30_adc"] - off["offset30_adc"], 2),
            tail_slope_adc_s=round(on["tail_slope_adc_s"] - off["tail_slope_adc_s"], 5),
            landing_adc=round(on["landing_adc"] - off["landing_adc"], 2))
        report[tag] = arms
        print(f"=== {tag} (E={e_level:.0f}, onset={onset:.2f}s) ===")
        for label, values in arms.items():
            print(f"   {label:24s} {json.dumps(values, ensure_ascii=False)}")

    (OUT / "verify_recommended_guard.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    print(f"输出: {OUT / 'verify_recommended_guard.json'}")


if __name__ == "__main__":
    main()
