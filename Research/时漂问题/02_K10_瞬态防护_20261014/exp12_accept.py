# -*- coding: utf-8 -*-
"""防护验收：阻断瞬态污染，并量化对真实加载/卸载的稳态代价。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLBOX))
from guarded_k9 import compensate  # noqa: E402
from plugins.k9_creep_observer import _compensate  # noqa: E402

FPS = 100.0
RUN_S = 1150.0
INJECT_AT = 1000.0
GUARD = dict(enabled=True)


def load_stream(name: str) -> pd.DataFrame:
    path = FOLDER / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def main() -> None:
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    params = session["algorithm"]["params"]
    fps = float(session["devices"][0]["observed_fps"])
    pre = load_stream("device_001_pre_seg0.csv")
    values_all = np.asarray(pre.iloc[:, 3:], dtype=float)
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    time_all = (frame - frame[0]) / fps
    limit = int(np.searchsorted(time_all, RUN_S))
    values = values_all[:limit].copy()
    time = time_all[:limit]

    reference, _ = _compensate(values, time, params)
    parity = float(np.max(np.abs(np.asarray(reference, dtype=float)
                                - compensate(values, time, params, dict(enabled=False))[0])))

    normal = np.median(values_all[(time_all >= 194.2) & (time_all < 194.45)], axis=0)
    band = (time_all >= 194.45) & (time_all < 194.75)
    dip_shape = values_all[band] - normal
    dip_frames = dip_shape.shape[0]
    start = int(INJECT_AT * fps)

    def window_median(sig, lo, hi):
        mask = (time >= lo) & (time < hi)
        return float(np.median(sig[mask])) if mask.any() else float("nan")

    def shift_of(sig, lo=10.0, hi=60.0):
        return window_median(sig, INJECT_AT + lo, INJECT_AT + hi) - window_median(sig, INJECT_AT - 30.0, INJECT_AT - 1.0)

    def scenario(inject_fn, tag):
        base = inject_fn(values)
        dirty = np.asarray(_compensate(base, time, params)[0], dtype=float).sum(axis=1)
        guarded = np.asarray(compensate(base, time, params, GUARD)[0], dtype=float).sum(axis=1)
        diff = np.abs(dirty - guarded)
        peak_index = int(np.argmax(diff))
        return dict(
            tag=tag,
            dirty_shift=round(shift_of(dirty), 2), guarded_shift=round(shift_of(guarded), 2),
            peak_instant_diff=round(float(diff.max()), 2),
            peak_at_s=round(float(time[peak_index]), 2),
            steady_diff_60_300s=round(abs(window_median(dirty, INJECT_AT + 60.0, INJECT_AT + 300.0)
                                        - window_median(guarded, INJECT_AT + 60.0, INJECT_AT + 300.0)), 2),
            steady_diff_last100s=round(abs(window_median(dirty, float(time[-1]) - 100.0, float(time[-1]))
                                           - window_median(guarded, float(time[-1]) - 100.0, float(time[-1]))), 2),
        )

    def with_dip(v):
        out = v.copy()
        out[start:start + dip_frames] += dip_shape
        return out

    def with_step(v):
        out = v.copy()
        out[start:start + int(20.0 * fps)] += 2000.0 / v.shape[1]
        return out

    def with_unload(v):
        out = v.copy()
        out[start:] -= v[start:].mean(axis=0) * 0.9
        return out

    def with_nothing(v):
        return v.copy()

    report = dict(
        parity_vs_toolbox_max_abs=round(parity, 6),
        guard_config=GUARD,
        scenarios=[scenario(with_dip, "dip_injection"),
                   scenario(with_step, "sustained_step_20s"),
                   scenario(with_unload, "real_unload_90pct"),
                   scenario(with_nothing, "no_injection_control")],
    )
    (HERE / "guard_accept.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
