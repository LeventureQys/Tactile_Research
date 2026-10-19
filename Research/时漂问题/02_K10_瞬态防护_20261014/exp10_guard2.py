# -*- coding: utf-8 -*-
"""恢复性判据抑制器：偏离段结束后回到原电平才抹平；停在新电平则保留（真实变化）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
OUT = Path(__file__).resolve().parent
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(TOOLBOX))
from plugins.k9_creep_observer import _compensate  # noqa: E402

FPS = 100.0
RUN_S = 1150.0
INJECT_AT = 1000.0
WIN_S = 1.0
ABS_MIN = 30.0
SIGMA_MULT = 6.0
MAX_RUN_S = 2.0
SETTLE_S = 0.15
CHECK_S = 0.5
RECOVER_TOL_ADC = 25.0


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


def suppress(values: np.ndarray):
    window = int(WIN_S * FPS)
    reference = pd.DataFrame(values).rolling(window, min_periods=5).median().shift(1).bfill().to_numpy()
    step = np.diff(values, axis=0, prepend=values[:1])
    sigma = 1.4826 * np.median(np.abs(step - np.median(step, axis=0)), axis=0) / np.sqrt(2.0)
    threshold = np.maximum(ABS_MIN, SIGMA_MULT * np.maximum(sigma, 1e-9))
    flagged = np.abs(values - reference) > threshold
    out = values.copy()
    settle = int(SETTLE_S * FPS)
    check = int(CHECK_S * FPS)
    max_run = int(MAX_RUN_S * FPS)
    replaced = 0
    kept = 0
    for channel in range(values.shape[1]):
        column = flagged[:, channel]
        if not column.any():
            continue
        index = np.flatnonzero(column)
        for segment in np.split(index, np.flatnonzero(np.diff(index) > 1) + 1):
            first, last = int(segment[0]), int(segment[-1])
            if segment.size > max_run:
                kept += 1
                continue
            base_start = max(0, first - 20)
            base = float(np.median(values[base_start:first, channel])) if first > base_start else float(reference[first, channel])
            after_start = min(values.shape[0] - 1, last + settle)
            after_end = min(values.shape[0], last + settle + check)
            if after_end - after_start < 5:
                kept += 1
                continue
            after = float(np.median(values[after_start:after_end, channel]))
            if abs(after - base) <= RECOVER_TOL_ADC:
                out[first:last + 1, channel] = reference[first:last + 1, channel]
                replaced += 1
            else:
                kept += 1
    return out, flagged, replaced, kept


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

    normal = np.median(values_all[(time_all >= 194.2) & (time_all < 194.45)], axis=0)
    band = (time_all >= 194.45) & (time_all < 194.75)
    dip_shape = values_all[band] - normal
    dip_frames = dip_shape.shape[0]
    start = int(INJECT_AT * fps)

    def shift(total, lo=10.0, hi=60.0):
        before = (time >= INJECT_AT - 30.0) & (time < INJECT_AT - 1.0)
        after = (time >= INJECT_AT + lo) & (time < INJECT_AT + hi)
        return float(np.median(total[after]) - np.median(total[before]))

    injected = values.copy()
    injected[start:start + dip_frames] += dip_shape
    suppressed, flagged, replaced, kept = suppress(injected)

    clean = np.asarray(_compensate(values, time, params)[0], dtype=float).sum(axis=1)
    dirty = np.asarray(_compensate(injected, time, params)[0], dtype=float).sum(axis=1)
    guarded = np.asarray(_compensate(suppressed, time, params)[0], dtype=float).sum(axis=1)

    held = values.copy()
    hold_frames = int(20.0 * fps)
    held[start:start + hold_frames] += 2000.0 / values.shape[1]
    held_suppressed, held_flagged, held_replaced, held_kept = suppress(held)
    held_dirty = np.asarray(_compensate(held, time, params)[0], dtype=float).sum(axis=1)
    held_guarded = np.asarray(_compensate(held_suppressed, time, params)[0], dtype=float).sum(axis=1)

    unload = values.copy()
    unload[start:] -= values[start:].mean(axis=0) * 0.9
    unload_suppressed, _, unload_replaced, unload_kept = suppress(unload)
    unload_dirty = np.asarray(_compensate(unload, time, params)[0], dtype=float).sum(axis=1)
    unload_guarded = np.asarray(_compensate(unload_suppressed, time, params)[0], dtype=float).sum(axis=1)

    report = dict(
        config=dict(WIN_S=WIN_S, ABS_MIN=ABS_MIN, SIGMA_MULT=SIGMA_MULT, MAX_RUN_S=MAX_RUN_S,
                    SETTLE_S=SETTLE_S, CHECK_S=CHECK_S, RECOVER_TOL_ADC=RECOVER_TOL_ADC),
        dip_injection=dict(
            dip_frames=int(dip_frames),
            clean_shift=round(shift(clean), 2), dirty_shift=round(shift(dirty), 2),
            guarded_shift=round(shift(guarded), 2),
            dirty_vs_clean=round(shift(dirty) - shift(clean), 2),
            guarded_vs_clean=round(shift(guarded) - shift(clean), 2),
            suppressed_segments=int(replaced), kept_segments=int(kept),
            flagged_channels_at_event=int(flagged[start:start + dip_frames].sum()),
        ),
        sustained_step=dict(
            max_abs_output_diff=round(float(np.max(np.abs(held_dirty - held_guarded))), 3),
            suppressed_segments=int(held_replaced), kept_segments=int(held_kept),
            dirty_shift=round(shift(held_dirty), 2), guarded_shift=round(shift(held_guarded), 2),
        ),
        real_unload=dict(
            max_abs_output_diff=round(float(np.max(np.abs(unload_dirty - unload_guarded))), 3),
            suppressed_segments=int(unload_replaced), kept_segments=int(unload_kept),
        ),
    )
    (OUT / "guard2_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
