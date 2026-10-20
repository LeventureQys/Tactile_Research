# -*- coding: utf-8 -*-
"""短时偏离抑制：识别并替换「偏离长窗参考且持续较短」的整段，保留持续真实变化。"""
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
ABS_MIN = 40.0
SIGMA_MULT = 6.0
MAX_RUN_S = 0.5


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


def suppress_short_excursions(values: np.ndarray):
    window = int(WIN_S * FPS)
    frame = pd.DataFrame(values)
    reference = frame.rolling(window, min_periods=5).median().shift(1).bfill().to_numpy()
    step = np.diff(values, axis=0, prepend=values[:1])
    robust_sigma = 1.4826 * np.median(np.abs(step - np.median(step, axis=0)), axis=0) / np.sqrt(2.0)
    threshold = np.maximum(ABS_MIN, SIGMA_MULT * np.maximum(robust_sigma, 1e-9))
    flagged = np.abs(values - reference) > threshold
    out = values.copy()
    max_run = int(MAX_RUN_S * FPS)
    replaced_segments = 0
    for channel in range(values.shape[1]):
        column = flagged[:, channel]
        if not column.any():
            continue
        index = np.flatnonzero(column)
        for segment in np.split(index, np.flatnonzero(np.diff(index) > 1) + 1):
            if segment.size <= max_run:
                out[segment, channel] = reference[segment, channel]
                replaced_segments += 1
    return out, flagged, threshold, replaced_segments


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

    results = {}

    # 场景 A：真实固件下探波形注入
    injected = values.copy()
    injected[start:start + dip_frames] += dip_shape
    suppressed, flagged, threshold, segments = suppress_short_excursions(injected)
    clean = np.asarray(_compensate(values, time, params)[0], dtype=float).sum(axis=1)
    dirty = np.asarray(_compensate(injected, time, params)[0], dtype=float).sum(axis=1)
    guarded = np.asarray(_compensate(suppressed, time, params)[0], dtype=float).sum(axis=1)
    results["dip_injection"] = dict(
        dip_frames=int(dip_frames),
        clean_shift=round(shift(clean), 2), dirty_shift=round(shift(dirty), 2),
        guarded_shift=round(shift(guarded), 2),
        dirty_vs_clean=round(shift(dirty) - shift(clean), 2),
        guarded_vs_clean=round(shift(guarded) - shift(clean), 2),
        flagged_channels_at_event=int(flagged[start:start + dip_frames].sum()),
        replaced_segments=int(segments),
        sigma_median=round(float(np.median(np.maximum(1.4826 * np.median(np.abs(np.diff(values, axis=0) - np.median(np.diff(values, axis=0), axis=0)), axis=0) / np.sqrt(2.0), 1e-9))), 3),
    )

    # 场景 B：真实持续加载（+2000 ADC 持续 20s），检查抑制器是否误滤
    held = values.copy()
    hold_frames = int(20.0 * fps)
    held[start:start + hold_frames] += 2000.0 / values.shape[1]
    held_suppressed, held_flagged, _, held_segments = suppress_short_excursions(held)
    held_dirty = np.asarray(_compensate(held, time, params)[0], dtype=float).sum(axis=1)
    held_guarded = np.asarray(_compensate(held_suppressed, time, params)[0], dtype=float).sum(axis=1)
    window = (time >= INJECT_AT + 5.0) & (time < INJECT_AT + 20.0)
    results["sustained_step"] = dict(
        max_abs_output_diff=round(float(np.max(np.abs(held_dirty - held_guarded))), 3),
        flagged_frames=int(held_flagged[start:start + hold_frames].any(axis=1).sum()),
        replaced_segments=int(held_segments),
        dirty_shift=round(shift(held_dirty), 2), guarded_shift=round(shift(held_guarded), 2),
    )

    (OUT / "guard_result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
