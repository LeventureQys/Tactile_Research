# -*- coding: utf-8 -*-
"""验证瞬态保护：1) 与工具箱实现对拍；2) 测试床污染是否被阻断；3) 真实变化是否被保留。"""
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

    # 1) 对拍：guard 关闭应等价工具箱
    reference, _ = _compensate(values, time, params)
    plain, _ = compensate(values, time, params, dict(enabled=False))
    parity = float(np.max(np.abs(np.asarray(reference, dtype=float) - plain)))

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

    clean = np.asarray(_compensate(values, time, params)[0], dtype=float).sum(axis=1)
    dirty = np.asarray(_compensate(injected, time, params)[0], dtype=float).sum(axis=1)
    guarded_raw, guard_stats = compensate(injected, time, params, dict(enabled=True))
    guarded = np.asarray(guarded_raw, dtype=float).sum(axis=1)

    held = values.copy()
    hold_frames = int(20.0 * fps)
    held[start:start + hold_frames] += 2000.0 / values.shape[1]
    held_dirty = np.asarray(_compensate(held, time, params)[0], dtype=float).sum(axis=1)
    held_guarded = np.asarray(compensate(held, time, params, dict(enabled=True))[0], dtype=float).sum(axis=1)

    unloaded = values.copy()
    unloaded[start:] -= values[start:].mean(axis=0) * 0.9
    unload_dirty = np.asarray(_compensate(unloaded, time, params)[0], dtype=float).sum(axis=1)
    unload_guarded = np.asarray(compensate(unloaded, time, params, dict(enabled=True))[0], dtype=float).sum(axis=1)

    report = dict(
        parity_vs_toolbox_max_abs=round(parity, 6),
        dip_injection=dict(
            dip_frames=int(dip_frames),
            clean_shift=round(shift(clean), 2), dirty_shift=round(shift(dirty), 2),
            guarded_shift=round(shift(guarded), 2),
            dirty_vs_clean=round(shift(dirty) - shift(clean), 2),
            guarded_vs_clean=round(shift(guarded) - shift(clean), 2),
            guard_frames=int(guard_stats.get("guard_frames", 0)),
        ),
        sustained_step=dict(
            dirty_shift=round(shift(held_dirty), 2), guarded_shift=round(shift(held_guarded), 2),
            max_instant_diff=round(float(np.max(np.abs(held_dirty - held_guarded))), 2),
        ),
        real_unload=dict(
            max_instant_diff=round(float(np.max(np.abs(unload_dirty - unload_guarded))), 2),
        ),
    )
    (HERE / "guarded_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
