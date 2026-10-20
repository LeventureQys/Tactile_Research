# -*- coding: utf-8 -*-
"""测试床：长时重放让 y_floor 接近工作电平后注入下探；对比输入侧瞬态抑制的效果。

链路假设（cpp creep_observer.cpp 行号）：
  104-105  y_floor 下行瞬时跟随
  111-113  idle 成立 -> load_dwell 归零、zero 更新
  174-180  idle_k -> x_slow 按 tau_r_slow_idle_s=0.2s 快泄放
=> 短时深下探可让 y_floor 触底、触发 idle，永久扣减慢态补偿量。
"""
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


def suppress_transients(values: np.ndarray, win_frames: int = 5, abs_min: float = 60.0,
                        sigma_mult: float = 8.0):
    """因果输入侧瞬态抑制：与上一小窗中值偏离过大时用该中值替换；只修复短段。"""
    frame = pd.DataFrame(values)
    reference = frame.rolling(win_frames, min_periods=1).median().shift(1)
    reference = reference.bfill().to_numpy()
    residual = values - reference
    step = np.diff(values, axis=0, prepend=values[:1])
    robust_sigma = 1.4826 * np.median(np.abs(step - np.median(step, axis=0)), axis=0) / np.sqrt(2.0)
    threshold = np.maximum(abs_min, sigma_mult * np.maximum(robust_sigma, 1e-9))
    flagged = np.abs(residual) > threshold
    out = values.copy()
    out[flagged] = reference[flagged]
    return out, flagged


def applied_total(values: np.ndarray) -> np.ndarray:
    return values.sum(axis=1)


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

    # 提取固件 194.5s 下探的逐通道波形（相对事件前中值）
    normal = np.median(values_all[(time_all >= 194.2) & (time_all < 194.45)], axis=0)
    band = (time_all >= 194.45) & (time_all < 194.75)
    dip_shape = values_all[band] - normal
    dip_frames = dip_shape.shape[0]

    start = int(INJECT_AT * fps)
    injected = values.copy()
    injected[start:start + dip_frames] += dip_shape

    suppressed, flagged = suppress_transients(injected)

    clean_out, _ = _compensate(values, time, params)
    dirty_out, _ = _compensate(injected, time, params)
    guarded_out, _ = _compensate(suppressed, time, params)

    clean_total = np.asarray(clean_out, dtype=float).sum(axis=1)
    dirty_total = np.asarray(dirty_out, dtype=float).sum(axis=1)
    guarded_total = np.asarray(guarded_out, dtype=float).sum(axis=1)

    before = (time >= INJECT_AT - 30.0) & (time < INJECT_AT - 1.0)
    after = (time >= INJECT_AT + 10.0) & (time < INJECT_AT + 60.0)

    def shift(total):
        return float(np.median(total[after]) - np.median(total[before]))

    report = dict(
        run_s=RUN_S, inject_at=INJECT_AT, dip_frames=int(dip_frames),
        dip_total_min=round(float(dip_shape.sum(axis=1).min()), 1),
        clean_shift=round(shift(clean_total), 2),
        dirty_shift=round(shift(dirty_total), 2),
        guarded_shift=round(shift(guarded_total), 2),
        dirty_vs_clean=round(shift(dirty_total) - shift(clean_total), 2),
        guarded_vs_clean=round(shift(guarded_total) - shift(clean_total), 2),
        flagged_frames_total=int(flagged.any(axis=1).sum()),
        flagged_frames_at_event=int(flagged[start:start + dip_frames].any(axis=1).sum()),
        flagged_per_channel_at_event=int(flagged[start:start + dip_frames].sum()),
    )
    (OUT / "bed_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
