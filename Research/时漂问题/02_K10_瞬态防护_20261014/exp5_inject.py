# -*- coding: utf-8 -*-
"""受控复现：把固件实测的 194.5s 下探波形注入合成受载背景，观察是否产生永久基线抬升。"""
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
CH = 52
BASELINE = 6.0
LOAD = 837.0
LOAD_AT = 2.0
DURATION = 200.0
EVENT_AT = 60.0


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


def extract_dip_shape():
    """从固件 pre 提取 194.5s 下探的相对逐通道波形与持续帧数。"""
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    pre = load_stream("device_001_pre_seg0.csv")
    v = np.asarray(pre.iloc[:, 3:], dtype=float)
    frame_index = np.asarray(pre.iloc[:, 2], dtype=float)
    t = (frame_index - frame_index[0]) / fps

    normal = np.median(v[(t >= 194.2) & (t < 194.45)], axis=0)
    band = (t >= 194.45) & (t < 194.75)
    idx = np.flatnonzero(band)
    shape = v[idx] - normal
    total = shape.sum(axis=1)
    deepest = int(np.argmin(total))
    return dict(fps=round(fps, 4), normal_total=round(float(normal.sum()), 1),
                frames=int(shape.shape[0]), deepest_frame=deepest,
                total_trace=[round(float(x), 1) for x in total],
                shape=shape, normal=normal, t=t[idx] - t[idx][0])


def build_background():
    n = int(DURATION * FPS)
    time = np.arange(n) / FPS
    values = np.full((n, CH), BASELINE)
    values[time >= LOAD_AT] += LOAD
    return time, values


def med(sig, t, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def main() -> None:
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    params = session["algorithm"]["params"]
    dip = extract_dip_shape()
    time, background = build_background()

    # 把固件下探波形叠加到合成背景的 EVENT_AT 处（保持逐通道形状与帧序列）
    injected = background.copy()
    start = int(EVENT_AT * FPS)
    count = dip["shape"].shape[0]
    injected[start:start + count] += dip["shape"]

    clean_out, _ = _compensate(background, time, params)
    dip_out, _ = _compensate(injected, time, params)
    clean_total = np.asarray(clean_out, dtype=float).sum(axis=1)
    dip_total = np.asarray(dip_out, dtype=float).sum(axis=1)

    applied_clean_matrix = background - np.asarray(clean_out, dtype=float)
    applied_dip_matrix = injected - np.asarray(dip_out, dtype=float)
    applied_clean = applied_clean_matrix.sum(axis=1)
    applied_dip = applied_dip_matrix.sum(axis=1)

    before = (EVENT_AT - 20.0, EVENT_AT - 0.5)
    after = (EVENT_AT + 3.0, EVENT_AT + 20.0)
    report = dict(
        dip_frames=int(count),
        dip_total_min=round(float(np.min(dip["shape"].sum(axis=1))), 1),
        dip_deepest_channel=int(np.argmin(dip["shape"].min(axis=0))),
        dip_deepest_value=round(float(dip["shape"].min()), 1),
        output_shift_adc=round(med(dip_total, time, *after) - med(dip_total, time, *before)
                               - (med(clean_total, time, *after) - med(clean_total, time, *before)), 2),
        applied_shift_adc=round(med(applied_dip, time, *after) - med(applied_dip, time, *before), 2),
        clean_drift_adc=round(med(clean_total, time, *after) - med(clean_total, time, *before), 2),
        per_channel_applied_change=[
            round(float(x), 1) for x in
            (applied_dip_matrix[(time >= after[0]) & (time < after[1])].mean(axis=0)
             - applied_dip_matrix[(time >= before[0]) & (time < before[1])].mean(axis=0))],
    )
    (OUT / "inject_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_channel_applied_change"}, ensure_ascii=False, indent=1))
    trace = [(round(float(time[i]), 2), round(float(dip_total[i] - clean_total[i]), 1))
             for i in range(int((EVENT_AT - 2) * FPS), int((EVENT_AT + 5) * FPS), 50)]
    print("注入响应轨迹（秒, 补偿差异ADC）:", trace)


if __name__ == "__main__":
    main()
