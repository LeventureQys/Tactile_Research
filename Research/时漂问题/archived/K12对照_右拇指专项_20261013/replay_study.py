# -*- coding: utf-8 -*-
"""右拇指指腹 K9 时漂专项重放。
只读六份原始录制；候选数据全部在内存中生成，不修改旧文件。
运行：python replay_study.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
DATA_ROOT = Path(__file__).parents[1] / "data" / "右拇指指腹"
OUT_JSON = Path(__file__).with_name("replay_study_results.json")

sys.path.insert(0, str(TOOLBOX))
from dptool.session_csv import load_table  # noqa: E402
from plugins.k9_creep_observer import _compensate  # noqa: E402

CANDIDATES = [0.1, 0.2, 0.3, 0.4]
SAMPLE_DIRS = [
    DATA_ROOT / "设备1" / "测试1-2026-10-13_16-32-15",
    DATA_ROOT / "设备1" / "测试2-2026-10-13_16-43-59",
    DATA_ROOT / "设备1" / "10g反复增减荷载-2026-10-13_17-07-18",
    DATA_ROOT / "设备2" / "测试1-2026-10-13_16-53-24",
    DATA_ROOT / "设备2" / "测试2-2026-10-13_17-01-06",
    DATA_ROOT / "设备2" / "10g反复增减荷载-2026-10-13_17-05-32",
]

def med(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if x.size else None

def load_sample(directory: Path):
    with (directory / "session.json").open(encoding="utf-8") as f:
        session = json.load(f)
    fps = float(session["devices"][0].get("observed_fps") or 100.0)
    table = load_table(str(directory / "device_001_pre_seg0.csv"), use_cache=False)
    values = table.channels().astype(float)
    time = table.time_axis("frame_index").astype(float) / fps
    out = load_table(str(directory / "device_001_seg000.csv"), use_cache=False).channels().astype(float)
    actual_total = np.nansum(out, axis=1)
    return session, values, time, actual_total, fps

def replay_params(values, time, params):
    output, stats = _compensate(values, time, dict(params))
    return np.nansum(output, axis=1), stats

def replay(values, time, params, rslow):
    p = dict(params)
    p["r_slow_max"] = float(rslow)
    output, stats = _compensate(values, time, p)
    return np.nansum(output, axis=1), stats

def first_load(time, signal):
    """历史口径：取总值相邻帧 abs(diff) 首次达到2000 ADC的跳变后时刻。"""
    jumps = np.abs(np.diff(np.asarray(signal, float)))
    indexes = np.flatnonzero(jumps >= 2000.0)
    return float(time[indexes[0] + 1]) if indexes.size else float(time[0])

def window_metrics(time, signal, start):
    a, b = start + 15.0, float(time[-1])
    m = (time >= a) & (time <= b)
    if not np.any(m):
        return {"start_s": a, "end_s": b, "n": 0, "net_change_adc": None, "max_abs_offset_adc_5s_median": None}
    # 净变化：窗口首尾各5 s中值；最大偏移：5 s滚动中值相对首5 s中值。
    first = med(signal[m & (time < a + 5)])
    last = med(signal[time >= float(time[-1]) - 5.0])
    starts = np.arange(a, b, 1.0)
    offsets = []
    for s in starts:
        z = signal[(time >= s) & (time < min(s + 5, b + 1e-9))]
        if z.size: offsets.append(med(z) - first)
    return {"start_s": a, "end_s": b, "n": int(np.sum(m)),
            "net_change_adc": float(last - first),
            "max_abs_offset_adc_5s_median": float(np.max(np.abs(offsets))) if offsets else None}

def inject(values, time, event_start, duration=15.0):
    x = values.copy()
    m = (time >= event_start) & (time < event_start + duration)
    x[m, :] += 100.0 / x.shape[1]
    return x, int(np.sum(m))

def injection_metrics(values, time, params, event_start, rslow):
    if rslow == "recorded_params":
        clean, _ = replay_params(values, time, params)
    else:
        clean, _ = replay(values, time, params, float(rslow))
    injected_values, n = inject(values, time, event_start)
    if rslow == "recorded_params":
        altered, _ = replay_params(injected_values, time, params)
    else:
        altered, _ = replay(injected_values, time, params, float(rslow))
    delta = altered - clean
    # 15 s事件后保留率取事件结束后5 s窗；同时报告事件中段，覆盖尾段吞掉情形。
    during = (time >= event_start + 10) & (time < event_start + 15)
    after = (time >= event_start + 15) & (time < event_start + 20)
    return {"event_start_s": event_start, "event_duration_s": 15.0,
            "injected_total_adc": 100.0, "frames": n,
            "retention_during_last5s": med(delta[during]) / 100.0 if np.any(during) else None,
            "retention_after_first5s": med(delta[after]) / 100.0 if np.any(after) else None,
            "output_delta_during_last5s_adc": med(delta[during]) if np.any(during) else None,
            "output_delta_after_first5s_adc": med(delta[after]) if np.any(after) else None}

def main():
    results = {"method": {"toolbox": str(TOOLBOX), "algorithm": "dptool plugins.k9_creep_observer._compensate",
                           "window": "加载后15s至尾端5s；净变化=首/尾5s中值差；最大偏移=窗口内5s滚动中值相对首5s中值",
                           "injection": "固定底噪背景上向57通道均匀注入总100 ADC、持续15s；排除首次加载；事件点=加载后30s与尾段前30s",
                           "ten_g_note": "两10g记录没有真值；所有候选波动仅作候选波动，不解释为真实10g事件"}, "samples": []}
    for directory in SAMPLE_DIRS:
        session, values, time, actual, fps = load_sample(directory)
        params = session["algorithm"]["params"]
        name = directory.parent.name + "/" + directory.name
        load_start = first_load(time, np.nansum(values, axis=1))
        kind = "10g" if "10g" in directory.name else "constant_load"
        sample = {"id": name, "kind": kind, "path": str(directory), "observed_fps": fps,
                  "time_axis": "frame_index / observed_fps", "duration_s": float(time[-1]),
                  "load_start_s": load_start, "active_recording_window": window_metrics(time, actual, load_start),
                  "replays": {}, "injection_pair_replays": {}}
        for r in ["recorded"] + CANDIDATES:
            if r == "recorded":
                sig = actual
                sample["replays"][r] = {"source": "device_001_seg000.csv", "window": window_metrics(time, sig, load_start)}
            else:
                sig, stats = replay(values, time, params, r)
                sample["replays"][str(r)] = {"r_slow_max": r, "window": window_metrics(time, sig, load_start),
                                              "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))}}
        # 注入点至少两个；太短记录跳过不完整尾段点。
        points = [load_start + 30.0, float(time[-1]) - 30.0]
        for point in points:
            if point >= load_start + 15 and point + 20 <= time[-1]:
                key = "load_plus_30s" if point < time[-1] - 40 else "tail_minus_30s"
                for r in ["recorded_params", 0.1, 0.3]:
                    sample["injection_pair_replays"].setdefault(key, {})[str(r)] = injection_metrics(values, time, params, point, r)
        results["samples"].append(sample)
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_JSON)

if __name__ == "__main__":
    main()
