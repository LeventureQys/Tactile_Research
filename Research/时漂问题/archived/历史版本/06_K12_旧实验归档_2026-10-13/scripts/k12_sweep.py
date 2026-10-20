# -*- coding: utf-8 -*-
"""K12 参数组合扫描：验证「提高慢态预算 + 保护层防过扣」能否同时压漂移与保变化。

对照臂：
  K9-base   录制参数（r_slow_max=0.1）           —— 现役基线
  K9-big    仅把 r_slow_max 提到 0.3             —— 只调参（预期：漂移↓但可能过扣）
  K12-big   r_slow_max=0.3 + 保护层              —— 参数 + 结构
  K12-base  录制参数 + 保护层                    —— 只加结构
指标（全部在「加载完成后」窗口内）：
  漂移   = 末段中位 − 加载完成段中位（ADC）
  事件保留率 = 去趋势残差事件的幅度比 out/pre
用法：python k12_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLBOX = r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具"
if TOOLBOX not in sys.path:
    sys.path.insert(0, TOOLBOX)
from dptool import algorithms as alg  # noqa: E402
from dptool import session_csv as scsv  # noqa: E402

DATA_ROOT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右拇指指腹")
ADC_PER_N = 1000.0
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def med_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(x, pad, mode="edge")
    stride = padded.strides[0]
    view = np.lib.stride_tricks.as_strided(padded, shape=(x.size, win), strides=(stride, stride))
    return np.median(view, axis=1)


def residual(total: np.ndarray, time: np.ndarray, short_s=0.6, long_s=6.0) -> np.ndarray:
    dt = float(np.median(np.diff(time)))
    return med_smooth(total, int(short_s / dt)) - med_smooth(total, int(long_s / dt))


def event_amps(resid: np.ndarray, time: np.ndarray, min_amp=60.0):
    """残差上的显著峰谷幅度列表（带 0.3s 最小间隔）。"""
    dt = float(np.median(np.diff(time)))
    gap = max(1, int(0.3 / dt))
    amps, last = [], None
    for i in range(gap, resid.size - gap):
        window = resid[i - gap:i + gap + 1]
        kind = 0
        if resid[i] == window.max() and resid[i] > window.min():
            kind = 1
        elif resid[i] == window.min() and resid[i] < window.max():
            kind = -1
        if kind == 0:
            continue
        if last is not None and kind == last[2]:
            if (kind > 0 and resid[i] > last[1]) or (kind < 0 and resid[i] < last[1]):
                last = (i, resid[i], kind)
            continue
        if last is not None:
            amp = abs(resid[i] - last[1])
            if amp >= min_amp:
                amps.append((float(time[i]), float(amp)))
        last = (i, resid[i], kind)
    return amps


def load_pre(session_dir: Path):
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0].get("observed_fps") or 100.45)
    table = scsv.load_table(str(session_dir / "device_001_pre_seg0.csv"), use_cache=False)
    values = table.channels().astype(np.float64)
    time_axis = np.asarray(table.data[:, 2], dtype=np.float64) / fps
    return values, time_axis, tuple(table.channel_ids), meta, fps


def on_index(total: np.ndarray, time: np.ndarray) -> int:
    """加载完成帧：1s 中值平滑后首次超过「空载中位 + 50% 量程」处。"""
    smooth = med_smooth(total, 101)
    lo = float(np.median(smooth[:200]))
    hi = float(np.percentile(smooth, 90))
    idx = np.argmax(smooth > lo + 0.5 * (hi - lo))
    return int(idx) if smooth[idx] > lo + 0.5 * (hi - lo) else 0


def main() -> int:
    registry = alg.PluginRegistry()
    registry.scan()
    k9 = registry.get("k9_creep_observer")
    k12 = registry.get("k12_motion_guard")
    base_defaults = {k: v for k, v in alg.default_params(k9.specs).items() if k != "param_set"}

    sessions = [p for p in sorted(DATA_ROOT.glob("设备*/*")) if (p / "session.json").exists()]
    print("将处理 %d 个会话" % len(sessions))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for session_dir in sessions:
        values, time_axis, ids, meta, fps = load_pre(session_dir)
        recorded = meta["algorithm"]["params"]
        base = dict(base_defaults)
        base.update({k: v for k, v in recorded.items() if k in base})

        ctx = alg.AlgorithmContext(mode="dataset", source_path=str(session_dir), time_axis=time_axis,
                                   values=values, channel_ids=ids, meta=meta)
        pre_total = values.sum(axis=1)
        on = on_index(pre_total, time_axis)

        arms = {}
        arms["K9-base"] = (k9, dict(base))
        big = dict(base, r_slow_max=0.3)
        arms["K9-big"] = (k9, dict(big))
        arms["K12-big"] = (k12, dict(big, motion_guard_enable=True, motion_thresh_adc=60.0))

        resid_pre = residual(pre_total, time_axis)
        events_pre = event_amps(resid_pre, time_axis)
        w = max(20, (pre_total.size - on) // 10)

        print("\n" + "=" * 96)
        print("会话 %s  (%.1fs, 加载完成 %.2fs, pre 漂移 %+.0f ADC)"
              % (session_dir.name, time_axis[-1], time_axis[on],
                 float(np.median(pre_total[-w:]) - np.median(pre_total[on:on + w]))))
        print("=" * 96)
        print("  %-10s %10s %10s %12s %10s" % ("臂", "漂移ADC", "漂移N", "事件保留", "保护%"))

        for name, (plugin, params) in arms.items():
            result = plugin.factory().run(ctx, params)
            total = result.outputs[0].y
            drift = float(np.median(total[-w:]) - np.median(total[on:on + w]))
            # 事件保留率：在 pre 残差事件时刻，比较 out **残差**的同窗峰谷幅度（排除漂移成分）
            resid_out = residual(total, time_axis)
            keep = []
            for t_event, amp_pre in events_pre:
                window = np.abs(time_axis - t_event) < 1.0
                seg = resid_out[window]
                if seg.size < 3 or amp_pre <= 1e-9:
                    continue
                amp_out = float(seg.max() - seg.min())
                keep.append(amp_out / amp_pre)
            keep_med = float(np.median(keep)) if keep else float("nan")
            guard_pct = float("nan")
            if "保护触发" in result.note:
                try:
                    guard_pct = float(result.note.split("保护触发 ")[1].split(" 帧")[0]) / pre_total.size * 100.0
                except (IndexError, ValueError):
                    guard_pct = float("nan")
            print("  %-10s %10.0f %10.3f %12.2f %10s"
                  % (name, drift, drift / ADC_PER_N, keep_med,
                     "-" if np.isnan(guard_pct) else "%.1f" % guard_pct))
            table_rows.append({
                "session": session_dir.name, "arm": name, "drift_adc": drift,
                "drift_n": drift / ADC_PER_N, "keep_median": keep_med,
                "guard_pct": None if np.isnan(guard_pct) else guard_pct,
            })

    with open(OUT_DIR / "k12_sweep.json", "w", encoding="utf-8") as handle:
        json.dump(table_rows, handle, ensure_ascii=False, indent=2)
    print("\n结果写入 results/k12_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
