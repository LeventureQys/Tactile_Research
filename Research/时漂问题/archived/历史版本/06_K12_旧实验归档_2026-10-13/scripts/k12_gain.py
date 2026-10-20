# -*- coding: utf-8 -*-
"""K12 保护强度代价扫描：motion_slow_gain 与漂移压制的取舍。

保护层是「用慢态压制能力换真实变化保护」的结构。本脚本在 10g 会话上量化这条曲线：
    gain=1.0 → 保护无效（等价 K9-big）
    gain=0.5 → 保护期慢态半速
    gain=0.0 → 保护期完全冻结（默认）
若漂移随 gain 下降而单调恶化，则保护层在当前数据上是纯代价。

用法：python k12_gain.py
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
OUT = Path(__file__).resolve().parent.parent / "results"


def med_smooth(x, win):
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(x, pad, mode="edge")
    stride = padded.strides[0]
    view = np.lib.stride_tricks.as_strided(padded, shape=(x.size, win), strides=(stride, stride))
    return np.median(view, axis=1)


def residual(total, time, short_s=0.6, long_s=6.0):
    dt = float(np.median(np.diff(time)))
    return med_smooth(total, int(short_s / dt)) - med_smooth(total, int(long_s / dt))


def event_amps(resid, time, min_amp=60.0):
    dt = float(np.median(np.diff(time)))
    gap = max(1, int(0.3 / dt))
    amps, last = [], None
    for i in range(gap, resid.size - gap):
        window = resid[i - gap:i + gap + 1]
        kind = 1 if (resid[i] == window.max() and resid[i] > window.min()) else (
            -1 if (resid[i] == window.min() and resid[i] < window.max()) else 0)
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


def main() -> int:
    registry = alg.PluginRegistry()
    registry.scan()
    k9 = registry.get("k9_creep_observer")
    k12 = registry.get("k12_motion_guard")
    defaults = {k: v for k, v in alg.default_params(k9.specs).items() if k != "param_set"}

    targets = sorted(p for p in DATA_ROOT.glob("设备*/*10g*") if (p / "session.json").exists())
    rows = []
    for session_dir in targets:
        meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        table = scsv.load_table(str(session_dir / "device_001_pre_seg0.csv"), use_cache=False)
        values = table.channels().astype(np.float64)
        fps = float(meta["devices"][0].get("observed_fps") or 100.45)
        time_axis = np.asarray(table.data[:, 2], dtype=np.float64) / fps
        base = dict(defaults)
        base.update({k: v for k, v in meta["algorithm"]["params"].items() if k in base})
        big = dict(base, r_slow_max=0.3)
        ctx = alg.AlgorithmContext(mode="dataset", source_path=str(session_dir),
                                   time_axis=time_axis, values=values,
                                   channel_ids=tuple(table.channel_ids), meta=meta)

        pre_total = values.sum(axis=1)
        smooth = med_smooth(pre_total, 101)
        lo, hi = float(np.median(smooth[:200])), float(np.percentile(smooth, 90))
        on = int(np.argmax(smooth > lo + 0.5 * (hi - lo)))
        w = max(20, (pre_total.size - on) // 10)

        resid_pre = residual(pre_total, time_axis)
        events_pre = event_amps(resid_pre, time_axis)

        print("\n" + "=" * 88)
        print("会话 %s（%.1fs，pre 漂移 %+.0f ADC，事件 %d 个）"
              % (session_dir.name, time_axis[-1],
                 float(np.median(pre_total[-w:]) - np.median(pre_total[on:on + w])),
                 len(events_pre)))
        print("=" * 88)
        print("  %-16s %10s %10s %10s" % ("臂", "漂移ADC", "保护%", "事件保留"))

        arms = [("K9-big(无保护)", k9, dict(big))]
        for gain in (1.0, 0.5, 0.25, 0.0):
            arms.append(("K12 gain=%.2f" % gain, k12,
                         dict(big, motion_guard_enable=True, motion_thresh_adc=60.0,
                              motion_slow_gain=gain)))

        for name, plugin, params in arms:
            result = plugin.factory().run(ctx, params)
            total = result.outputs[0].y
            drift = float(np.median(total[-w:]) - np.median(total[on:on + w]))
            guard_pct = float("nan")
            if "保护触发" in result.note:
                try:
                    guard_pct = float(result.note.split("保护触发 ")[1].split(" 帧")[0]) / pre_total.size * 100.0
                except (IndexError, ValueError):
                    pass
            resid_out = residual(total, time_axis)
            keep = []
            for t_event, amp_pre in events_pre:
                window = np.abs(time_axis - t_event) < 1.0
                seg = resid_out[window]
                if seg.size >= 3 and amp_pre > 1e-9:
                    keep.append(float(seg.max() - seg.min()) / amp_pre)
            print("  %-16s %10.0f %10s %10.2f"
                  % (name, drift, "-" if np.isnan(guard_pct) else "%.1f" % guard_pct,
                     float(np.median(keep)) if keep else float("nan")))
            rows.append({"session": session_dir.name, "arm": name, "drift_adc": drift,
                         "guard_pct": None if np.isnan(guard_pct) else guard_pct,
                         "keep": float(np.median(keep)) if keep else None})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k12_gain.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\n结果写入 results/k12_gain.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
