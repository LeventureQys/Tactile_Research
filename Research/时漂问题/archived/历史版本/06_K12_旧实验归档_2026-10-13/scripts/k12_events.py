# -*- coding: utf-8 -*-
"""K12 事件提取：用「去趋势残差」把 10g 级真实增减从噪声与慢漂里分离出来。

思路（全部因果，可在线实现）：
    smooth = 短窗中值(pre)          —— 压掉高频噪声
    trend  = 长窗中值(smooth)       —— 慢漂/蠕变的趋势
    resid  = smooth − trend         —— 局部真实波动（10g 增减应落在这里）
在 resid 上找显著峰谷，即为「真实小荷载事件」的候选。

同时在同一口径下比较 pre / out(K9)，直接回答：
    K9 补偿后，这些残差事件还剩多少幅度（= 是否被慢态吞掉）。

用法：python k12_events.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLBOX = r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具"
if TOOLBOX not in sys.path:
    sys.path.insert(0, TOOLBOX)
from dptool import session_csv as scsv  # noqa: E402

DATA_ROOT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右拇指指腹")
ADC_PER_N = 1000.0   # 用户口径：1000 ADC ≈ 1N
FPS = 100.45


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


def residual_events(total: np.ndarray, time: np.ndarray, short_s=0.6, long_s=6.0,
                    min_amp=60.0):
    """返回 (resid, 事件列表)；事件 = resid 上 |幅差| > min_amp 的相邻峰谷对。"""
    dt = float(np.median(np.diff(time)))
    resid = med_smooth(total, int(short_s / dt)) - med_smooth(total, int(long_s / dt))
    # 找局部极值（三点比较，带 0.3s 最小间隔）
    gap = max(1, int(0.3 / dt))
    extrema = []
    for i in range(gap, resid.size - gap):
        window = resid[i - gap:i + gap + 1]
        if resid[i] == window.max() and resid[i] > window.min():
            extrema.append((i, resid[i], 1))
        elif resid[i] == window.min() and resid[i] < window.max():
            extrema.append((i, resid[i], -1))
    # 去重 + 相邻峰谷配对
    events = []
    last = None
    for idx, val, kind in extrema:
        if last is not None and kind == last[2]:
            # 同向连续极值取更极端者
            if (kind > 0 and val > last[1]) or (kind < 0 and val < last[1]):
                last = (idx, val, kind)
            continue
        if last is not None:
            amp = abs(val - last[1])
            if amp >= min_amp:
                events.append({
                    "t": float(time[idx]),
                    "amp": float(amp),
                    "dir": "up" if val > last[1] else "down",
                    "dur": float(time[idx] - time[last[0]]),
                })
        last = (idx, val, kind)
    return resid, events


def main() -> int:
    report = []
    for session_dir in sorted(DATA_ROOT.glob("设备*/*")):
        if not (session_dir / "session.json").exists():
            continue
        def tab(name):
            path = session_dir / name
            return scsv.load_table(str(path), use_cache=False) if path.exists() else None

        pre_t, out_t = tab("device_001_pre_seg0.csv"), tab("device_001_seg000.csv")
        if pre_t is None or out_t is None:
            continue
        n = min(pre_t.n_rows, out_t.n_rows)
        time = np.asarray(pre_t.data[:n, 2], dtype=np.float64) / FPS
        p = pre_t.channels().astype(np.float64)[:n].sum(axis=1)
        o = out_t.channels().astype(np.float64)[:n].sum(axis=1)

        rp, ep = residual_events(p, time)
        ro, eo = residual_events(o, time)

        tag = "10g" if "10g" in session_dir.name else "恒载"
        print("=" * 104)
        print("[%s] %s   %.1fs  残差σ: pre %.1f / out %.1f ADC"
              % (tag, session_dir.name, time[-1], rp.std(), ro.std()))
        print("=" * 104)
        amps_p = np.array([e["amp"] for e in ep])
        amps_o = np.array([e["amp"] for e in eo])
        print("  真实变化事件(残差 |峰谷差| ≥ 60 ADC)：")
        print("    pre 事件 %3d 个  幅度 p25/p50/p75 = %.0f / %.0f / %.0f ADC"
              % (amps_p.size, *(np.percentile(amps_p, [25, 50, 75]) if amps_p.size else [np.nan] * 3)))
        print("    out 事件 %3d 个  幅度 p25/p50/p75 = %.0f / %.0f / %.0f ADC"
              % (amps_o.size, *(np.percentile(amps_o, [25, 50, 75]) if amps_o.size else [np.nan] * 3)))

        # 逐事件配对：pre 事件处，out 上同窗口的幅度保留率
        if amps_p.size:
            keep = []
            for event in ep:
                window = np.abs(time - event["t"]) < max(event["dur"], 1.0)
                if window.sum() < 3:
                    continue
                span_p = float(p[window].max() - p[window].min())
                span_o = float(o[window].max() - o[window].min())
                if span_p > 1e-9:
                    keep.append(span_o / span_p)
            if keep:
                keep = np.array(keep)
                print("    事件窗口幅度保留率(out/pre)：中位 %.2f  p25 %.2f  p75 %.2f"
                      % (np.median(keep), np.percentile(keep, 25), np.percentile(keep, 75)))

        # 漂移（加载完成后）
        on = int(np.searchsorted(time, 3.0))
        sm_p, sm_o = med_smooth(p, 201), med_smooth(o, 201)
        print("  加载后(>3s)漂移：pre %+.0f ADC (%+.2f N)   out %+.0f ADC (%+.2f N)"
              % (sm_p[-1] - sm_p[on], (sm_p[-1] - sm_p[on]) / ADC_PER_N,
                 sm_o[-1] - sm_o[on], (sm_o[-1] - sm_o[on]) / ADC_PER_N))
        report.append({
            "session": session_dir.name, "tag": tag, "dur_s": float(time[-1]),
            "resid_sigma_pre": float(rp.std()), "resid_sigma_out": float(ro.std()),
            "events_pre": int(amps_p.size), "events_out": int(amps_o.size),
            "amp_p50_pre": float(np.median(amps_p)) if amps_p.size else None,
            "amp_p50_out": float(np.median(amps_o)) if amps_o.size else None,
            "drift_pre": float(sm_p[-1] - sm_p[on]), "drift_out": float(sm_o[-1] - sm_o[on]),
        })
    with open(Path(__file__).with_name("k12_events.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("\n明细已写入 scripts/k12_events.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
