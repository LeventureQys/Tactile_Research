# -*- coding: utf-8 -*-
"""真实台阶能不能被路线B 的创新检测看见？—— 从数据本身复算创新量。

路线B（route_b.py）的判据是：
    short = 最近 0.25 s 的因果均值；ref = 0.5 s 之前的 short
    total = Σ(short − ref)；innovation = total − 0.5·trend；|innovation| ≥ 45 才算候选
其中 short/ref 完全由输入数据决定，可以脱离算法状态逐帧复算。
本脚本给出真实录制上台阶处 total 的峰值与台阶自身的上升时间，
用于判断「没触发」是阈值问题还是数据形态问题。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import SESSIONS, load_session  # noqa: E402

WINDOW_S = 0.25     # route_b CONFIG["window_s"]
SEPARATION_S = 0.50  # route_b CONFIG["separation_s"]
THRESHOLD = 45.0     # route_b CONFIG["threshold_adc"]


def causal_mean(t: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    """按时间做因果（含当前帧）移动均值，窗口内不足也照算（与 deque 裁剪行为一致）。"""
    out = np.empty_like(y)
    left = 0
    acc = 0.0
    for i in range(y.size):
        acc += y[i]
        while t[i] - t[left] > window_s:
            acc -= y[left]
            left += 1
        out[i] = acc / (i - left + 1)
    return out


def shift_lookup(t: np.ndarray, value: np.ndarray, lag_s: float) -> np.ndarray:
    """取「lag_s 之前」最近一帧的值；不足则前向填充首值。"""
    idx = np.searchsorted(t, t - lag_s, side="right") - 1
    idx = np.clip(idx, 0, value.size - 1)
    return value[idx]


def rise_time(t: np.ndarray, y: np.ndarray, t0: float, span: float = 12.0) -> dict:
    """台阶 10%~90% 上升时间（以台阶前后各 2 s 中值为端点）。"""
    before = np.median(y[(t >= t0 - 2.5) & (t < t0 - 0.5)])
    after = np.median(y[(t >= t0 + 2.0) & (t < t0 + 4.0)])
    amp = after - before
    if abs(amp) < 1.0:
        return dict(amp_adc=float(amp), t10=None, t90=None, rise_s=None)
    lo, hi = before + 0.1 * amp, before + 0.9 * amp
    mask = (t >= t0 - 1.0) & (t <= t0 + span)
    seg_t, seg_y = t[mask], y[mask]
    t10 = t90 = None
    for tt, yy in zip(seg_t, seg_y):
        if t10 is None and (yy >= lo if amp > 0 else yy <= lo):
            t10 = float(tt)
        if t90 is None and (yy >= hi if amp > 0 else yy <= hi):
            t90 = float(tt)
            break
    return dict(amp_adc=float(amp), t10=t10, t90=t90,
                rise_s=None if (t10 is None or t90 is None) else float(t90 - t10))


def main() -> int:
    for label, device, name in SESSIONS:
        sess = load_session(device, name)
        t = sess["t"]
        total = sess["pre"].sum(1)
        short = causal_mean(t, total, WINDOW_S)
        ref = shift_lookup(t, short, SEPARATION_S)
        innovation_raw = short - ref
        load = t[int(np.argmax(np.abs(np.diff(total)) >= 2000)) + 1] if np.any(
            np.abs(np.diff(total)) >= 2000) else float(t[0])

        print("\n=== %s（%.1f s，%d 帧）" % (label, t[-1], t.size))
        print("  原始创新量 |Σ(short-ref)|：全程 max=%.1f，>%.0f 的帧 %.2f%%，"
              "首次大加载后 10 s 起的 max=%.1f"
              % (float(np.max(np.abs(innovation_raw))),
                 THRESHOLD, 100.0 * float(np.mean(np.abs(innovation_raw) >= THRESHOLD)),
                 float(np.max(np.abs(innovation_raw[t >= load + 10.])))
                 if np.any(t >= load + 10.) else float("nan")))

        # 真实台阶：用较细阈值，逐帧跳变 > 120 ADC 或 2 秒窗口内平台差 > 120 ADC
        steps = platform_steps(t, total, 120.0)
        print("  平台台阶（前后 2 s 中值差 ≥120 ADC，已合并）共 %d 个：" % len(steps))
        print("    %-8s %9s %9s %9s %9s  %s"
              % ("时刻s", "台阶ADC", "上升时间s", "创新峰值", "创新均值", "≥45?"))
        for step in steps[:14]:
            ts = step["t"]
            mask = (t >= ts - 0.3) & (t <= ts + 6.0)
            peak = float(np.max(np.abs(innovation_raw[mask]))) if mask.any() else 0.0
            mean = float(np.mean(innovation_raw[mask])) if mask.any() else 0.0
            info = rise_time(t, total, ts)
            print("    %-8.2f %9.1f %9s %9.1f %9.1f  %s"
                  % (ts, info["amp_adc"],
                     "—" if info["rise_s"] is None else "%.2f" % info["rise_s"],
                     peak, mean, "是" if peak >= THRESHOLD else "否"))
    return 0


def platform_steps(t: np.ndarray, total: np.ndarray, min_adc: float,
                   refractory_s: float = 3.0) -> list[dict]:
    """相邻 2 s 平台中值差 ≥ min_adc 的台阶；用于刻画真实人手上的加载动作。"""
    step = max(1, int(round(0.05 / float(np.median(np.diff(t))))))
    grid = np.arange(t[0] + 2.0, t[-1] - 4.0, 0.1)
    before = np.array([np.median(total[(t >= g - 2.0) & (t < g)]) for g in grid])
    after = np.array([np.median(total[(t >= g) & (t < g + 2.0)]) for g in grid])
    diff = after - before
    hits = np.flatnonzero(np.abs(diff) >= min_adc)
    out: list[dict] = []
    for i in hits:
        if out and grid[i] - out[-1]["t"] < refractory_s:
            continue
        out.append(dict(t=float(grid[i]), adc=float(diff[i])))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
