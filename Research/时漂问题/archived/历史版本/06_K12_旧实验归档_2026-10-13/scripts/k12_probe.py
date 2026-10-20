# -*- coding: utf-8 -*-
"""K12 数据探查：为「微小变化保护层」确定阈值尺度。

回答三个问题：
  1. 静止平台的噪声尺度是多少（决定保护阈值下限）？
  2. 10g 反复增减在补偿前(pre)流上的真实台阶幅度是多少（决定保护阈值上限）？
  3. 8.9N 恒载长保压的漂移率是多少（决定慢态积分必须压制多少）？

用法：python k12_probe.py
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
ADC_PER_N = 1000.0  # 关键换算比：1000 ADC ≈ 1N


def load_session(session_dir: Path):
    """返回 dict：时间轴(秒)、pre 矩阵、out 矩阵、元信息。"""
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0].get("observed_fps") or 100.0)

    def table_at(name: str):
        path = session_dir / name
        return scsv.load_table(str(path), use_cache=False) if path.exists() else None

    pre = table_at("device_001_pre_seg0.csv")
    out = table_at("device_001_seg000.csv")
    frame = np.asarray(pre.data[:, 2], dtype=np.float64)
    time_axis = frame / fps
    return {
        "session": session_dir.name,
        "time": time_axis,
        "pre": pre.channels().astype(np.float64),
        "out": None if out is None else out.channels().astype(np.float64),
        "fps": fps,
        "params": meta["algorithm"]["params"],
    }


def high_freq_sigma(x: np.ndarray) -> float:
    """一阶差分估计噪声：静止段 diff 的稳健尺度 / sqrt(2)（1.4826·MAD 抗离群）。"""
    d = np.diff(x)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def step_sizes(total: np.ndarray, time: np.ndarray, smooth_s: float = 0.5):
    """找总值的台阶：用 1.5s 中值平滑后做 2s 窗差分，取显著峰的正负幅度。"""
    n = total.size
    # 用滑动中值平滑（窗口 = smooth_s / dt）
    dt = float(np.median(np.diff(time))) if n > 2 else 0.01
    win = max(3, int(round(smooth_s / max(dt, 1e-6))))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(total, pad, mode="edge")
    smooth = np.array([np.median(padded[i:i + win]) for i in range(n)])
    lag = max(1, int(round(1.5 / max(dt, 1e-6))))
    diff = np.full(n, np.nan)
    diff[lag:] = smooth[lag:] - smooth[:-lag]
    return smooth, diff


def main() -> int:
    sessions = sorted(p for p in DATA_ROOT.glob("设备*/*") if (p / "session.json").exists())
    print("=" * 100)
    print("K12 数据探查（换算比 %.0f ADC/N，即 1 ADC ≈ 1 mN）" % ADC_PER_N)
    print("=" * 100)

    rows = []
    for session_dir in sessions:
        data = load_session(session_dir)
        pre, out, time_axis = data["pre"], data["out"], data["time"]
        total_pre = pre.sum(axis=1)
        total_out = None if out is None else out.sum(axis=1)

        # 噪声：用前半段的静止平台估计（避开加载段）
        quiet = slice(int(pre.shape[0] * 0.02), int(pre.shape[0] * 0.06))
        sigma_pre = high_freq_sigma(total_pre[quiet])
        sigma_ch = float(np.median([high_freq_sigma(pre[quiet, k]) for k in range(pre.shape[1])]))

        smooth, diff = step_sizes(total_pre, time_axis)
        tail = slice(int(pre.shape[0] * 0.10), pre.shape[0])
        drift_rate = float(np.polyfit(time_axis[tail], smooth[tail], 1)[0]) if pre.shape[0] > 100 else float("nan")

        span_pre = float(np.nanpercentile(total_pre[tail], 95) - np.nanpercentile(total_pre[tail], 5))
        span_out = (float(np.nanpercentile(total_out[tail], 95) - np.nanpercentile(total_out[tail], 5))
                    if total_out is not None else float("nan"))
        net_out = (float(np.median(total_out[-max(20, pre.shape[0] // 20):])
                        - np.median(total_out[:max(20, pre.shape[0] // 20)]))
                   if total_out is not None else float("nan"))

        rows.append({
            "session": data["session"],
            "dur_s": float(time_axis[-1]),
            "frames": int(pre.shape[0]),
            "sigma_total": sigma_pre,
            "sigma_ch": sigma_ch,
            "drift_rate_pre": drift_rate,
            "span_pre": span_pre,
            "span_out": span_out,
            "net_out": net_out,
        })

        tag = "10g" if "10g" in data["session"] else "恒载"
        print("\n[%s] %s" % (tag, data["session"]))
        print("  时长 %.1fs / %d 帧 / fps %.2f" % (time_axis[-1], pre.shape[0], data["fps"]))
        print("  噪声(总值高频σ) %.2f ADC | 单通道σ中位 %.2f ADC" % (sigma_pre, sigma_ch))
        print("  pre 跨度(p95-p5) %.0f ADC | out 跨度 %.0f ADC" % (span_pre, span_out))
        print("  pre 末段漂移率 %.2f ADC/s (%.3f N/s)" % (drift_rate, drift_rate / ADC_PER_N))
        print("  out 首尾净变化 %.0f ADC (%.3f N)" % (net_out, net_out / ADC_PER_N))

        # 10g 会话：提取显著台阶
        if "10g" in data["session"]:
            valid = np.abs(diff[np.isfinite(diff)])
            thr = max(3.0 * sigma_pre, 5.0)
            peaks = np.abs(diff[np.isfinite(diff) & (np.abs(diff) > thr)])
            print("  台阶阈值 %.1f ADC → 显著变化点 %d 个" % (thr, peaks.size))
            if peaks.size:
                for q in (25, 50, 75, 90, 95):
                    v = float(np.percentile(peaks, q))
                    print("    p%-2d 台阶幅度 %7.1f ADC  (%6.1f mN)" % (q, v, v))

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print("%-42s %8s %10s %12s %12s" % ("session", "时长s", "σ_total", "pre跨度", "out净变化"))
    for r in rows:
        print("%-42s %8.1f %10.2f %12.0f %12.0f"
              % (r["session"], r["dur_s"], r["sigma_total"], r["span_pre"], r["net_out"]))
    with open(Path(__file__).with_name("k12_probe.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print("\n明细已写入 scripts/k12_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
