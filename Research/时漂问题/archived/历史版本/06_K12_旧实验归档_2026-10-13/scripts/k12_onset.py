# -*- coding: utf-8 -*-
"""K12 加载段细查：定位「10g 反复增减」的真实时刻与幅度。

前一轮看到 pre 总值在加载后单调上升（+20 ADC/s），没有明显锯齿。
本脚本把加载段落到 0.05~0.2s 分辨率，并同时给出：
  - pre / out / raw 三条流的总值；
  - 逐通道空间分布（用 pre 的 57 通道，raw 流的 channels() 为空）；
  - 一阶差分的高分位，用于区分「真实台阶」与「噪声转折」。

用法：python k12_onset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLBOX = r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具"
if TOOLBOX not in sys.path:
    sys.path.insert(0, TOOLBOX)
from dptool import session_csv as scsv  # noqa: E402

DATA_ROOT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右拇指指腹")
FPS = 100.45


def load_triple(session_dir: Path):
    def tab(name):
        path = session_dir / name
        return scsv.load_table(str(path), use_cache=False) if path.exists() else None

    pre, out, raw = tab("device_001_pre_seg0.csv"), tab("device_001_seg000.csv"), tab("device_001_raw_seg000.csv")
    n = min(t.n_rows for t in (pre, out, raw) if t is not None)
    frame = np.asarray(pre.data[:n, 2], dtype=np.float64)
    return frame / FPS, pre.channels().astype(np.float64)[:n], out.channels().astype(np.float64)[:n], raw


def print_trace(name, time, series, t0, t1, dt):
    """按 dt 分辨率打印一段总值轨迹。"""
    mask = (time >= t0) & (time <= t1)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    step = max(1, int(round(dt / float(np.median(np.diff(time))))))
    sel = idx[::step]
    print("\n[%s] %.1f~%.1fs 每 %.2fs 一点" % (name, t0, t1, dt))
    print("   t: " + " ".join("%6.2f" % time[i] for i in sel))
    for label, arr in series:
        print("  %-4s " % label + " ".join("%6.0f" % arr[i] for i in sel))


def main() -> int:
    for session_dir in sorted(DATA_ROOT.glob("设备*/*10g*")):
        time, pre, out, raw = load_triple(session_dir)
        p, o = pre.sum(axis=1), out.sum(axis=1)
        print("=" * 110)
        print("会话 %s   帧 %d   时长 %.1fs" % (session_dir.name, time.size, time[-1]))
        print("=" * 110)

        # 加载段（0~6s）高分辨率
        print_trace("加载段", time, [("pre", p), ("out", o)], 0.0, 6.0, 0.10)
        # 保压段抽样（看有无锯齿）
        print_trace("保压段抽样", time, [("pre", p), ("out", o)], 10.0, 30.0, 0.5)

        # 一阶差分统计：真实台阶 vs 噪声
        d = np.diff(p)
        print("\n[差分] pre 总值一阶差分：σ=%.1f  p99.9=%.1f  max=%.1f  min=%.1f"
              % (d.std(), np.percentile(np.abs(d), 99.9), d.max(), d.min()))
        d_out = np.diff(o)
        print("[差分] out 总值一阶差分：σ=%.1f  p99.9=%.1f  max=%.1f  min=%.1f"
              % (d_out.std(), np.percentile(np.abs(d_out), 99.9), d_out.max(), d_out.min()))

        # 空间分布：加载前后逐通道（用 pre 通道矩阵）
        quiet = slice(int(0.2 / float(np.median(np.diff(time)))),
                      int(0.9 / float(np.median(np.diff(time)))))
        early = pre[int(3.0 / float(np.median(np.diff(time)))):int(5.0 / float(np.median(np.diff(time))))].mean(axis=0)
        late = pre[int((time[-1] - 3.0) / float(np.median(np.diff(time)))):].mean(axis=0)
        delta = late - early
        order = np.argsort(-np.abs(delta))[:14]
        print("\n[空间] pre 域：t=3~5s 均值 → 末 3s 均值 的逐通道变化")
        print("  通道: " + " ".join("%6d" % k for k in order))
        print("  变化: " + " ".join("%6.1f" % delta[k] for k in order))
        nz = np.flatnonzero(np.abs(delta) > 10.0)
        print("  |Δ|>10 ADC 的通道 %d/%d；这些通道 Δ 合计 %.0f（总值 Δ %.0f）"
              % (nz.size, delta.size, delta[nz].sum(), delta.sum()))
        print("  全通道 Δ 的均值 %.1f，中位 %.1f（共模成分），残差 σ %.1f（空间差异）"
              % (delta.mean(), np.median(delta), (delta - np.median(delta)).std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
