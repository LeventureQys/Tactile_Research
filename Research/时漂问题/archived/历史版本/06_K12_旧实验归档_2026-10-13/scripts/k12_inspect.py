# -*- coding: utf-8 -*-
"""K12 数据细查：10g 反复增减的真实形态与空间分布。

修正 k12_probe 的两处口径：
  1. 漂移评估窗必须跳过加载瞬态（用「总值首次越过空载带」定位加载完成时刻）。
  2. 总值噪声 ~40 ADC 远大于单通道 σ·√N（≈7.9），说明存在强共模；
     需要判断 10g 变化到底落在总值上还是少数通道上。

用法：python k12_inspect.py
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
ADC_PER_N = 1000.0


def load_pair(session_dir: Path):
    def tab(name):
        path = session_dir / name
        return scsv.load_table(str(path), use_cache=False) if path.exists() else None

    pre = tab("device_001_pre_seg0.csv")
    out = tab("device_001_seg000.csv")
    raw = tab("device_001_raw_seg000.csv")
    n = min(t.n_rows for t in (pre, out, raw) if t is not None)
    frame = np.asarray(pre.data[:n, 2], dtype=np.float64)
    fps = 100.45
    return (frame / fps,
            pre.channels().astype(np.float64)[:n],
            out.channels().astype(np.float64)[:n],
            raw.channels().astype(np.float64)[:n])


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


def find_on_time(total: np.ndarray, time: np.ndarray) -> float:
    """加载完成时刻：平滑总值首次超过「空载带 + 20% 量程」的时刻。"""
    smooth = med_smooth(total, 101)
    lo = float(np.percentile(smooth[:200], 50))
    hi = float(np.percentile(smooth, 90))
    thr = lo + 0.5 * (hi - lo)
    idx = np.argmax(smooth > thr)
    return float(time[idx]) if smooth[idx] > thr else float(time[0])


def main() -> int:
    for session_dir in sorted(DATA_ROOT.glob("设备*/*10g*")):
        time, pre, out, raw = load_pair(session_dir)
        t_pre = pre.sum(axis=1)
        t_out = out.sum(axis=1)
        t_raw = raw.sum(axis=1)
        t_on = find_on_time(t_pre, time)
        print("=" * 104)
        print("会话 %s" % session_dir.name)
        print("时长 %.1fs  加载完成 ≈ %.1fs" % (time[-1], t_on))
        print("=" * 104)

        # 1) 加载完成后的漂移（用 2s 中值平滑，窗口跳过前 1.5s）
        start = int(np.searchsorted(time, t_on + 1.5))
        end = time.size
        seg = slice(start, end)
        sm_pre = med_smooth(t_pre, 201)
        sm_out = med_smooth(t_out, 201)
        print("\n[漂移] 加载完成后 %d 帧（%.1fs）" % (end - start, time[-1] - time[start]))
        print("  pre  首→末 %8.0f → %8.0f ADC   净 %+8.0f (%.3f N)"
              % (sm_pre[start], sm_pre[-1], sm_pre[-1] - sm_pre[start],
                 (sm_pre[-1] - sm_pre[start]) / ADC_PER_N))
        print("  out  首→末 %8.0f → %8.0f ADC   净 %+8.0f (%.3f N)"
              % (sm_out[start], sm_out[-1], sm_out[-1] - sm_out[start],
                 (sm_out[-1] - sm_out[start]) / ADC_PER_N))
        rate_pre = np.polyfit(time[seg], sm_pre[seg], 1)[0]
        rate_out = np.polyfit(time[seg], sm_out[seg], 1)[0]
        print("  pre 斜率 %+.3f ADC/s   out 斜率 %+.3f ADC/s" % (rate_pre, rate_out))

        # 2) 1s 中值曲线采样（每 2s 一个点），直观看反复增减
        print("\n[总值曲线] 1s 中值平滑，每 2s 采样（pre / out）")
        step = int(2.0 / float(np.median(np.diff(time))))
        header = "   t(s): " + " ".join("%7.0f" % time[i] for i in range(0, time.size, step))
        print(header)
        print("   pre : " + " ".join("%7.0f" % sm_pre[i] for i in range(0, time.size, step)))
        print("   out : " + " ".join("%7.0f" % sm_out[i] for i in range(0, time.size, step)))

        # 3) 峰谷检测：1s 平滑上的显著转折（幅度 > 3σ_noise 的相邻极值差）
        sm = sm_pre
        diff = np.gradient(sm)
        sign = np.sign(diff)
        turn = np.flatnonzero(np.diff(sign) != 0)
        turn = turn[turn > start]
        if turn.size:
            vals = sm[turn]
            amps = np.abs(np.diff(vals))
            big = amps[amps > 60.0]
            print("\n[峰谷] 转折点 %d 个；相邻转折幅度 > 60 ADC 的有 %d 个" % (turn.size, big.size))
            if big.size:
                for q in (25, 50, 75, 90):
                    print("    p%-2d 转折幅度 %7.1f ADC (%6.1f mN)" % (q, np.percentile(big, q),
                                                                      np.percentile(big, q)))
                top = np.argsort(amps)[-8:][::-1]
                print("    最大的 8 个转折：")
                for i in top:
                    if amps[i] <= 60.0:
                        continue
                    print("      t=%6.1fs  幅度 %7.1f ADC  (%6.1f mN)  %s"
                          % (time[turn[i]], amps[i], amps[i], "↑" if np.diff(vals)[i] > 0 else "↓"))

        # 4) 空间分布：加载完成后，哪些通道有变化（用 raw 更干净）
        print("\n[空间] 加载完成后各通道变化（raw 域，首 5s 中位 vs 末 5s 中位）")
        n_quiet = max(50, int(5.0 / float(np.median(np.diff(time)))))
        head = raw[start:start + n_quiet].mean(axis=0)
        tail = raw[max(start, raw.shape[0] - n_quiet):].mean(axis=0)
        delta = tail - head
        order = np.argsort(-np.abs(delta))[:12]
        print("  通道: " + " ".join("%6d" % k for k in order))
        print("  变化: " + " ".join("%6.1f" % delta[k] for k in order))
        nz = np.flatnonzero(np.abs(delta) > 5.0)
        print("  |变化|>5 ADC 的通道数 %d / %d；这些通道的 Δ 合计 %.0f ADC，总值 Δ %.0f ADC"
              % (nz.size, delta.size, delta[nz].sum(), delta.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
