"""临时合成数据：真值已知的 60 min 恒载，用来量"下漂"有多少是算法自己造的。

真实录制里没有外部力参考，无法区分"读数下降"是载荷真的变小、还是补偿做过了。
这份合成数据的载荷恒定不变，因此任何输出变化都是纯算法误差。

构成（固定随机种子，可复现）：
    pre[t, i] = base_i + zero_drift_i * t/T + noise
    base_i     ：57 通道合计 30000 ADC（约 30 N），恒定载荷
    zero_drift ：每通道线性零点漂移，合计 +600 ADC / 60 min（与掌心同量级）
    noise      ：σ = 3 ADC/通道 白噪声
    前 60 s    ：0 → 满载荷线性加载
真值恒为 30000 ADC，理想输出 = 30000 一条直线。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ADC_PER_N = 1000.0
FPS = 100.0
DUR = 3600.0
N_CH = 57
LOAD_ADC = 30000.0
ZERO_DRIFT_ADC = 600.0
NOISE_SIGMA = 3.0
RAMP_S = 60.0
SEED = 20261019
GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]


def make_stream():
    rng = np.random.default_rng(SEED)
    n = int(DUR * FPS)
    t = np.arange(n) / FPS
    base = np.full(N_CH, LOAD_ADC / N_CH)
    drift_vec = np.full(N_CH, ZERO_DRIFT_ADC / N_CH)
    ramp = np.clip(t / RAMP_S, 0.0, 1.0)[:, None]
    zero = drift_vec[None, :] * (t / DUR)[:, None]
    noise = rng.normal(0.0, NOISE_SIGMA, size=(n, N_CH))
    pre = base[None, :] * ramp + zero + noise
    return t, pre


def to_1s(t, *arrs):
    sec = np.floor(t).astype(np.int64)
    idx = np.unique(sec)
    pos = np.searchsorted(sec, idx)
    return [idx.astype(np.float64)] + [a[pos] for a in arrs]


def main():
    import sys
    sys.path.insert(0, str(HERE.parent / "05_阶梯上飘诊断_20261019"))
    import core

    base_root = Path(r"D:\Temp\拇指指腹-2026-10-19_17-22-01")
    raw = json.loads((base_root / "session.json").read_text(encoding="utf-8"))["algorithm"]["params"]
    base = {k: float(v) for k, v in raw.items() if k in core.BASE}

    t, pre = make_stream()
    print(f"synth: {pre.shape[0]} frames x {pre.shape[1]} ch, 真值恒定 {LOAD_ADC:.0f} ADC, "
          f"零点漂移 +{ZERO_DRIFT_ADC:.0f} ADC/60min, 噪声 σ={NOISE_SIGMA}")

    rows = {}
    curves = {}
    for v in GRID:
        p = dict(base)
        p["r_slow_max"] = float(v)
        p["floor_rise_gate"] = 1
        out = core.run(pre, t, p, log=False)
        t1, pre1, cur1 = to_1s(t, pre.sum(1), out.sum(1))
        rel = cur1 - LOAD_ADC
        rows[f"{v:.2f}"] = dict(
            out_end=float(cur1[-1]), out_mean_tail=float(cur1[t1 >= DUR - 600].mean()),
            err_end=float(cur1[-1] - LOAD_ADC),
            err_min=float(rel.min()), err_max=float(rel.max()),
            err_p2p=float(rel.max() - rel.min()),
            sink=float(max(0.0, -rel.min())), rise=float(max(0.0, rel.max())),
        )
        curves[f"{v:.2f}"] = cur1
        print(f"  r_slow_max={v:.2f}  end_err={rows[f'{v:.2f}']['err_end']:+8.1f} ADC  "
              f"err_range=[{rows[f'{v:.2f}']['err_min']:+8.1f},{rows[f'{v:.2f}']['err_max']:+8.1f}]  "
              f"sink={rows[f'{v:.2f}']['sink']:7.1f}", flush=True)

    np.savez_compressed(OUT / "synth_hold.npz", t=t1, pre=pre1,
                        truth=np.full_like(t1, LOAD_ADC),
                        **{f"out_{k}": c for k, c in curves.items()})
    (OUT / "synth_hold.json").write_text(json.dumps(dict(
        config=dict(dur_s=DUR, fps=FPS, channels=N_CH, load_adc=LOAD_ADC,
                    zero_drift_adc=ZERO_DRIFT_ADC, noise_sigma=NOISE_SIGMA,
                    ramp_s=RAMP_S, seed=SEED),
        results=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", OUT / "synth_hold.json")


if __name__ == "__main__":
    main()
