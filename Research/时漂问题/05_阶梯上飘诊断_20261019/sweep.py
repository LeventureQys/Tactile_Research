"""在 2026-10-19 拇指指腹会话上扫参数化硬化方案，量化"阶梯上飘"是否消失。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load_cells(name):
    path = ROOT / name
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


VARIANTS = {
    "K9 现役（本次录制实跑）": {},
    "K10 防护（guard=1）": {"guard_enable": 1},
    "+idle 绝对地板 100": {"idle_band_floor_adc": 100.0},
    "+idle 绝对地板 200": {"idle_band_floor_adc": 200.0},
    "+idle 绝对地板 300": {"idle_band_floor_adc": 300.0},
    "+band200 +zero 受载门 2%": {"idle_band_floor_adc": 200.0, "zero_track_gate": 0.02},
    "+band200 +zero 速率上限 5": {"idle_band_floor_adc": 200.0, "zero_rate_cap_adc_s": 5.0},
    "+band200 +关掉快泄放": {"idle_band_floor_adc": 200.0, "tau_r_slow_idle_s": 0.0},
    "K10+band200": {"guard_enable": 1, "idle_band_floor_adc": 200.0},
    "+band200 +zero门2% +快泄放确认(K10)": {"idle_band_floor_adc": 200.0, "zero_track_gate": 0.02,
                                            "guard_enable": 1},
}


def metrics(t, pre, alg, lo=1100.0, hi=2850.0):
    m = (t >= lo) & (t <= hi)
    a = alg.sum(1)[m]
    ts = t[m]
    d = a[-1] - a[0]
    # 逐秒 applied 台阶数（单秒跌幅 >30 ADC）
    sec = np.floor(ts).astype(np.int64)
    nsec = sec.max() - sec.min() + 1
    s0 = sec - sec.min()
    cnt = np.bincount(s0, minlength=nsec)
    ap = np.bincount(s0, weights=(pre.sum(1)[m] - a), minlength=nsec) / np.maximum(cnt, 1)
    dap = np.diff(ap)
    steps = int((dap < -30).sum())
    return dict(first=float(a[0]), last=float(a[-1]), drift=d,
                p2p=float(a.max() - a.min()), steps=steps,
                applied_start=float(ap[0]), applied_end=float(ap[-1]),
                applied_min=float(ap.min()))


def main():
    ts, pre = load_cells("device_001_pre_seg0.csv")
    t = ts - ts[0]
    rows = {}
    print(f"{'变体':34s} {'恒载首尾漂移':>12s} {'p2p':>9s} {'台阶数':>6s} {'补偿首/末':>18s}")
    for name, over in VARIANTS.items():
        v = core.run(pre, ts, over)
        mt = metrics(t, pre, v)
        rows[name] = mt
        print(f"{name:34s} {mt['drift']:+12.1f} {mt['p2p']:9.1f} {mt['steps']:6d} "
              f"{mt['applied_start']:9.1f}/{mt['applied_end']:8.1f}")
    (OUT / "sweep_20261019.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {OUT/'sweep_20261019.json'}")


if __name__ == "__main__":
    main()
