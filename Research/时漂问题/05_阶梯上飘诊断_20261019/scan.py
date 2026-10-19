"""扫描 2026-10-19 拇指指腹会话，量化"阶梯状上飘"。

输出：分段时间序列统计 + 关键通道 trace，供定位与复现。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    path = ROOT / name
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.startswith("##Data"):
                break
        header_lines += 1  # 列名行
    df = pd.read_csv(path, skiprows=header_lines, header=None, engine="c")
    return df


def main():
    names = {
        "raw": "device_001_raw_seg000.csv",
        "pre": "device_001_pre_seg0.csv",
        "algo": "device_001_seg000.csv",
        "algoadc": "device_001_algoadc_seg0.csv",
        "ft": "device_001_forcetotal_seg0.csv",
    }
    data = {}
    for k, v in names.items():
        df = load(v)
        t = df.iloc[:, 1].to_numpy(dtype=np.float64)
        cells = df.iloc[:, 3:].to_numpy(dtype=np.float64)
        data[k] = (t, cells)
        print(f"{k:8s} frames={len(t):7d} cells={cells.shape[1]:3d} "
              f"t=[{t[0]:.2f},{t[-1]:.2f}]s mean_cell_sum={cells.sum(1).mean():.2f}", flush=True)

    t = data["algo"][0]
    raw, pre, algo = data["raw"][1], data["pre"][1], data["algo"][1]

    # 每秒聚合：raw 总和 / pre 总和 / algo 总和
    sec = np.floor(t).astype(np.int64)
    nsec = sec.max() + 1
    def by_sec(x):
        s = np.bincount(sec, weights=x, minlength=nsec)
        c = np.bincount(sec, minlength=nsec)
        return np.divide(s, np.maximum(c, 1))
    ts_raw = by_sec(raw.sum(1))
    ts_pre = by_sec(pre.sum(1))
    ts_algo = by_sec(algo.sum(1))
    applied = ts_pre - ts_algo
    np.savez_compressed(OUT / "sec.npz", ts_raw=ts_raw, ts_pre=ts_pre, ts_algo=ts_algo,
                        applied=applied, t=t, raw_sum=raw.sum(1), pre_sum=pre.sum(1),
                        algo_sum=algo.sum(1))

    print("\n每分钟：raw总和  pre总和  algo总和  applied(=pre-algo)")
    for m in range(0, nsec // 60 + 1):
        a, b = m * 60, min((m + 1) * 60, nsec)
        if a >= nsec:
            break
        print(f"  m{m:02d}  raw={ts_raw[a:b].mean():10.1f}  pre={ts_pre[a:b].mean():10.1f}  "
              f"algo={ts_algo[a:b].mean():10.1f}  applied={applied[a:b].mean():9.1f}  "
              f"algo_delta={ts_algo[b-1]-ts_algo[a]:+9.1f}")

    # 15~36 分钟逐 10s 明细
    print("\n15~36min 每 10s（sec_idx, raw, pre, algo, applied）")
    for s in range(15 * 60, min(36 * 60, nsec), 10):
        print(f"  {s:5d} {s/60:6.2f}min raw={ts_raw[s]:10.1f} pre={ts_pre[s]:10.1f} "
              f"algo={ts_algo[s]:10.1f} applied={applied[s]:9.1f}")

    # 逐通道 applied 变化：找上飘贡献最大的通道
    ap_cell = pre - algo  # frames x cells
    seg_a, seg_b = 15 * 60, 36 * 60
    m = (t >= seg_a) & (t < seg_b)
    d = ap_cell[m][-1] - ap_cell[m][0]
    order = np.argsort(d)[:12]
    print("\n18~33min 窗口内 applied 下降最多的通道（上飘来源）：")
    for k in order:
        print(f"  cell{k:2d}: applied {ap_cell[m][0][k]:8.1f} -> {ap_cell[m][-1][k]:8.1f}  "
              f"delta={d[k]:+8.1f}  raw均值={raw[m][:,k].mean():8.1f}")

    np.save(OUT / "ap_cell_window_delta.npy", d)


if __name__ == "__main__":
    main()
