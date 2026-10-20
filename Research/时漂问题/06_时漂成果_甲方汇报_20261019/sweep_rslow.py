"""r_slow_max 扫描：修复档（floor_rise_idle_only=1）下慢态上限对力值下漂的影响。

只读原始录制、只写 out/sweep_*，不触碰 metrics.json 与已交付 PPT。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "05_阶梯上飘诊断_20261019"))
import core  # noqa: E402

OUT = HERE / "out"
ADC_PER_N = 1000.0

GROUPS = {
    "thumb": dict(root=Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49"), win=(1100.0, 2850.0)),
    "palm": dict(root=Path(r"D:\Temp\拇指指腹-2026-10-19_17-22-01"), win=(600.0, 5700.0)),
}
GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]


def load_cells(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(np.float64), df.iloc[:, 3:].to_numpy(np.float64)


def session_params(root: Path):
    raw = json.loads((root / "session.json").read_text(encoding="utf-8"))["algorithm"]["params"]
    return {k: float(v) for k, v in raw.items() if k in core.BASE}


def to_1s(t, *arrs):
    sec = np.floor(t).astype(np.int64)
    idx = np.unique(sec)
    pos = np.searchsorted(sec, idx)
    return [idx.astype(np.float64)] + [a[pos] for a in arrs]


def metrics(t1, pre1, cur1, win):
    m = (t1 >= win[0]) & (t1 <= win[1])
    tw, pw, cw = t1[m], pre1[m], cur1[m]
    n_head = max(1, len(cw) // 60)
    head = cw[:n_head].mean()
    tail = cw[-n_head:].mean()
    rel = cw - head
    ap = pw - cw
    last = tw >= (tw[-1] - 600.0)
    slope = float(np.polyfit(tw[last], cw[last], 1)[0] * 60.0)
    return dict(
        out_head=float(head), out_tail=float(tail), out_drift=float(tail - head),
        out_p2p=float(cw.max() - cw.min()),
        rise=float(max(0.0, rel.max())), sink=float(max(0.0, -rel.min())),
        applied_head=float(ap[0]), applied_tail=float(ap[-1]),
        applied_min=float(ap.min()), applied_max=float(ap.max()),
        slope_tail_adc_min=slope,
        steps=int((np.diff(ap) < -30.0).sum()),
    )


def run_group(key, grid):
    g = GROUPS[key]
    root = g["root"]
    print(f"[{key}] load", flush=True)
    ts, pre = load_cells(root / "device_001_pre_seg0.csv")
    base = session_params(root)
    base["floor_rise_gate"] = 1
    rows, curves = {}, {}
    for v in grid:
        p = dict(base)
        p["r_slow_max"] = float(v)
        t0 = time.time()
        out = core.run(pre, ts, p, log=False)
        t1, pre1, cur1 = to_1s(ts - ts[0], pre.sum(1), out.sum(1))
        rows[f"{v:.2f}"] = metrics(t1, pre1, cur1, g["win"])
        curves[f"{v:.2f}"] = cur1
        print(f"[{key}] r_slow_max={v:.2f}  drift={rows[f'{v:.2f}']['out_drift']:+.1f} "
              f"p2p={rows[f'{v:.2f}']['out_p2p']:.1f} sink={rows[f'{v:.2f}']['sink']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del out
    np.savez_compressed(OUT / f"sweep_rslow_{key}.npz", t=t1, pre=pre1,
                        **{f"out_{k}": c for k, c in curves.items()})
    return rows


def main():
    keys = sys.argv[1:] or ["thumb", "palm"]
    mpath = OUT / "sweep_rslow.json"
    res = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for k in keys:
        res[k] = run_group(k, GRID)
        mpath.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, rows in res.items():
        print(f"\n== {k}")
        print(f"{'r_slow_max':>10} {'drift':>9} {'p2p':>8} {'rise':>8} {'sink':>8} "
              f"{'ap_head':>9} {'ap_tail':>9} {'slope':>8}")
        for v, r in rows.items():
            print(f"{v:>10} {r['out_drift']:>+9.1f} {r['out_p2p']:>8.1f} {r['rise']:>8.1f} "
                  f"{r['sink']:>8.1f} {r['applied_head']:>9.0f} {r['applied_tail']:>9.0f} "
                  f"{r['slope_tail_adc_min']:>+8.2f}")


if __name__ == "__main__":
    main()
