"""回归 v2：逐帧差（改动是否触及历史工况）+ 加载后恒载段漂移 + 10g 事件画像。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

DATA = Path(__file__).resolve().parents[1] / "data" / "archived"
OUT = Path(__file__).resolve().parent / "out"


def read_csv_cells(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


def load_params(d):
    params = {}
    sj = d / "session.json"
    if sj.exists():
        try:
            ap = json.loads(sj.read_text(encoding="utf-8")).get("algorithm", {}).get("params", {})
            for k in core.BASE:
                if k in ap:
                    params[k] = ap[k]
        except Exception:
            pass
    return params


def hold_window(raw, t):
    """返回加载完成后进入恒载的起点索引（总和首次达到峰值 80% 后 30s）。"""
    peak = raw.max()
    base = np.median(raw[: max(5, len(raw) // 50)])
    if peak - base < 50:
        return None
    above = np.where(raw > base + 0.8 * (peak - base))[0]
    if not len(above):
        return None
    i = above[0]
    j = np.searchsorted(t, t[i] + 30.0)
    return j if j < len(t) - 100 else None


def main():
    rows = []
    for pre_path in sorted(DATA.rglob("device_001_pre_seg0.csv")):
        d = pre_path.parent
        name = str(d.relative_to(DATA))
        ts, pre = read_csv_cells(pre_path)
        t = ts - ts[0]
        p = load_params(d)
        raw = pre.sum(1)
        v9 = core.run(pre, ts, dict(p))
        vg = core.run(pre, ts, {**p, "floor_rise_gate": 1})
        dframe = np.abs(vg - v9)
        s9, sg = v9.sum(1), vg.sum(1)
        h = hold_window(raw, t)
        row = dict(session=name, frames=len(t), dur_s=round(float(t[-1]), 1),
                   max_abs_frame_diff=float(dframe.max()),
                   mean_abs_frame_diff=float(dframe.mean()),
                   sum_abs_diff_max=float(np.abs(sg - s9).max()),
                   changed_frames_gt1=int((dframe.max(1) > 1.0).sum()))
        if h is not None:
            row["hold_start_s"] = round(float(t[h]), 1)
            row["hold_dur_s"] = round(float(t[-1] - t[h]), 1)
            row["hold_pre_drift"] = float(raw[-1] - raw[h])
            row["hold_k9_drift"] = float(s9[-1] - s9[h])
            row["hold_gate_drift"] = float(sg[-1] - sg[h])
            row["hold_k9_p2p"] = float(s9[h:].max() - s9[h:].min())
            row["hold_gate_p2p"] = float(sg[h:].max() - sg[h:].min())
        rows.append(row)
        print(f"  {name[:52]:52s} f={len(t):6d} dur={t[-1]:7.0f}s "
              f"逐帧max差={row['max_abs_frame_diff']:7.3f} "
              f"总和max差={row['sum_abs_diff_max']:7.2f} "
              f">1ADC帧={row['changed_frames_gt1']:5d}"
              + (f" 恒载{row['hold_dur_s']:6.0f}s 漂移 K9={row['hold_k9_drift']:+8.1f} "
                 f"GATE={row['hold_gate_drift']:+8.1f}（输入{row['hold_pre_drift']:+8.1f}）"
                 if h is not None else ""), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regress_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\n全部 {len(df)} 会话：逐帧|Δ| 最大={df['max_abs_frame_diff'].max():.4f} ADC")
    print(f"已写 {OUT/'regress_v2.csv'}")


if __name__ == "__main__":
    main()
