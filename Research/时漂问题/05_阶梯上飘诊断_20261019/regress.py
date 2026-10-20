"""回归：把 floor_rise_gate 放到全部历史会话上，检查漂移与 10g 事件保留率。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

DATA = Path(__file__).resolve().parents[1] / "data" / "archived"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


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


def resample(t, x, ds=0.05):
    """按 ds 秒分箱平均，返回 (ts, xs)。"""
    b = np.floor(t / ds).astype(np.int64)
    b -= b[0]
    n = b[-1] + 1
    s = np.bincount(b, weights=x, minlength=n)
    c = np.bincount(b, minlength=n)
    return np.arange(n) * ds + t[0], s / np.maximum(c, 1)


def detrend(t, x, win_s=120.0):
    ts, xs = resample(t, x)
    w = max(int(win_s / (ts[1] - ts[0])), 5)
    s = pd.Series(xs)
    return s - s.rolling(w, center=True, min_periods=1).median()


def peak_events(d, thr_abs, refractory_s, ds=0.05):
    a = np.abs(d)
    idx = []
    i, n = 1, len(a)
    gap = int(refractory_s / ds)
    while i < n - 1:
        if a[i] > thr_abs and a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            idx.append(i)
            i += gap
        else:
            i += 1
    return np.array(idx, dtype=np.int64)


def main():
    sessions = sorted(DATA.rglob("device_001_pre_seg0.csv"))
    rows = []
    for pre_path in sessions:
        d = pre_path.parent
        name = str(d.relative_to(DATA))
        ts, pre = read_csv_cells(pre_path)
        t = ts - ts[0]
        raw = pre.sum(1)
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
        out = {"session": name, "frames": len(t), "dur_s": round(float(t[-1]), 1),
               "r_slow_max": params.get("r_slow_max", core.BASE["r_slow_max"]),
               "guard_enable": params.get("guard_enable", 0),
               "pre_first": float(raw[0]), "pre_last": float(raw[-1]),
               "pre_drift": float(raw[-1] - raw[0])}
        runs = {"K9": dict(params),
                "K10": {**params, "guard_enable": 1},
                "GATE": {**params, "floor_rise_gate": 1}}
        sums = {}
        for tag, ov in runs.items():
            v = core.run(pre, ts, ov)
            s = v.sum(1)
            sums[tag] = s
            out[f"{tag}_drift"] = float(s[-1] - s[0])
            out[f"{tag}_applied_first"] = float(raw[0] - s[0])
            out[f"{tag}_applied_last"] = float(raw[-1] - s[-1])
            out[f"{tag}_p2p"] = float(s.max() - s.min())
            out[f"{tag}_detr_p2p"] = float(np.ptp(detrend(t, s).to_numpy()))
        if "10g" in name or "反复" in name:
            d_in = detrend(t, raw)
            thr = max(float(d_in.std()) * 2.5, 30.0)
            pk = peak_events(d_in.to_numpy(), thr, 1.0)
            out["n_events"] = int(len(pk))
            out["thr"] = round(thr, 1)
            if len(pk):
                amp_in = np.abs(d_in.to_numpy()[pk])
                for tag in ("K9", "K10", "GATE"):
                    d_out = detrend(t, sums[tag])
                    r = np.abs(d_out.to_numpy()[pk]) / amp_in
                    out[f"{tag}_ret_mean"] = float(np.mean(r))
                    out[f"{tag}_ret_median"] = float(np.median(r))
                    out[f"{tag}_ret_p10"] = float(np.percentile(r, 10))
                out["amp_in_mean"] = float(amp_in.mean())
        rows.append(out)
        print(f"  {name[:56]:56s} f={len(t):6d} pre={out['pre_drift']:+8.1f} "
              f"K9={out['K9_drift']:+8.1f} K10={out['K10_drift']:+8.1f} GATE={out['GATE_drift']:+8.1f}"
              f"  GATE-detr={out['GATE_detr_p2p']:7.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regress_all.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写 {OUT/'regress_all.csv'}")
    if (df["n_events"].fillna(0) > 0).any():
        print("\n10g 事件保留率：")
        print(df[df["n_events"].fillna(0) > 0][
            ["session", "n_events", "amp_in_mean", "K9_ret_median", "GATE_ret_median",
             "K9_ret_mean", "GATE_ret_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
