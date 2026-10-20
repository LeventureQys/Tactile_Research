"""两组长恒载实录上的 K10 目标修复（floor_rise_idle_only=1）重放与指标提取。

每组数据跑三臂，参数一律取自该次录制的 session.json（保证是"现场参数 + 单一开关"的对照）：

    SITE  现场现役档：gate=0，guard 取录制时实际值            —— 修复前
    FIX   目标修复档：gate=1，guard=0                          —— 修复后
    FIXG  目标修复 + 瞬态防护：gate=1，guard=1                 —— 修复后（防护同时开启）

SITE 臂与录制里的 device_001_algoadc_seg0.csv 逐帧对拍，用于证明复刻保真。

用法：
    python run_analysis.py            # 两组全跑
    python run_analysis.py thumb      # 只跑一组
输出：
    out/series_<group>.npz   秒级曲线（绘图用）
    out/metrics.json         全部指标
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
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    # 拇指指腹：48 min 恒载；窗口沿用 05 报告口径
    "thumb": dict(
        label="拇指指腹",
        root=Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49"),
        win=(1100.0, 2850.0),
    ),
    # 掌心：96.5 min 恒载；整段受载，窗口取稳态主体
    "palm": dict(
        label="掌心",
        root=Path(r"D:\Temp\拇指指腹-2026-10-19_17-22-01"),
        win=(600.0, 5700.0),
    ),
    # 四指指腹：74 min 恒载，补偿在 20–28 min 段逐级塌陷
    "third": dict(
        label="四指指腹",
        root=Path(r"E:\算法相关\拇指指腹-2026-10-19_18-58-56"),
        win=(600.0, 4350.0),
    ),
}

ARMS = [
    ("SITE", dict(floor_rise_gate=0, guard_enable=None)),
    ("FIX", dict(floor_rise_gate=1, guard_enable=0)),
    ("FIXG", dict(floor_rise_gate=1, guard_enable=1)),
]

# 修复档的慢态上限：现场录制的 0.6 在恒载下会把真实载荷当蠕变扣掉，表现为读数下探。
# 设为 None 表示沿用录制里的现场值（0.6）。
RSLOW_MAX_FIXED = 0.25


def load_cells(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    ts = df.iloc[:, 0].to_numpy(dtype=np.float64)
    cells = df.iloc[:, 3:].to_numpy(dtype=np.float64)
    return ts, cells


def session_params(root: Path):
    j = json.loads((root / "session.json").read_text(encoding="utf-8"))
    raw = j["algorithm"]["params"]
    p = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
         for k, v in raw.items() if k in core.BASE}
    meta = dict(
        session_id=j.get("session_id", ""),
        param_set=raw.get("param_set", ""),
        algorithm_id=j.get("algorithm", {}).get("id", ""),
        guard_enable=int(raw.get("guard_enable", 0)),
        r_slow_max=float(raw.get("r_slow_max", 0.1)),
        app_version=j.get("app_version", ""),
        frames=int(j["devices"][0]["frame_count"]),
    )
    return p, meta


def resample_1s(ts_abs: np.ndarray, *arrays: np.ndarray):
    el = ts_abs - ts_abs[0]
    sec = np.floor(el).astype(np.int64)
    idx = np.unique(sec)
    pos = np.searchsorted(sec, idx)
    out = [idx.astype(np.float64)]
    for a in arrays:
        out.append(a[pos])
    return out


def window_metrics(t: np.ndarray, pre: np.ndarray, cur: np.ndarray, win, label: str):
    m = (t >= win[0]) & (t <= win[1])
    tw, pre_w, cur_w = t[m], pre[m], cur[m]
    head = cur_w[: max(1, len(cur_w) // 60)].mean()
    tail = cur_w[-max(1, len(cur_w) // 60):].mean()
    pre_head = pre_w[: max(1, len(pre_w) // 60)].mean()
    pre_tail = pre_w[-max(1, len(pre_w) // 60):].mean()
    d = cur_w - head
    # 台阶：逐秒 applied 单步跌幅 > 30 ADC
    applied_w = pre_w - cur_w
    drop = np.diff(applied_w)
    steps = int((drop < -30.0).sum())
    # 稳态斜率（窗口末 10 min 线性拟合，ADC/min）
    k = float(tw[-1] - tw[0])
    tail_m = tw >= (tw[-1] - 600.0)
    slope = float(np.polyfit(tw[tail_m], cur_w[tail_m], 1)[0] * 60.0) if tail_m.sum() > 10 else float("nan")
    return dict(
        label=label,
        window=[float(win[0]), float(win[1])],
        out_head=float(head), out_tail=float(tail), out_drift=float(tail - head),
        out_p2p=float(cur_w.max() - cur_w.min()),
        out_min=float(cur_w.min()), out_max=float(cur_w.max()),
        out_std=float(d.std()),
        pre_head=float(pre_head), pre_tail=float(pre_tail), pre_drift=float(pre_tail - pre_head),
        pre_p2p=float(pre_w.max() - pre_w.min()),
        applied_head=float(applied_w[0]), applied_tail=float(applied_w[-1]),
        applied_min=float(applied_w.min()), applied_max=float(applied_w.max()),
        steps=steps,
        slope_tail_adc_min=slope,
        span_s=float(k),
    )


def run_group(key: str):
    g = GROUPS[key]
    root: Path = g["root"]
    t0 = time.time()
    print(f"[{key}] load {root.name}", flush=True)
    ts, pre = load_cells(root / "device_001_pre_seg0.csv")
    _, algoadc = load_cells(root / "device_001_algoadc_seg0.csv")
    el = ts - ts[0]
    print(f"[{key}] {pre.shape[0]} frames x {pre.shape[1]} ch, {el[-1]:.1f} s", flush=True)

    params, meta = session_params(root)
    site_guard = meta["guard_enable"]

    arms_out = {}
    arms_applied = {}
    parity = {}
    for name, ov in ARMS:
        p = dict(params)
        p["floor_rise_gate"] = ov["floor_rise_gate"]
        p["guard_enable"] = site_guard if ov["guard_enable"] is None else ov["guard_enable"]
        if ov["floor_rise_gate"] == 1 and RSLOW_MAX_FIXED is not None:
            p["r_slow_max"] = float(RSLOW_MAX_FIXED)
        t1 = time.time()
        v = core.run(pre, ts, p, log=False)
        run_s = time.time() - t1
        if name == "SITE":
            err = np.abs(v - algoadc)
            lim = g["win"][1]
            mid = el <= lim
            q = max(1, len(el) // 4)
            parity = dict(
                max_abs=float(err.max()),
                mean_abs=float(err.mean()),
                max_abs_in_window=float(err[mid].max()) if mid.any() else None,
                mean_abs_in_window=float(err[mid].mean()) if mid.any() else None,
                mean_abs_by_quarter=[float(err[i * q:(i + 1) * q].mean()) for i in range(4)],
                window_end_s=float(lim),
            )
            print(f"[{key}] SITE vs algoadc: max={parity['max_abs']:.3f} "
                  f"mean={parity['mean_abs']:.6f} | <= {lim:.0f}s: "
                  f"max={parity['max_abs_in_window']:.4f} mean={parity['mean_abs_in_window']:.6f}",
                  flush=True)
        arms_out[name] = v.sum(1)
        arms_applied[name] = (pre - v).sum(1)
        print(f"[{key}] arm {name} done in {run_s:.1f}s", flush=True)
        del v

    n_ch = int(pre.shape[1])
    pre_sum = pre.sum(1)
    meas_sum = algoadc.sum(1)  # 现场实测算法输出（"升级前"的权威曲线）
    del pre, algoadc

    t = el
    t1s, pre1s, *rest = resample_1s(ts, pre_sum, meas_sum,
                                    *[arms_out[k] for k, _ in ARMS],
                                    *[arms_applied[k] for k, _ in ARMS])
    keys = [k for k, _ in ARMS]
    series = dict(t=t1s, pre=pre1s, out_MEAS=rest[0],
                  **{f"out_{k}": rest[1 + i] for i, k in enumerate(keys)},
                  **{f"applied_{k}": rest[1 + len(keys) + i] for i, k in enumerate(keys)})

    win = g["win"]
    metrics = dict(
        key=key, label=g["label"], root=str(root), **meta,
        duration_s=float(el[-1]), channels=n_ch,
        r_slow_max_site=float(params["r_slow_max"]),
        r_slow_max_fix=(float(RSLOW_MAX_FIXED) if RSLOW_MAX_FIXED is not None
                        else float(params["r_slow_max"])),
        arms={**{k: window_metrics(t1s, series["pre"], series[f"out_{k}"], win, k)
                 for k in ["MEAS"] + keys}},
        parity_site_vs_algoadc=parity,
    )

    np.savez_compressed(OUT / f"series_{key}.npz", **series)
    print(f"[{key}] total {time.time() - t0:.1f}s -> out/series_{key}.npz", flush=True)
    return metrics


def main():
    which = sys.argv[1:] or [k for k, g in GROUPS.items() if g["root"] is not None]
    result = {}
    mpath = OUT / "metrics.json"
    if mpath.exists():
        result = json.loads(mpath.read_text(encoding="utf-8"))
    for k in which:
        if GROUPS[k]["root"] is None:
            continue
        result[k] = run_group(k)
        mpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {a: v["arms"][a]["out_drift"] for a in v["arms"]} for k, v in result.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
