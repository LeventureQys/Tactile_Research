"""权威实现对拍：portable-1.1 的 C ABI（dv6_portable.dll）在两份实录上跑两档，
与合成到 out/series_*.npz 的曲线指标比对。

用途（这是"K10 修复"的验收动作，不是 PPT 素材）：
  1) DLL 复刻现场档（gate=0，guard=录制值）与录制里的 algoadc 逐帧对拍 —— 证明现场档可被复刻；
  2) DLL 修复档（gate=1）的总力指标与 Python 参考实现（core.py）比对 —— 证明修复结论不是实现差异。
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "03_K10_算法迁移包_20261014"
OUT = HERE / "out"
ADC_PER_N = 1000.0

PARAM_FIELDS = [
    "r_fast", "tau_c_fast_s", "tau_r_fast_s", "r_slow_max", "tau_r_slow_s",
    "slope_cap_frac", "slope_gate_frac", "tau_zero_s", "idle_frac", "y_max_tau_s",
    "tau_slope_s", "edge_slope_thres", "edge_refract_s", "edge_boost_s",
    "tau_c_fast_boost_s", "soft_unfreeze_s", "hold_eps", "hold_tau_s",
    "slow_confirm_s", "y_floor_tau_s", "tau_r_slow_idle_s", "bypass_release_frac",
    "bypass_engage_frac", "bypass_noise_sigma", "bypass_base_tau_s",
    "ramp_slope_min", "ramp_full_s",
]

GROUPS = {
    "thumb": dict(root=Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49"), win=(1100.0, 2850.0)),
    "palm": dict(root=Path(r"D:\Temp\拇指指腹-2026-10-19_17-22-01"), win=(600.0, 5700.0)),
}


class Dv6Params(ctypes.Structure):
    """必须与 creep_observer_portable.h 的 dv6_params 逐字段一致（含 1.1/1.2 新增项）。"""

    _fields_ = (
        [(name, ctypes.c_double) for name in PARAM_FIELDS]
        + [("guard_enable", ctypes.c_int),
           ("guard_floor_confirm_s", ctypes.c_double),
           ("guard_idle_confirm_s", ctypes.c_double),
           ("guard_floor_deep", ctypes.c_double),
           ("floor_rise_idle_only", ctypes.c_int),
           ("guard_floor_recover_s", ctypes.c_double)]
    )


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


def make_params(root: Path, gate: int, guard: int) -> Dv6Params:
    raw = json.loads((root / "session.json").read_text(encoding="utf-8"))["algorithm"]["params"]
    p = Dv6Params()
    for k in PARAM_FIELDS:
        setattr(p, k, float(raw[k]))
    p.guard_enable = int(guard)
    p.guard_floor_confirm_s = float(raw.get("guard_floor_confirm_s", 0.5))
    p.guard_idle_confirm_s = float(raw.get("guard_idle_confirm_s", 0.3))
    p.guard_floor_deep = float(raw.get("guard_floor_deep", 200.0))
    p.floor_rise_idle_only = int(gate)
    p.guard_floor_recover_s = 0.0
    return p


def run_dll(dll, values, t_rel, params):
    n_ch = values.shape[1]
    handle = dll.dv6_portable_new(n_ch, ctypes.byref(params), 2)
    if not handle:
        raise SystemExit("dv6_portable_new failed")
    produced = np.empty_like(values)
    buf = (ctypes.c_double * n_ch)()
    view = np.ctypeslib.as_array(buf)
    for i in range(values.shape[0]):
        view[:] = values[i]
        dll.dv6_portable_process(handle, float(t_rel[i]), buf, n_ch)
        produced[i] = view
    dll.dv6_portable_free(handle)
    return produced


def window_metrics(t1, pre1, cur1, win):
    m = (t1 >= win[0]) & (t1 <= win[1])
    tw, pre_w, cur_w = t1[m], pre1[m], cur1[m]
    head = cur_w[: max(1, len(cur_w) // 60)].mean()
    tail = cur_w[-max(1, len(cur_w) // 60):].mean()
    steps = int((np.diff(pre_w - cur_w) < -30.0).sum())
    return dict(out_drift=float(tail - head), out_p2p=float(cur_w.max() - cur_w.min()),
                steps=steps)


def to_1s(t, *arrs):
    sec = np.floor(t).astype(np.int64)
    idx = np.unique(sec)
    pos = np.searchsorted(sec, idx)
    return [idx.astype(np.float64)] + [a[pos] for a in arrs]


def main():
    keys = sys.argv[1:] or ["thumb", "palm"]
    dll = ctypes.CDLL(str(PKG / "dv6_portable.dll"))
    dll.dv6_portable_new.restype = ctypes.c_void_p
    dll.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(Dv6Params), ctypes.c_int]
    dll.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                         ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    dll.dv6_portable_free.argtypes = [ctypes.c_void_p]
    dll.dv6_portable_params_bytes.restype = ctypes.c_int
    c_bytes = int(dll.dv6_portable_params_bytes())
    py_bytes = int(ctypes.sizeof(Dv6Params))
    print(f"params bytes: dll={c_bytes} ctypes={py_bytes}")
    if c_bytes != py_bytes:
        raise SystemExit("dv6_params 结构体失配，先核对字段。")

    report = {}
    for key in keys:
        g = GROUPS[key]
        root = g["root"]
        ts, pre = load_cells(root / "device_001_pre_seg0.csv")
        _, algoadc = load_cells(root / "device_001_algoadc_seg0.csv")
        t_rel = ts - ts[0]
        raw = json.loads((root / "session.json").read_text(encoding="utf-8"))["algorithm"]["params"]
        site_guard = int(raw.get("guard_enable", 0))

        res = {}
        for tag, gate, guard in (("SITE", 0, site_guard), ("FIX", 1, 0), ("FIXG", 1, 1)):
            p = make_params(root, gate, guard)
            t0 = time.time()
            v = run_dll(dll, pre, t_rel, p)
            err = np.abs(v - algoadc)
            t1, pre1, meas1, v1 = to_1s(t_rel, pre.sum(1), algoadc.sum(1), v.sum(1))
            # Python 参考实现的秒级曲线
            py = np.load(OUT / f"series_{key}.npz")
            pykey = "MEAS" if tag == "SITE" else tag
            py1 = np.interp(t1, py["t"], py[f"out_{pykey}"])
            res[tag] = dict(
                dll_metrics=window_metrics(t1, pre1, v1, g["win"]),
                vs_algoadc_max=float(err.max()),
                vs_algoadc_mean=float(err.mean()),
                vs_algoadc_max_in_window=float(err[t_rel <= g["win"][1]].max()),
                py_curve_max_abs_diff=float(np.abs(v1 - py1).max()),
                run_s=round(time.time() - t0, 1),
            )
            print(f"[{key}] {tag}: dll drift={res[tag]['dll_metrics']['out_drift']:.1f} "
                  f"p2p={res[tag]['dll_metrics']['out_p2p']:.1f} steps={res[tag]['dll_metrics']['steps']} | "
                  f"vs algoadc max={err.max():.3f} mean={err.mean():.4f} | "
                  f"vs py total max={res[tag]['py_curve_max_abs_diff']:.3f}", flush=True)
            del v
        report[key] = res

    (OUT / "dll_parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("->", OUT / "dll_parity.json")


if __name__ == "__main__":
    main()
