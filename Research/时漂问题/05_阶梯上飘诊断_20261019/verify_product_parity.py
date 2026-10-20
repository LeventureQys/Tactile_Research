"""产品源码 creep_observer.cpp vs portable 权威实现：48min 实录逐帧对拍。

对照四臂：
  prod  gate=0 / gate=1        ← 产品 src/domain/drift_v6/creep_observer.cpp（经 harness.exe）
  port  gate=0 / gate=1        ← 03_K10_算法迁移包/creep_observer_portable.h（经 DLL）
另取旧 DLL（portable-1.0）作基线，确认 gate=0 时产品未偏离历史行为。
"""
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core

HARNESS_DIR = Path(__file__).resolve().parent / "product_harness"
HARNESS = HARNESS_DIR / "bin" / "product_harness.exe"
WORK = Path(r"D:\Temp\k10fix_check")
PKG = Path(__file__).resolve().parents[1] / "03_K10_算法迁移包_20261014"
ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"

FIELDS = ["r_fast", "tau_c_fast_s", "tau_r_fast_s", "r_slow_max", "tau_r_slow_s",
          "slope_cap_frac", "slope_gate_frac", "tau_zero_s", "idle_frac", "y_max_tau_s",
          "tau_slope_s", "edge_slope_thres", "edge_refract_s", "edge_boost_s",
          "tau_c_fast_boost_s", "soft_unfreeze_s", "hold_eps", "hold_tau_s",
          "slow_confirm_s", "y_floor_tau_s", "tau_r_slow_idle_s", "bypass_release_frac",
          "bypass_engage_frac", "bypass_noise_sigma", "bypass_base_tau_s",
          "ramp_slope_min", "ramp_full_s"]


class P_old(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_double) for n in FIELDS]
                + [("guard_enable", ctypes.c_int),
                   ("guard_floor_confirm_s", ctypes.c_double),
                   ("guard_idle_confirm_s", ctypes.c_double),
                   ("guard_floor_deep", ctypes.c_double)])


class P_new(ctypes.Structure):
    _fields_ = list(P_old._fields_) + [("floor_rise_idle_only", ctypes.c_int)]


def bind(path, cls):
    d = ctypes.CDLL(str(path))
    d.dv6_portable_new.restype = ctypes.c_void_p
    d.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(cls), ctypes.c_int]
    d.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    d.dv6_portable_free.argtypes = [ctypes.c_void_p]
    d.dv6_portable_params_bytes.restype = ctypes.c_int
    return d


def fill(cls, vals, guard, gate=None):
    p = cls()
    for k in FIELDS:
        setattr(p, k, float(vals[k]))
    p.guard_enable = guard
    p.guard_floor_confirm_s = 0.5
    p.guard_idle_confirm_s = 0.3
    p.guard_floor_deep = 200.0
    if gate is not None:
        p.floor_rise_idle_only = gate
    return p


def run_dll(dll, values, t, p):
    n = values.shape[1]
    h = dll.dv6_portable_new(n, ctypes.byref(p), 2)
    buf = (ctypes.c_double * n)()
    out = np.empty_like(values)
    for i in range(values.shape[0]):
        for k in range(n):
            buf[k] = float(values[i, k])
        dll.dv6_portable_process(h, float(t[i]), buf, n)
        for k in range(n):
            out[i, k] = buf[k]
    dll.dv6_portable_free(h)
    return out


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


def main():
    if not HARNESS.exists():
        raise SystemExit(f"缺少脚手架 {HARNESS}；先运行 product_harness\\build_harness.bat")
    ts, pre = load_cells("device_001_pre_seg0.csv")
    t = ts - ts[0]
    n = pre.shape[1]
    WORK.mkdir(parents=True, exist_ok=True)

    inp = WORK / "frames.bin"
    packed = np.empty((pre.shape[0], n + 1), dtype=np.float64)
    packed[:, 0] = ts
    packed[:, 1:] = pre
    packed.tofile(inp)
    print(f"输入打包 {inp}  {packed.shape}  {inp.stat().st_size/1e6:.1f} MB")

    def run_prod(gate, guard, tag):
        out = WORK / f"prod_{tag}.bin"
        r = subprocess.run([str(HARNESS), str(inp), str(out), str(n), str(gate), str(guard), "1"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"harness {tag} failed: {r.stdout} {r.stderr}")
        print("  " + r.stderr.strip())
        return np.fromfile(out, dtype=np.float64).reshape(-1, n)

    prods = {"g0": run_prod(0, 0, "g0"), "g1": run_prod(1, 0, "g1"), "k10": run_prod(1, 1, "k10")}

    new = bind(PKG / "dv6_portable.dll", P_new)
    old = bind(PKG / "dv6_portable_k9_20261014.dll.bak", P_old)
    print(f"  sizeof(dv6_params): portable-1.1={new.dv6_portable_params_bytes()} "
          f"ctypes={ctypes.sizeof(P_new)}  portable-1.0={old.dv6_portable_params_bytes()} "
          f"ctypes={ctypes.sizeof(P_old)}")

    sj = json.loads((ROOT / "session.json").read_text(encoding="utf-8"))
    vals = dict(core.BASE)
    vals.update({k: v for k, v in sj["algorithm"]["params"].items() if k in core.BASE})

    ports = {
        "g0": run_dll(new, pre, ts, fill(P_new, vals, 0, gate=0)),
        "g1": run_dll(new, pre, ts, fill(P_new, vals, 0, gate=1)),
        "k10": run_dll(new, pre, ts, fill(P_new, vals, 1, gate=1)),
        "old0": run_dll(old, pre, ts, fill(P_old, vals, 0)),
    }

    print("\n=== 逐帧对拍 ===")
    pairs = [("prod gate=0", prods["g0"], "portable-1.0 gate=0", ports["old0"]),
             ("prod gate=0", prods["g0"], "portable-1.1 gate=0", ports["g0"]),
             ("prod gate=1", prods["g1"], "portable-1.1 gate=1", ports["g1"]),
             ("prod gate=1+guard", prods["k10"], "portable-1.1 gate=1+guard", ports["k10"])]
    report = {}
    for an, a, bn, b in pairs:
        d = np.abs(a - b)
        key = f"{an} vs {bn}"
        report[key] = dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()))
        print(f"  {an:18s} vs {bn:26s} max|Δ|={d.max():.6e}  mean={d.mean():.6e}")

    mt = (t >= 1100) & (t <= 2850)
    print("\n=== 恒载段漂移（1100–2850 s，ADC 总和）===")
    for name, arr in (("prod gate=0", prods["g0"]), ("prod gate=1", prods["g1"]),
                      ("prod gate=1+guard", prods["k10"])):
        s = arr.sum(1)
        ap = (pre.sum(1) - s)
        report[f"drift {name}"] = float(s[mt][-1] - s[mt][0])
        print(f"  {name:20s} 漂移={s[mt][-1]-s[mt][0]:+9.1f}  applied 末={ap[mt][-1]:8.1f} "
              f"applied 最小={ap[mt].min():8.1f}")

    (OUT / "product_parity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\n已写 {OUT/'product_parity.json'}")


if __name__ == "__main__":
    main()
