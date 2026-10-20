"""新旧 DLL 逐帧等价性 + C/Python 分叉定位。"""
import ctypes
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

PKG = Path(__file__).resolve().parents[1] / "03_K10_算法迁移包_20261014"
ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")

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


def bind(path, struct_cls):
    d = ctypes.CDLL(str(path))
    d.dv6_portable_new.restype = ctypes.c_void_p
    d.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(struct_cls), ctypes.c_int]
    d.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    d.dv6_portable_free.argtypes = [ctypes.c_void_p]
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


ts, pre = load_cells("device_001_pre_seg0.csv")
t = ts - ts[0]
sj = json.loads((ROOT / "session.json").read_text(encoding="utf-8"))
vals = dict(core.BASE)
vals.update({k: v for k, v in sj["algorithm"]["params"].items() if k in core.BASE})

old = bind(PKG / "dv6_portable_k9_20261014.dll.bak", P_old)
new = bind(PKG / "dv6_portable.dll", P_new)

for guard in (0, 1):
    a = run_dll(old, pre, ts, fill(P_old, vals, guard))
    b = run_dll(new, pre, ts, fill(P_new, vals, guard, gate=0))
    d = np.abs(a - b)
    print(f"guard={guard} gate=0: 旧DLL vs 新DLL 最大差={d.max():.6e} 平均={d.mean():.6e} "
          f"等价={d.max() == 0.0}")

# C(gate=1) vs Python(core.py gate=1) 全量
c_gate = run_dll(new, pre, ts, fill(P_new, vals, 0, gate=1))
p_gate = core.run(pre, ts, {**vals, "floor_rise_gate": 1})
d = np.abs(c_gate - p_gate)
print(f"\ngate=1: C vs Python 最大差={d.max():.6e} 平均={d.mean():.6e}")
i = int(np.argmax(d.max(1)))
print(f"  最大差帧 #{i} t={t[i]:.3f}s 通道={int(np.argmax(d[i]))}")
mt = (t >= 1100) & (t <= 2850)
for nm, arr in (("C gate=1", c_gate.sum(1)), ("Py gate=1", p_gate.sum(1))):
    print(f"  {nm}: 恒载段漂移={arr[mt][-1]-arr[mt][0]:+9.1f}  末值={arr[mt][-1]:9.1f}")

# C(gate=0) vs Python(gate=0) 分叉定位
c0 = run_dll(new, pre, ts, fill(P_new, vals, 0, gate=0))
p0 = core.run(pre, ts, {**vals, "floor_rise_gate": 0})
d0 = np.abs(c0 - p0)
i0 = int(np.argmax(d0.max(1) > 1e-9))
print(f"\ngate=0: C vs Python 首个差>1e-9 的帧 #{i0} t={t[i0]:.4f}s 最大差={d0.max():.6e}")
print("  前后帧 pre:", pre[i0 - 1][:6], pre[i0][:6])
print("  C :", c0[i0][:6])
print("  Py:", p0[i0][:6])
