"""验证 portable-1.1：
① floor_rise_idle_only=0 时与 portable-1.0 黄金向量逐帧一致（默认不改行为）；
② floor_rise_idle_only=1 时 C 实现与 core.py（Python 复刻）逐帧一致，且修掉阶梯上飘。
"""
import ctypes
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

PKG = Path(__file__).resolve().parents[1] / "03_K10_算法迁移包_20261014"
GOLD = PKG / "golden_vectors"
ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"

FIELDS = ["r_fast", "tau_c_fast_s", "tau_r_fast_s", "r_slow_max", "tau_r_slow_s",
          "slope_cap_frac", "slope_gate_frac", "tau_zero_s", "idle_frac", "y_max_tau_s",
          "tau_slope_s", "edge_slope_thres", "edge_refract_s", "edge_boost_s",
          "tau_c_fast_boost_s", "soft_unfreeze_s", "hold_eps", "hold_tau_s",
          "slow_confirm_s", "y_floor_tau_s", "tau_r_slow_idle_s", "bypass_release_frac",
          "bypass_engage_frac", "bypass_noise_sigma", "bypass_base_tau_s",
          "ramp_slope_min", "ramp_full_s"]


class Dv6Params(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_double) for n in FIELDS]
                + [("guard_enable", ctypes.c_int),
                   ("guard_floor_confirm_s", ctypes.c_double),
                   ("guard_idle_confirm_s", ctypes.c_double),
                   ("guard_floor_deep", ctypes.c_double),
                   ("floor_rise_idle_only", ctypes.c_int)])


def bind(dll):
    dll.dv6_portable_new.restype = ctypes.c_void_p
    dll.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(Dv6Params), ctypes.c_int]
    dll.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                         ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    dll.dv6_portable_free.argtypes = [ctypes.c_void_p]
    return dll


def make_params(vals, guard=0, gate=0):
    p = Dv6Params()
    for k in FIELDS:
        setattr(p, k, float(vals[k]))
    p.guard_enable = guard
    p.guard_floor_confirm_s = 0.5
    p.guard_idle_confirm_s = 0.3
    p.guard_floor_deep = 200.0
    p.floor_rise_idle_only = gate
    return p


def run_dll(dll, values, t, p):
    n = values.shape[1]
    h = dll.dv6_portable_new(n, ctypes.byref(p), 2)
    if not h:
        raise SystemExit("dv6_portable_new failed")
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
    dll = bind(ctypes.CDLL(str(PKG / "dv6_portable.dll")))
    gp = json.loads((GOLD / "golden_params.json").read_text(encoding="utf-8"))
    vals = gp["params"]

    print("=== ① 黄金向量（portable-1.0 期望输出）===")
    tin = np.loadtxt(GOLD / "golden_time_s.csv", skiprows=1, delimiter=",")
    xin = np.loadtxt(GOLD / "golden_input.csv", delimiter=",")
    yexp = np.loadtxt(GOLD / "golden_expected_output.csv", delimiter=",")
    y0 = run_dll(dll, xin, tin, make_params(vals, guard=0, gate=0))
    d0 = np.abs(y0 - yexp)
    print(f"  frames={xin.shape[0]} 通道={xin.shape[1]}")
    print(f"  gate=0 逐帧最大差={d0.max():.6e}  平均={d0.mean():.6e}  等价={d0.max() < 1e-6}")

    print("\n=== ② 本会话：C 实现 vs Python 复刻 ===")
    ts, pre = load_cells("device_001_pre_seg0.csv")
    t = ts - ts[0]
    sj = json.loads((ROOT / "session.json").read_text(encoding="utf-8"))
    live = {k: v for k, v in sj["algorithm"]["params"].items() if k in core.BASE}
    vals2 = dict(core.BASE)
    vals2.update(live)
    res = {}
    for gate, guard in ((0, 0), (0, 1), (1, 0)):
        yc = run_dll(dll, pre, ts, make_params(vals2, guard=guard, gate=gate))
        yp = core.run(pre, ts, {**live, "guard_enable": guard, "floor_rise_gate": gate})
        dd = np.abs(yc - yp)
        res[(guard, gate)] = yc
        print(f"  guard={guard} gate={gate}: C vs Py 最大差={dd.max():.6e}  平均={dd.mean():.6e}"
              f"  等价={dd.max() < 1e-6}")
    mt = (t >= 1100) & (t <= 2850)
    for (guard, gate), y in res.items():
        s = y.sum(1)
        print(f"  guard={guard} gate={gate}: 恒载段漂移={s[mt][-1]-s[mt][0]:+9.1f} ADC")

    print("\n=== ③ 阶梯消失判据（逐秒 applied 单步跌幅 >30 的事件数）===")
    for (guard, gate), y in res.items():
        ap = (pre.sum(1) - y.sum(1))[mt]
        tt = t[mt]
        sec = np.floor(tt).astype(np.int64)
        s0 = sec - sec.min()
        cnt = np.bincount(s0, minlength=s0[-1] + 1)
        aps = np.bincount(s0, weights=ap, minlength=cnt.size) / np.maximum(cnt, 1)
        print(f"  guard={guard} gate={gate}: 台阶数={int((np.diff(aps) < -30).sum()):3d}  "
              f"applied 末值={ap[-1]:7.1f}")

    np.savez_compressed(OUT / "dll_verify.npz", t=t,
                        c_g0=res[(0, 0)].sum(1), c_k10=res[(0, 1)].sum(1), c_gate=res[(1, 0)].sum(1))


if __name__ == "__main__":
    main()
