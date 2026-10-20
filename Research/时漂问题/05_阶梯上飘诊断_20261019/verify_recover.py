"""K10 档「下探回抬锚」的产品 ↔ 本包对拍与效果验证。

四臂：
  recover=0  gate=1 guard=0/1   ← 默认档，必须与上一版逐帧一致
  recover=30 gate=1 guard=0/1   ← 新能力开启档
"""
import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import core

HARNESS = (Path(__file__).resolve().parent / "product_harness" / "bin"
           / "product_harness.exe")
WORK = Path(r"D:\Temp\k10_recover")
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


class P(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_double) for n in FIELDS]
                + [("guard_enable", ctypes.c_int),
                   ("guard_floor_confirm_s", ctypes.c_double),
                   ("guard_idle_confirm_s", ctypes.c_double),
                   ("guard_floor_deep", ctypes.c_double),
                   ("floor_rise_idle_only", ctypes.c_int),
                   ("guard_floor_recover_s", ctypes.c_double)])


def bind(path):
    d = ctypes.CDLL(str(path))
    d.dv6_portable_new.restype = ctypes.c_void_p
    d.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(P), ctypes.c_int]
    d.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                       ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    d.dv6_portable_free.argtypes = [ctypes.c_void_p]
    d.dv6_portable_params_bytes.restype = ctypes.c_int
    return d


def fill(vals, guard, gate, recover):
    p = P()
    for k in FIELDS:
        setattr(p, k, float(vals[k]))
    p.guard_enable = guard
    p.guard_floor_confirm_s = 0.5
    p.guard_idle_confirm_s = 0.3
    p.guard_floor_deep = 200.0
    p.floor_rise_idle_only = gate
    p.guard_floor_recover_s = recover
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
    WORK.mkdir(parents=True, exist_ok=True)
    ts, pre = load_cells("device_001_pre_seg0.csv")
    t = ts - ts[0]
    n = pre.shape[1]
    mt = (t >= 1100) & (t <= 2850)

    inp = WORK / "frames.bin"
    if not inp.exists():
        packed = np.empty((pre.shape[0], n + 1))
        packed[:, 0] = ts
        packed[:, 1:] = pre
        packed.tofile(inp)

    dll = bind(PKG / "dv6_portable.dll")
    if int(dll.dv6_portable_params_bytes()) != ctypes.sizeof(P):
        raise SystemExit(f"结构体失配 DLL={dll.dv6_portable_params_bytes()} "
                         f"ctypes={ctypes.sizeof(P)}")
    sj = json.loads((ROOT / "session.json").read_text(encoding="utf-8"))
    vals = dict(core.BASE)
    vals.update({k: v for k, v in sj["algorithm"]["params"].items() if k in core.BASE})

    def prod(gate, guard, recover, tag):
        out = WORK / f"prod_{tag}.bin"
        r = subprocess.run([str(HARNESS), str(inp), str(out), str(n), str(gate),
                            str(guard), "1", str(recover)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{tag}: {r.stdout} {r.stderr}")
        return np.fromfile(out, dtype=np.float64).reshape(-1, n)

    arms = {}
    print(f"{'档位':44s} {'逐帧差':>10s} {'漂移':>9s} {'ap末':>8s} {'yfl末':>9s} {'yfl最小':>9s}")
    for gate, guard, recover, tag in (
        (1, 0, 0.0, "v34_r0"), (1, 0, 30.0, "v34_r30"),
        (1, 1, 0.0, "k10_r0"), (1, 1, 30.0, "k10_r30"),
    ):
        a = prod(gate, guard, recover, tag)
        b = run_dll(dll, pre, ts, fill(vals, guard, gate, recover))
        d = np.abs(a - b)
        s = a.sum(1)
        ap = pre.sum(1) - s
        _, L = None, None
        arms[tag] = (a, d)
        name = f"gate={gate} guard={guard} recover={recover:.0f}"
        print(f"{name:44s} {d.max():10.3e} {s[mt][-1]-s[mt][0]:+9.1f} {ap[mt][-1]:8.1f}")

    # y_floor 轨迹用 Python 复刻取（产品不导出内部状态）
    print("\n=== y_floor 末值 / 最小值（Python 复刻，仅 recover=30 需 fix_first_frame）===")
    import k10_recover_probe as probe
    for guard in (0, 1):
        for recover in (0.0, 30.0):
            o = dict(core.BASE)
            o["floor_rise_gate"] = 1
            o["guard_enable"] = guard
            v, L = probe.run(pre, ts, o, recover_tau=recover,
                             fix_first_frame=recover > 0, log=True)
            s = v.sum(1)
            yfl = L["y_floor"]
            print(f"  guard={guard} recover={recover:2.0f}: 漂移={s[mt][-1]-s[mt][0]:+8.1f} "
                  f"yfl末={yfl[mt][-1].sum():8.1f} yfl最小={yfl[mt].min(0).sum():8.1f} "
                  f"span末={(L['y_floor'][mt][-1]*0).sum():.0f}".replace(" span末=0", ""))

    report = {k: {"max_abs_diff_vs_portable": float(v[1].max())} for k, v in arms.items()}
    (OUT / "k10_recover_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {OUT/'k10_recover_result.json'}")


if __name__ == "__main__":
    main()
