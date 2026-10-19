# -*- coding: utf-8 -*-
"""验证 portable 实现的瞬态防护层（guard_enable=1）与 Python 参考实现一致。"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "02_K10_瞬态防护_20261014"
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
FRAMES = 3000

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PREV))
sys.path.insert(0, str(TOOLBOX))
from verify_portable import Dv6Params, PARAM_FIELDS  # noqa: E402
from guarded_k9 import compensate as py_compensate  # noqa: E402


def load_stream(name: str) -> pd.DataFrame:
    path = DATA / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def main() -> None:
    dll = ctypes.CDLL(str(HERE / "dv6_portable.dll"))
    dll.dv6_portable_new.restype = ctypes.c_void_p
    dll.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(Dv6Params), ctypes.c_int]
    dll.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                         ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    dll.dv6_portable_free.argtypes = [ctypes.c_void_p]

    session = json.loads((DATA / "session.json").read_text(encoding="utf-8"))
    params = {k: float(v) for k, v in session["algorithm"]["params"].items()
              if k in PARAM_FIELDS}
    fps = float(session["devices"][0]["observed_fps"])

    pre = load_stream("device_001_pre_seg0.csv")
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame - frame[0]) / fps
    values = np.asarray(pre.iloc[:, 3:], dtype=float)[:FRAMES]
    time = time[:FRAMES]
    n_ch = values.shape[1]

    p = Dv6Params()
    for key, value in params.items():
        setattr(p, key, float(value))
    p.guard_enable = 1
    p.guard_floor_confirm_s = 0.5
    p.guard_idle_confirm_s = 0.3
    p.guard_floor_deep = 200.0

    reference = np.asarray(py_compensate(values, time, dict(params), dict(enabled=True))[0],
                           dtype=float)

    handle = dll.dv6_portable_new(n_ch, ctypes.byref(p), 2)
    if not handle:
        raise SystemExit("create failed")
    buffer = (ctypes.c_double * n_ch)()
    produced = np.empty_like(values)
    for index in range(values.shape[0]):
        for k in range(n_ch):
            buffer[k] = float(values[index, k])
        dll.dv6_portable_process(handle, float(time[index]), buffer, n_ch)
        for k in range(n_ch):
            produced[index, k] = buffer[k]
    dll.dv6_portable_free(handle)

    diff = np.abs(produced - reference)
    report = dict(frames=int(values.shape[0]), channels=int(n_ch), guard_enabled=True,
                  max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                  equivalent=bool(diff.max() < 1e-6))
    (HERE / "guard_parity_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
