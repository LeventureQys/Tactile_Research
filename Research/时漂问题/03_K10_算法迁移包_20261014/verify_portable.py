# -*- coding: utf-8 -*-
"""验证 creep_observer_portable.h（编译为 DLL）与工具箱 Python 参考实现逐帧等价，
并导出黄金向量（输入 / 期望输出）供任何新实现自检。

portable-1.1 起同时导出**抬升门**（floor_rise_idle_only=1）的期望输出：
两个开关默认关闭，关闭档与 portable-1.0 的契约完全相同。

运行：
    python verify_portable.py            # 验证 + 生成黄金向量
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\archived"
            r"\右拇指指腹\设备1\测试2-2026-10-13_16-43-59")
GOLDEN = HERE / "golden_vectors"
FRAMES = 3000

sys.path.insert(0, str(TOOLBOX))
from plugins.k9_creep_observer import _compensate  # noqa: E402

PARAM_FIELDS = [
    "r_fast", "tau_c_fast_s", "tau_r_fast_s", "r_slow_max", "tau_r_slow_s",
    "slope_cap_frac", "slope_gate_frac", "tau_zero_s", "idle_frac", "y_max_tau_s",
    "tau_slope_s", "edge_slope_thres", "edge_refract_s", "edge_boost_s",
    "tau_c_fast_boost_s", "soft_unfreeze_s", "hold_eps", "hold_tau_s",
    "slow_confirm_s", "y_floor_tau_s", "tau_r_slow_idle_s", "bypass_release_frac",
    "bypass_engage_frac", "bypass_noise_sigma", "bypass_base_tau_s",
    "ramp_slope_min", "ramp_full_s",
]


class Dv6Params(ctypes.Structure):
    """必须与 creep_observer_portable.h 的 dv6_params 逐字段、逐顺序一致。"""

    _fields_ = (
        [(name, ctypes.c_double) for name in PARAM_FIELDS]
        + [("guard_enable", ctypes.c_int),
           ("guard_floor_confirm_s", ctypes.c_double),
           ("guard_idle_confirm_s", ctypes.c_double),
           ("guard_floor_deep", ctypes.c_double),
           ("floor_rise_idle_only", ctypes.c_int),
           ("guard_floor_recover_s", ctypes.c_double)]
    )


def load_stream(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def run_dll(dll, values: np.ndarray, time: np.ndarray, params: Dv6Params,
            floor_gate: int) -> np.ndarray:
    params.floor_rise_idle_only = floor_gate
    n_ch = values.shape[1]
    handle = dll.dv6_portable_new(n_ch, ctypes.byref(params), 2)
    if not handle:
        raise SystemExit("dv6_portable_new failed")
    buffer = (ctypes.c_double * n_ch)()
    produced = np.empty_like(values)
    for index in range(values.shape[0]):
        for k in range(n_ch):
            buffer[k] = float(values[index, k])
        dll.dv6_portable_process(handle, float(time[index]), buffer, n_ch)
        for k in range(n_ch):
            produced[index, k] = buffer[k]
    dll.dv6_portable_free(handle)
    return produced


def main() -> None:
    dll_path = HERE / "dv6_portable.dll"
    if not dll_path.exists():
        raise SystemExit("先运行 build_dll.bat 生成 dv6_portable.dll")
    dll = ctypes.CDLL(str(dll_path))
    dll.dv6_portable_new.restype = ctypes.c_void_p
    dll.dv6_portable_new.argtypes = [ctypes.c_int, ctypes.POINTER(Dv6Params), ctypes.c_int]
    dll.dv6_portable_process.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                         ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    dll.dv6_portable_free.argtypes = [ctypes.c_void_p]
    dll.dv6_portable_params_bytes.restype = ctypes.c_int

    # 结构体失配是静默错误：C 侧会读到缓冲区外的 4 字节，参数值随机。必须先卡死。
    c_bytes = int(dll.dv6_portable_params_bytes())
    py_bytes = int(ctypes.sizeof(Dv6Params))
    if c_bytes != py_bytes:
        raise SystemExit(f"dv6_params 结构体失配：DLL={c_bytes} bytes，ctypes={py_bytes} bytes。"
                         f"请核对本文件 Dv6Params 的字段与 creep_observer_portable.h。")

    session = json.loads((DATA / "session.json").read_text(encoding="utf-8"))
    raw_params = session["algorithm"]["params"]
    fps = float(session["devices"][0]["observed_fps"])

    p = Dv6Params()
    defaults = {  # 与 dv6_params_default / plan_v4 保持一致
        "r_fast": 0.01, "tau_c_fast_s": 40.0, "tau_r_fast_s": 6.0, "r_slow_max": 0.2,
        "tau_r_slow_s": 150.0, "slope_cap_frac": 0.05, "slope_gate_frac": 0.05,
        "tau_zero_s": 8.0, "idle_frac": 0.05, "y_max_tau_s": 600.0, "tau_slope_s": 1.0,
        "edge_slope_thres": 60.0, "edge_refract_s": 2.0, "edge_boost_s": 2.0,
        "tau_c_fast_boost_s": 2.0, "soft_unfreeze_s": 8.0, "hold_eps": 2.0,
        "hold_tau_s": 0.5, "slow_confirm_s": 5.0, "y_floor_tau_s": 300.0,
        "tau_r_slow_idle_s": 0.5, "bypass_release_frac": 1.15, "bypass_engage_frac": 1.25,
        "bypass_noise_sigma": 3.0, "bypass_base_tau_s": 10.0, "ramp_slope_min": 0.5,
        "ramp_full_s": 4.0,
    }
    plan_v4_overrides = {"r_fast": 0.03, "tau_c_fast_s": 2.0, "tau_r_fast_s": 0.1,
                         "r_slow_max": 0.1, "slope_cap_frac": 0.02, "slow_confirm_s": 3.0,
                         "soft_unfreeze_s": 2.0, "tau_r_slow_idle_s": 0.2}
    values_map = dict(defaults)
    values_map.update(plan_v4_overrides)
    # 两侧使用完全相同的参数合并规则：以会话参数覆盖（仅限已知键）
    session_overrides = {k: float(v) for k, v in raw_params.items() if k in defaults}
    values_map.update(session_overrides)
    for key, value in values_map.items():
        setattr(p, key, float(value))
    p.guard_enable = 0
    p.guard_floor_confirm_s = 0.5
    p.guard_idle_confirm_s = 0.3
    p.guard_floor_deep = 200.0
    p.floor_rise_idle_only = 0
    p.guard_floor_recover_s = 0.0

    pre = load_stream(DATA, "device_001_pre_seg0.csv")
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame - frame[0]) / fps
    values = np.asarray(pre.iloc[:, 3:], dtype=float)[:FRAMES]
    time = time[:FRAMES]
    n_ch = values.shape[1]

    # 工具箱插件（按 cpp 口径转写）作为等价性基准；它只实现关闭档
    reference, _ = _compensate(values, time, dict(values_map))
    reference = np.asarray(reference, dtype=float)

    produced = run_dll(dll, values, time, p, floor_gate=0)
    produced_gate = run_dll(dll, values, time, p, floor_gate=1)

    diff = np.abs(produced - reference)
    diff_gate = np.abs(produced_gate - produced)
    report = dict(
        frames=int(values.shape[0]), channels=int(n_ch),
        params_bytes=dict(dll=c_bytes, ctypes=py_bytes),
        max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
        max_abs_total_diff=float(np.abs(produced.sum(axis=1) - reference.sum(axis=1)).max()),
        equivalent=bool(diff.max() < 1e-6),
        floor_rise_gate=dict(
            max_abs_diff_vs_gate_off=float(diff_gate.max()),
            mean_abs_diff_vs_gate_off=float(diff_gate.mean()),
            max_abs_total_diff=float(np.abs(produced_gate.sum(axis=1)
                                            - produced.sum(axis=1)).max()),
            note="gate=1 是新增档，工具箱参考实现不含该逻辑，故只给与关闭档的差异量级；"
                 "其正确性由 golden_expected_output_gate1.csv 契约 + 产品侧逐帧对拍保证。",
        ),
    )

    GOLDEN.mkdir(exist_ok=True)
    np.savetxt(GOLDEN / "golden_time_s.csv", time, fmt="%.6f",
               header="timestamp_s", comments="")
    np.savetxt(GOLDEN / "golden_input.csv", values, fmt="%.6f", delimiter=",", comments="")
    np.savetxt(GOLDEN / "golden_expected_output.csv", reference, fmt="%.6f",
               delimiter=",", comments="")
    np.savetxt(GOLDEN / "golden_expected_output_gate1.csv", produced_gate, fmt="%.6f",
               delimiter=",", comments="")
    (GOLDEN / "golden_params.json").write_text(json.dumps({
        "note": "plan-v4 observer-k9 参数集（经会话实际参数覆盖）；时间轴首帧归零、单位秒；"
                "输入为算法输入显示值。guard_enable / floor_rise_idle_only 均取 0（关闭档）。",
        "params": {key: float(value) for key, value in values_map.items()},
        "guard": {"enabled": False},
        "floor_rise_idle_only": 0,
        "floor_rise_gate_vector": {
            "expected_output_file": "golden_expected_output_gate1.csv",
            "params_delta": {"floor_rise_idle_only": 1},
            "note": "同一份输入与参数，仅把抬升门打开；用于校验迁移实现的 portable-1.1 路径。",
        },
        "frame_count": int(values.shape[0]), "channels": int(n_ch),
        "source": str(DATA),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "verify_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("golden vectors ->", GOLDEN)


if __name__ == "__main__":
    main()
