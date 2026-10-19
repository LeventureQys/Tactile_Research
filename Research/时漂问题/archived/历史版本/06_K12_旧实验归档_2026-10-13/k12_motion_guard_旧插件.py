# -*- coding: utf-8 -*-
"""K12 抗蠕变补偿算法插件（dptool 算法插件协议 v1，dataset 模式）。

K12 = **K9（v3.4 观测器）原样保留 + 一个可关闭的「微小变化保护层」**。

────────────────────────────────────────────────────────────────────────
为什么需要这一层
────────────────────────────────────────────────────────────────────────
K9 的慢态 x_slow 用 `slope − dx1_rate` 积分，判据只看「斜率是否够小」。
真实的小荷载变化（10g 级）虽然是**阶跃**，但其幅度小、经过 τ_slope=1s 低通后
残余斜率很低，于是和「恒定慢漂移」落进同一个门内 —— 一旦受载确认期满
（slow_confirm_s），x_slow 就会把这部分真实变化当成蠕变一起扣掉。

K12 的判定量刻意做成**二阶高通**，使两类信号在数值上分开：

    ref_fast   τ = motion_fast_tau_s   （跟随局部真实变化）
    ref_slow   τ = motion_slow_tau_s   （跟随慢漂趋势）
    dev_raw  = ref_fast − ref_slow
    dev_base = 慢速均值(dev_raw, τ = motion_base_tau_s)
    dev_ac   = dev_raw − dev_base        ← 保护判定量

  · 恒定速率漂移（r 恒定）：dev_raw → r·(τ_slow − τ_fast) 趋于常数，
    dev_base 追上同一常数 ⇒ **dev_ac → 0**，保护层不介入，x_slow 照常压制漂移。
  · 真实阶跃/斜坡（10g 级）：dev_raw 冲高而 dev_base 滞后 ⇒ **dev_ac 出现尖峰**，
    越过 motion_thresh_adc 并持续 motion_confirm_s 后进入保护窗。

保护窗内的动作：
  ① x_slow 积分乘以 motion_slow_gain（默认 0 = 完全冻结）；
  ② 清零 load_dwell（受载确认），使该事件不被确认为慢漂移；
  ③ 窗长 motion_hold_s，期间若再次检出真实变化则顺延。

保护层**只抑制慢态积分**，不动 x_fast、零点跟踪、K6 预留池与 K7 空载旁路，
故 K9 的既有行为在未触发保护时逐位不变。

────────────────────────────────────────────────────────────────────────
与 K9 的关系
────────────────────────────────────────────────────────────────────────
`motion_guard_enable = False` 时，本插件与 K9 插件在数值上完全等价
（同一 `_compensate` 结构，保护相关状态全部旁路），可直接用作 A/B 对照。

参数键名与产线 `session.json` 的 `algorithm.params` 同名互操作：K9 的 28 项保持
原名，K12 新增 8 项以 `motion_` 前缀标识。

跨通道耦合：K7 全局总值旁路用 `sum(全部输入通道)` 判直通，故 dataset 模式必须
整表喂入（`ctx.values` 全部通道）。
"""
from __future__ import annotations

import math

import numpy as np

from dptool.algorithms import AlgorithmOutput, AlgorithmResult, Cancelled, ParamSpec

PLUGIN_META = {
    "id": "k12_motion_guard",
    "name": "K12 抗蠕变补偿（K9 + 微小变化保护）",
    "version": "1.0.0",
    "protocol": 1,
    "description": "K12 = K9（v3.4 观测器）原样保留 + 可关闭的微小变化保护层：用双参考线的"
                   "二阶高通量 dev_ac 区分「恒定慢漂移」与「真实小荷载变化」，在真实变化期间"
                   "冻结慢态积分，避免 10g 级荷载变化被当作蠕变扣除；关掉保护层即逐位等于 K9",
    "input_mode": "dataset",
    "time_modes": ["frame_fps", "elapsed", "timestamp"],
}

_PROGRESS_STRIDE = 2000
_NAN_SEGMENT_MIN_S = 1.0
_DENOM_FLOOR = 1e-9

_WORKTREE_DEFAULTS = {
    "r_slow_max": 0.2, "tau_r_slow_s": 150.0, "tau_slope_s": 1.0, "slope_gate_frac": 0.05,
    "slope_cap_frac": 0.05, "tau_zero_s": 8.0, "idle_frac": 0.05, "y_max_tau_s": 600.0,
    "r_fast": 0.01, "tau_c_fast_s": 40.0, "tau_r_fast_s": 6.0, "edge_slope_thres": 60.0,
    "edge_refract_s": 2.0, "edge_boost_s": 2.0, "tau_c_fast_boost_s": 2.0,
    "soft_unfreeze_s": 8.0, "hold_eps": 2.0, "hold_tau_s": 0.5, "slow_confirm_s": 5.0,
    "y_floor_tau_s": 300.0, "tau_r_slow_idle_s": 0.5, "bypass_release_frac": 1.15,
    "bypass_engage_frac": 1.25, "bypass_noise_sigma": 3.0, "bypass_base_tau_s": 10.0,
    "ramp_slope_min": 0.5, "ramp_full_s": 4.0,
}

_PARAM_SET_DEFAULT = "plan-v4 observer-k12"


def param_specs() -> tuple:
    """36 项参数 = K9 的 28 项（键名/默认值/顺序不变）+ K12 新增 8 项。

    K9 的 8 项设备寄存器参数（201~208）与 K12 的 8 项保护参数 `hidden=False`
    （渲染进 UI 表单）；其余保持 `hidden=True`。
    """
    return (
        # ── K9 参数（寄存器可下发 201~208）──
        ParamSpec(name="r_fast", label="快态幅度比 r_fast", type="float", default=0.01,
                  min=0.0, max=0.5, step=0.01, decimals=3, unit="", hidden=False,
                  help="受载朝 r_fast·e 收敛；0 = 关闭快态。寄存器 201 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="slow_confirm_s", label="慢态受载确认窗 slow_confirm_s", type="float",
                  default=5.0, min=0.0, max=60.0, step=0.5, decimals=2, unit="s", hidden=False,
                  help="x2 积分需连续受载确认时长；任一快变清零重计。寄存器 202 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="soft_unfreeze_s", label="慢态沿后软冻结窗 soft_unfreeze_s", type="float",
                  default=8.0, min=0.1, max=60.0, step=0.5, decimals=2, unit="s", hidden=False,
                  help="沿后 x2 软冻结窗。寄存器 203 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="tau_r_slow_idle_s", label="空载慢态泄放 τ tau_r_slow_idle_s", type="float",
                  default=0.5, min=0.0, max=300.0, step=0.5, decimals=2, unit="s", hidden=False,
                  help="空载期 x2 快泄放 τ。寄存器 204 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="tau_r_fast_s", label="快态空载恢复 τ tau_r_fast_s", type="float",
                  default=6.0, min=0.05, max=300.0, step=0.05, decimals=2, unit="s", hidden=False,
                  help="快态空载恢复时间常数。寄存器 205 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="tau_c_fast_s", label="快态受载收敛 τ tau_c_fast_s", type="float",
                  default=40.0, min=0.5, max=120.0, step=0.5, decimals=1, unit="s", hidden=False,
                  help="快态受载收敛时间常数。寄存器 206 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="slope_cap_frac", label="慢态积分速率上限 slope_cap_frac", type="float",
                  default=0.05, min=0.0, max=0.05, step=0.001, decimals=3, unit="1/s", hidden=False,
                  help="每帧 x2 增量钳在 ±该值·max(e,1)；0 = 关闭慢态积分。寄存器 207 · 定点 ×1000",
                  group="K9 参数（寄存器可下发）"),
        ParamSpec(name="r_slow_max", label="慢态幅度上限 r_slow_max", type="float", default=0.2,
                  min=0.0, max=0.6, step=0.01, decimals=3, unit="", hidden=False,
                  help="慢态上限 = r_slow_max·e；0 = 关闭慢态。寄存器 208 · 定点 ×100",
                  group="K9 参数（寄存器可下发）"),
        # ── K12 新增：微小变化保护层（8）──
        ParamSpec(name="motion_guard_enable", label="启用微小变化保护", type="bool", default=True,
                  hidden=False, help="关闭后 K12 与 K9 逐位等价（可用于 A/B 对照）",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_thresh_adc", label="保护触发阈值 motion_thresh_adc", type="float",
                  default=25.0, min=0.0, max=100000.0, step=1.0, decimals=1, unit="ADC",
                  hidden=False,
                  help="判定量 dev_ac 的绝对门限（ADC）。需高于噪声残差（实测 σ≈10~20 ADC）"
                       "且低于目标荷载变化（10g 实测 60~130 ADC）",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_fast_tau_s", label="快参考 τ motion_fast_tau_s", type="float",
                  default=0.5, min=0.01, max=1000.0, step=0.05, decimals=3, unit="s", hidden=False,
                  help="ref_fast 跟随真实变化的时间常数（越小对台阶越敏感）",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_slow_tau_s", label="慢参考 τ motion_slow_tau_s", type="float",
                  default=5.0, min=0.05, max=10000.0, step=0.1, decimals=3, unit="s", hidden=False,
                  help="ref_slow 跟随慢漂趋势的时间常数；与快参考之差构成一阶判定量 dev_raw",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_base_tau_s", label="判定量基线 τ motion_base_tau_s", type="float",
                  default=8.0, min=0.05, max=10000.0, step=0.1, decimals=3, unit="s", hidden=False,
                  help="dev_raw 的慢均值时间常数；dev_ac = dev_raw − 该基线，"
                       "使恒定速率漂移的判定量归零（二阶高通的关键）",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_confirm_s", label="保护确认时长 motion_confirm_s", type="float",
                  default=0.6, min=0.0, max=60.0, step=0.05, decimals=3, unit="s", hidden=False,
                  help="dev_ac 连续超阈达到该时长才进入保护（滤掉单帧尖峰）",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_hold_s", label="保护保持时长 motion_hold_s", type="float",
                  default=3.0, min=0.0, max=600.0, step=0.1, decimals=2, unit="s", hidden=False,
                  help="进入保护后的持续时长；期间再检出真实变化则顺延",
                  group="K12 微小变化保护"),
        ParamSpec(name="motion_slow_gain", label="保护期慢态增益 motion_slow_gain", type="float",
                  default=0.0, min=0.0, max=1.0, step=0.05, decimals=3, unit="×", hidden=False,
                  help="保护期内 x2 积分乘以该系数；0 = 完全冻结，1 = 保护无效",
                  group="K12 微小变化保护"),
        # ── v3.4 底座（6）──
        ParamSpec(name="tau_r_slow_s", label="慢态空载恢复 τ tau_r_slow_s", type="float",
                  default=150.0, min=0.001, max=100000.0, step=1.0, decimals=3, unit="s",
                  help="慢态空载恢复时间常数（非空载且 e≤0 时使用）",
                  group="v3.4 底座", hidden=True),
        ParamSpec(name="tau_slope_s", label="慢漂移速率低通 τ tau_slope_s", type="float",
                  default=1.0, min=0.001, max=10000.0, step=0.1, decimals=3, unit="s",
                  help="慢漂移速率估计的低通 τ",
                  group="v3.4 底座", hidden=True),
        ParamSpec(name="slope_gate_frac", label="慢态积分斜率门 slope_gate_frac", type="float",
                  default=0.05, min=0.0, max=10.0, step=0.01, decimals=3, unit="",
                  help="|dv/dt| > 该值·e/s 视为沿/快变，x2 不积分",
                  group="v3.4 底座", hidden=True),
        ParamSpec(name="tau_zero_s", label="零点跟踪 τ tau_zero_s", type="float", default=8.0,
                  min=0.001, max=100000.0, step=0.5, decimals=3, unit="s",
                  help="零点跟踪时间常数（仅近零带内生效）",
                  group="v3.4 底座", hidden=True),
        ParamSpec(name="idle_frac", label="近零带比例 idle_frac", type="float", default=0.05,
                  min=0.0, max=1.0, step=0.01, decimals=3, unit="",
                  help="去趋势空载判据的比例带",
                  group="v3.4 底座", hidden=True),
        ParamSpec(name="y_max_tau_s", label="量程包络衰减 τ y_max_tau_s", type="float",
                  default=600.0, min=0.001, max=1000000.0, step=10.0, decimals=1, unit="s",
                  help="量程包络 y_max 的衰减时间常数",
                  group="v3.4 底座", hidden=True),
        # ── H3 沿后前馈（4）──
        ParamSpec(name="edge_slope_thres", label="沿判定阈值 edge_slope_thres", type="float",
                  default=60.0, min=0.0, max=100000.0, step=1.0, decimals=1, unit="ADC/s",
                  help="低通导数上升沿判定阈值（ADC 域标定值）",
                  group="H3 沿后前馈", hidden=True),
        ParamSpec(name="edge_refract_s", label="沿触发最小间隔 edge_refract_s", type="float",
                  default=2.0, min=0.0, max=100000.0, step=0.5, decimals=2, unit="s",
                  help="沿触发最小间隔（实际取 max(edge_refract_s, edge_boost_s)）",
                  group="H3 沿后前馈", hidden=True),
        ParamSpec(name="edge_boost_s", label="沿后前馈窗 edge_boost_s", type="float",
                  default=2.0, min=0.0, max=100000.0, step=0.5, decimals=2, unit="s",
                  help="沿后前馈窗：窗内快态改用 tau_c_fast_boost_s 收敛",
                  group="H3 沿后前馈", hidden=True),
        ParamSpec(name="tau_c_fast_boost_s", label="沿后快态 τ tau_c_fast_boost_s", type="float",
                  default=2.0, min=0.001, max=100000.0, step=0.5, decimals=3, unit="s",
                  help="沿后前馈期（及缓坡满窗）的快态收敛时间常数",
                  group="H3 沿后前馈", hidden=True),
        # ── K6 预留池（2）──
        ParamSpec(name="hold_eps", label="预留池限额余量 hold_eps", type="float", default=2.0,
                  min=0.0, max=100000.0, step=0.5, decimals=2, unit="ADC/s",
                  help="受载期施加补偿的增速 ≤ 输入快速导数 + 该值；≤ 0 = 关闭预留池",
                  group="K6 预留池", hidden=True),
        ParamSpec(name="hold_tau_s", label="预留池快速低通 τ hold_tau_s", type="float",
                  default=0.5, min=0.001, max=10000.0, step=0.05, decimals=3, unit="s",
                  help="预留池限额用的快速低通 τ",
                  group="K6 预留池", hidden=True),
        # ── K7 棘轮与空载（5）──
        ParamSpec(name="y_floor_tau_s", label="基线包络爬升 τ y_floor_tau_s", type="float",
                  default=300.0, min=0.001, max=1000000.0, step=10.0, decimals=1, unit="s",
                  help="去趋势基线包络 y_floor 的爬升 τ；下行瞬时跟随",
                  group="K7 棘轮与空载", hidden=True),
        ParamSpec(name="bypass_release_frac", label="直通释放比 bypass_release_frac", type="float",
                  default=1.15, min=0.0, max=100.0, step=0.01, decimals=3, unit="",
                  help="总值 < max(该值·基线, 基线+3σ) 时整帧直通",
                  group="K7 棘轮与空载", hidden=True),
        ParamSpec(name="bypass_engage_frac", label="直通恢复比 bypass_engage_frac", type="float",
                  default=1.25, min=0.0, max=100.0, step=0.01, decimals=3, unit="",
                  help="总值 > max(该值·基线, 基线+4σ) 才恢复算法（迟滞防抖）",
                  group="K7 棘轮与空载", hidden=True),
        ParamSpec(name="bypass_noise_sigma", label="门限噪声带宽 bypass_noise_sigma", type="float",
                  default=3.0, min=0.0, max=100.0, step=0.5, decimals=2, unit="σ",
                  help="直通门限的噪声带宽",
                  group="K7 棘轮与空载", hidden=True),
        ParamSpec(name="bypass_base_tau_s", label="空载基线跟踪 τ bypass_base_tau_s", type="float",
                  default=10.0, min=0.001, max=10000.0, step=0.5, decimals=3, unit="s",
                  help="空载期总值基线/噪声的跟踪 τ",
                  group="K7 棘轮与空载", hidden=True),
        # ── K9 缓坡前馈（2）──
        ParamSpec(name="ramp_slope_min", label="缓坡斜率下限 ramp_slope_min", type="float",
                  default=0.5, min=0.0, max=100000.0, step=0.1, decimals=3, unit="ADC/s",
                  help="缓坡前馈的斜率下限；ADC 域标定值",
                  group="K9 缓坡前馈", hidden=True),
        ParamSpec(name="ramp_full_s", label="缓坡满窗时长 ramp_full_s", type="float", default=4.0,
                  min=0.0, max=100000.0, step=0.5, decimals=2, unit="s",
                  help="缓坡持续满该时长后 tc1 全额过渡到沿后前馈值；≤ 0 = 关闭缓坡前馈",
                  group="K9 缓坡前馈", hidden=True),
        # ── 标识（1）──
        ParamSpec(name="param_set", label="参数集标识 param_set", type="enum",
                  default=_PARAM_SET_DEFAULT, choices=(_PARAM_SET_DEFAULT,),
                  help="录制/下发所用参数集标识；仅用于互操作与追溯，不参与运算",
                  group="标识", hidden=True),
    )


def _spec_defaults() -> dict:
    return {spec.name: spec.default for spec in param_specs()}


def _coerce_params(params) -> dict:
    """按 schema 取全部参数；缺键/坏值回落 schema 默认值。

    bool 参数（motion_guard_enable）单独保留布尔语义，不进 float 通道。
    """
    merged = _spec_defaults()
    if isinstance(params, dict):
        for key, value in params.items():
            if key in merged:
                merged[key] = value
    out: dict = {}
    for name, value in merged.items():
        if name == "param_set":
            continue
        if name == "motion_guard_enable":
            if isinstance(value, str):
                out[name] = value.strip().lower() in ("1", "true", "yes", "on", "是")
            else:
                out[name] = bool(value)
            continue
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            out[name] = float(_spec_defaults()[name])
    return out


def _positive(value) -> float:
    number = float(value)
    return number if number > _DENOM_FLOOR else _DENOM_FLOOR


def _finite_row(row: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(row), row, 0.0)


def _nan_runs(finite_mask: np.ndarray, time_axis: np.ndarray) -> list:
    runs: list = []
    start = -1
    for index in range(finite_mask.size):
        if not bool(finite_mask[index]):
            if start < 0:
                start = index
            continue
        if start >= 0:
            runs.append((start, index - 1))
            start = -1
    if start >= 0:
        runs.append((start, finite_mask.size - 1))
    spans: list = []
    for first, last in runs:
        span = float(time_axis[last]) - float(time_axis[first])
        if not np.isfinite(span) or span < 0.0:
            span = 0.0
        if span > _NAN_SEGMENT_MIN_S:
            spans.append((first, last, span))
    return spans


def _compensate(values: np.ndarray, time_axis: np.ndarray, params: dict, progress=None,
                cancel=None):
    """逐帧补偿核心，返回 `(out, stats)`。

    K9 主体逐行保持 creep_observer.cpp 口径；K12 仅在慢态积分处插入保护门。
    """
    p = _coerce_params(params)
    guard_enable = bool(p["motion_guard_enable"])
    r_fast = p["r_fast"]
    tau_c_fast_s = _positive(p["tau_c_fast_s"])
    tau_r_fast_s = _positive(p["tau_r_fast_s"])
    r_slow_max = p["r_slow_max"]
    tau_r_slow_s = _positive(p["tau_r_slow_s"])
    tau_slope_s = _positive(p["tau_slope_s"])
    slope_gate_frac = p["slope_gate_frac"]
    slope_cap_frac = p["slope_cap_frac"]
    tau_zero_s = _positive(p["tau_zero_s"])
    idle_frac = p["idle_frac"]
    y_max_tau_s = _positive(p["y_max_tau_s"])
    edge_slope_thres = p["edge_slope_thres"]
    edge_refract_s = p["edge_refract_s"]
    edge_boost_s = p["edge_boost_s"]
    tau_c_fast_boost_s = _positive(p["tau_c_fast_boost_s"])
    soft_unfreeze_s = p["soft_unfreeze_s"]
    hold_eps = p["hold_eps"]
    hold_tau_s = _positive(p["hold_tau_s"])
    slow_confirm_s = p["slow_confirm_s"]
    y_floor_tau_s = _positive(p["y_floor_tau_s"])
    tau_r_slow_idle_s = p["tau_r_slow_idle_s"]
    bypass_release_frac = p["bypass_release_frac"]
    bypass_engage_frac = p["bypass_engage_frac"]
    bypass_noise_sigma = p["bypass_noise_sigma"]
    bypass_base_tau_s = _positive(p["bypass_base_tau_s"])
    ramp_slope_min = p["ramp_slope_min"]
    ramp_full_s = p["ramp_full_s"]
    # K12 保护层
    motion_thresh_adc = p["motion_thresh_adc"]
    tau_mf = _positive(p["motion_fast_tau_s"])
    tau_ms = _positive(p["motion_slow_tau_s"])
    tau_mb = _positive(p["motion_base_tau_s"])
    motion_confirm_s = p["motion_confirm_s"]
    motion_hold_s = p["motion_hold_s"]
    motion_slow_gain = min(max(p["motion_slow_gain"], 0.0), 1.0)

    n_rows, n_ch = int(values.shape[0]), int(values.shape[1])
    out = values.copy()
    bypass_mask = np.zeros(n_rows, dtype=bool)
    guard_mask = np.zeros(n_rows, dtype=bool)
    if n_rows == 0 or n_ch == 0:
        return out, {"bypass_mask": bypass_mask, "guard_mask": guard_mask,
                     "nan_runs": [], "nan_frames": 0, "guard_frames": 0}

    finite_mask = np.isfinite(values).all(axis=1)
    first = _finite_row(values[0])
    zero = first.copy()
    y_max = np.zeros(n_ch)
    v_lp = first.copy()
    v_fast_lp = first.copy()
    y_floor = first.copy()
    load_dwell = np.zeros(n_ch)
    ramp_dwell = np.zeros(n_ch)
    t_edge = np.full(n_ch, 1.0e6)
    applied = np.zeros(n_ch)
    x_fast = np.zeros(n_ch)
    x_slow = np.zeros(n_ch)
    # K12：双参考线 + 判定量基线 + 保护计时
    ref_fast = np.zeros(n_ch)
    ref_slow = np.zeros(n_ch)
    dev_base = np.zeros(n_ch)
    guard_dwell = np.zeros(n_ch)
    guard_hold = np.zeros(n_ch)
    total_baseline = float(first.sum())
    total_noise = 0.0
    last_ts = float(time_axis[0]) if time_axis.size else 0.0
    refract = max(edge_refract_s, edge_boost_s)

    for index in range(1, n_rows):
        raw = values[index]
        row_finite = bool(finite_mask[index])
        v = raw if row_finite else _finite_row(raw)

        dt = float(time_axis[index]) - last_ts
        last_ts = float(time_axis[index])
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.0
        else:
            dt = min(dt, 0.1)
        if not row_finite:
            dt = 0.0

        bypass_frame = False
        guard_on = np.zeros(n_ch, dtype=bool)
        if dt > 0.0:
            # ── K7 全局总值旁路（cpp:76-94）──
            total = float(v.sum())
            if np.isfinite(total):
                alpha = min(dt / bypass_base_tau_s, 1.0)
                release = max(bypass_release_frac * total_baseline,
                              total_baseline + bypass_noise_sigma * total_noise)
                engage = max(bypass_engage_frac * total_baseline,
                             total_baseline + (bypass_noise_sigma + 1.0) * total_noise)
                if total < engage:
                    total_baseline += alpha * (total - total_baseline)
                    total_noise += alpha * (abs(total - total_baseline) - total_noise)
                    if total_noise < 0.0:
                        total_noise = 0.0
                if total < release:
                    bypass_frame = True

        if dt > 0.0:
            # ── 零点跟踪 + 去趋势空载门限（cpp:96-117）──
            y_decay = math.exp(-dt / y_max_tau_s)
            y0 = v - zero
            y_floor = np.where(y0 < y_floor, y0, y_floor + (dt / y_floor_tau_s) * (y0 - y_floor))
            y_max = np.maximum(y_max * y_decay, np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            idle = (y0 - y_floor) < idle_frac * span
            load_dwell = np.where(idle, 0.0, load_dwell + dt)
            zero = np.where(idle, zero + (dt / tau_zero_s) * (v - zero), zero)
            y = v - zero
            # ── 低通导数（cpp:118-121）──
            slope = (v - v_lp) / tau_slope_s

            # ── K12 微小变化保护层（唯一新增结构；guard_enable=False 时整段旁路）──
            if guard_enable:
                ref_fast = ref_fast + min(dt / tau_mf, 1.0) * (y - ref_fast)
                ref_slow = ref_slow + min(dt / tau_ms, 1.0) * (y - ref_slow)
                dev_raw = ref_fast - ref_slow
                dev_base = dev_base + min(dt / tau_mb, 1.0) * (dev_raw - dev_base)
                dev_ac = dev_raw - dev_base
                active = np.abs(dev_ac) > motion_thresh_adc
                guard_dwell = np.where(active, guard_dwell + dt, 0.0)
                fire = guard_dwell >= motion_confirm_s
                # 边沿触发 + 触发时把判定量基线同步到当前 dev_raw：
                # 否则加载瞬态（数万 ADC 的直流偏置）会让 dev_base 以 τ 追赶数十秒，
                # 保护窗被"长尾"拖住，慢态在整个长尾内都被冻结（实测触发率可到 67%）。
                # 同步后 dev_ac 立即归零，保护只在「新变化出现」时短暂开启。
                if bool(fire.any()):
                    dev_base = np.where(fire, dev_raw, dev_base)
                    guard_dwell = np.where(fire, 0.0, guard_dwell)
                guard_hold = np.where(fire, motion_hold_s, np.maximum(guard_hold - dt, 0.0))
                guard_on = guard_hold > 0.0
                if bool(guard_on.any()):
                    guard_mask[index] = True

            # ── 快态 + H3 沿后前馈 + K9 缓坡前馈（cpp:122-152）──
            e_now = np.maximum(y - x_fast - x_slow, 0.0)
            t_edge = t_edge + dt
            fire_edge = (slope > edge_slope_thres) & (t_edge > refract)
            t_edge = np.where(fire_edge, 0.0, t_edge)
            ramp_on = (slope > ramp_slope_min) & (slope <= edge_slope_thres)
            ramp_dwell = np.where(ramp_on, ramp_dwell + dt, 0.0)
            load_dwell = np.where(np.abs(slope) > edge_slope_thres, 0.0, load_dwell)
            if edge_boost_s > 0.0:
                tc1 = np.where(t_edge < edge_boost_s, tau_c_fast_boost_s, tau_c_fast_s)
            else:
                tc1 = np.full(n_ch, tau_c_fast_s)
            if ramp_full_s > 0.0:
                weight = np.minimum(ramp_dwell / ramp_full_s, 1.0)
                tc1 = tc1 * (1.0 - weight) + tau_c_fast_boost_s * weight
            dx1_rate = np.where(e_now > 0.0,
                                (r_fast * e_now - x_fast) / tc1,
                                -x_fast / tau_r_fast_s)
            x_fast = np.maximum(x_fast + dx1_rate * dt, 0.0)
            v_lp = v_lp + (dt / tau_slope_s) * (v - v_lp)

            # ── 慢态（cpp:153-181）+ K12 保护门 ──
            e = np.maximum(y - x_fast - x_slow, 0.0)
            span_k = np.maximum(y_max - y_floor, 1.0)
            idle_k = (y - y_floor) < idle_frac * span_k
            e_pos = e > 0.0
            base = np.maximum(e, 1.0)
            cap = slope_cap_frac * base
            if soft_unfreeze_s > 0.0:
                soft_ok = (1.0 - np.exp(-t_edge / soft_unfreeze_s)) > 0.5
            else:
                soft_ok = t_edge > 0.0
            if guard_enable:
                # 保护期内只冻结 x2 积分（由 motion_slow_gain 控制），**不清零 load_dwell**：
                # 清零会让保护结束后还要重新等满 slow_confirm_s 才恢复积分，在高漂移率工况
                # （实测 10g 会话 pre 漂移 ~20 ADC/s）下保护窗反复触发 → 积分被反复推迟 →
                # 漂移反而压不住（实测漂移 182 → 1257 ADC）。冻结积分本身已足够阻止
                # 「短时间内被当作慢漂移吸收」，无需再动受载确认计时。
                pass
            dwell_ok = load_dwell >= slow_confirm_s
            gate_ok = np.abs(slope) < slope_gate_frac * base
            dx2 = np.clip(slope - dx1_rate, -cap, cap)
            if guard_enable:
                dx2 = dx2 * np.where(guard_on, motion_slow_gain, 1.0)
            x_slow = x_slow + np.where(e_pos & dwell_ok & soft_ok & gate_ok, dt * dx2, 0.0)
            x_slow = np.where(e_pos, np.minimum(np.maximum(x_slow, 0.0), r_slow_max * base), x_slow)
            release_mask = (~e_pos) | idle_k
            tau_r2 = np.where(idle_k & (tau_r_slow_idle_s > 0.0), tau_r_slow_idle_s, tau_r_slow_s)
            tau_r2 = np.where(tau_r2 > _DENOM_FLOOR, tau_r2, _DENOM_FLOOR)
            x_slow = np.where(release_mask,
                              np.maximum(x_slow - (dt / tau_r2) * x_slow, 0.0), x_slow)

            # ── K6 预留池（cpp:182-199）──
            if hold_eps > 0.0:
                slope_a = (v - v_fast_lp) / hold_tau_s
                v_fast_lp = v_fast_lp + (dt / hold_tau_s) * (v - v_fast_lp)
                d_des = x_fast + x_slow
                a_up = applied + dt * np.maximum(slope_a + hold_eps, 0.0)
                a_dn = applied + dt * (slope_a - hold_eps)
                a = np.where(d_des > applied, np.minimum(d_des, a_up), np.maximum(d_des, a_dn))
                applied = np.where(e_pos,
                                   np.minimum(np.maximum(a, 0.0), np.maximum(d_des, 0.0)), d_des)

        if bypass_frame:
            applied = x_fast + x_slow
            bypass_mask[index] = True
            out[index] = raw
        elif hold_eps > 0.0:
            out[index] = raw - applied
        else:
            out[index] = raw - x_fast - x_slow

        marker = index + 1
        if marker % _PROGRESS_STRIDE == 0:
            if cancel is not None and cancel():
                raise Cancelled()
            if progress is not None:
                progress(marker / float(n_rows), "%d/%d 帧" % (marker, n_rows))

    if cancel is not None and cancel():
        raise Cancelled()
    if progress is not None:
        progress(1.0, "%d/%d 帧" % (n_rows, n_rows))
    nan_runs = _nan_runs(finite_mask, time_axis)
    stats = {"bypass_mask": bypass_mask, "guard_mask": guard_mask, "nan_runs": nan_runs,
             "nan_frames": int((~finite_mask).sum()), "guard_frames": int(guard_mask.sum())}
    return out, stats


class Algorithm:
    def run(self, ctx, params, progress=None, cancel=None):
        values = np.asarray(ctx.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("K12 需要 dataset 模式的二维全通道矩阵，实际 shape=%r" % (values.shape,))
        n_rows, n_ch = int(values.shape[0]), int(values.shape[1])
        time_axis = np.asarray(ctx.time_axis, dtype=np.float64).reshape(-1)
        if time_axis.size != n_rows:
            raise ValueError("时间轴长度 %d 与帧数 %d 不一致" % (time_axis.size, n_rows))
        ids = tuple(int(item) for item in (ctx.channel_ids if ctx.channel_ids is not None else ()))
        if len(ids) != n_ch:
            ids = tuple(range(n_ch))
        if cancel is not None and cancel():
            raise Cancelled()

        out, stats = _compensate(values, time_axis, params, progress=progress, cancel=cancel)

        total = out.sum(axis=1)
        outputs = [AlgorithmOutput(key="total", label="K12 总值",
                                   y=np.ascontiguousarray(total), unit="")]
        outputs.extend(AlgorithmOutput(key="ch%d" % channel_id, label="K12 ch%d" % channel_id,
                                       y=np.ascontiguousarray(out[:, column]), unit="")
                       for column, channel_id in enumerate(ids))

        coerce = _coerce_params(params)
        note_parts: list = []
        mode_hint = ""
        meta = ctx.meta if isinstance(ctx.meta, dict) else {}
        for key in ("time_mode", "time_axis_mode"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                mode_hint = value.strip()
                break
        if mode_hint:
            note_parts.append("time=%s" % mode_hint)
        if not coerce["motion_guard_enable"]:
            note_parts.append("保护层关闭（=K9）")
        else:
            note_parts.append("保护触发 %d 帧（%.2f%%）"
                              % (stats["guard_frames"],
                                 100.0 * stats["guard_frames"] / max(1, n_rows)))
        runs = stats["nan_runs"]
        if runs:
            longest = max(item[2] for item in runs)
            note_parts.append("NaN 截段 %d 处（最长 %.1f s）" % (len(runs), longest))
        return AlgorithmResult(outputs=outputs, note="；".join(note_parts),
                               params_used=dict(params) if isinstance(params, dict) else {})


__all__ = ["PLUGIN_META", "param_specs", "Algorithm"]
