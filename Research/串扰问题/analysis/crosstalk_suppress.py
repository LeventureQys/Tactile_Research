# -*- coding: utf-8 -*-
"""串扰压制算法（横纵双向通用）。

观测规律（来自 raw_data 实测统计，见 analysis/out/）：
  1. 串扰只沿阵列的一个几何方向传播：
        · 大拇指指腹 —— 横向（同一行内、相邻列之间），比例 10%~45%
        · 四指指腹   —— 纵向（同一列内、相邻行之间），比例 1%~8%（个别位置 ~20%）
     正交方向上的耦合始终在噪声级（< 0.5%）。
  2. 串扰与主点在时间上严格同步（互相关峰值在 lag=0）。
  3. 串扰幅值随主点幅值增长很弱（近似固定偏置），因此"抑制"以置零/限幅最稳。
  4. 串扰比例与"主点位置"强相关，同一设备不同位置的串扰图案不同。

因此压制策略是「单点模式判定 + 串扰线置零」：
  * 一帧内响应若集中在「单一行」或「单一列」
        - 只有 1 行活跃、多列活跃  →  横向串扰  →  该行内除主点外全部压制
        - 只有 1 列活跃、多行活跃  →  纵向串扰  →  该列内除主点外全部压制
  * 若多行多列都活跃（大面积 / 多点触发），判定为非单点模式，原样输出，不做任何改动。

该判据天然兼容两种设备，无需预先知道设备型号；也可显式指定方向。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SuppressConfig:
    """压制参数。

    注意数值域：算法全程在**响应域**工作（y = raw − baseline），因此 `rest`
    指的是"响应域的静息水平"。数据域的静息底值由写回时加回 baseline 恢复
    （见 `crosstalk_core.restore_to_raw`），两者不可混淆。
    """

    direction: str = "auto"        # 'auto' | 'row'(横向) | 'col'(纵向) | 'off'
    mode: str = "auto"             # 'auto'(=zero) | 'zero' | 'subtract' | 'bias'
    min_peak_abs: float = 0.0      # 主点绝对下限（ADC 单位）
    min_peak_ratio: float = 0.25   # 主点下限 = 该系数 × 参考量程（由 calibrate 设定）
    direction_ratio: float = 2.0   # 方向判定的显著性：#1方向能量 / #2方向能量
    area_active_frac: float = 0.25 # 「活跃行/列」判定：能量 > 该系数 × 最大行(列)能量
    area_min_rows: int = 3         # 活跃行数 >= 该值 且 活跃列数 >= area_min_cols → 视为大面积
    area_min_cols: int = 3         # 说明：门槛设为 3 而非 2，是因为"主点 + 另一处独立接触"
                                   # 并不需要保护——压制只作用于主点所在的那一条线，
                                   # 另一处接触不在线上，不会被碰到（见移植手册 §3.3）。
                                   # 真正需要拦的是"整面/面状触发"，那必然多行多列同时活跃；
                                   # 且整面时 direction_ratio 与线内保护会二次兜底。
    line_energy_max: float = 3.0   # 主方向线上非主点总能量 / 主点值 的上限，超出即放行
    line_other_max: float = 0.90   # 主方向线上单个非主点 / 主点值 的上限，超出即放行
    compensate: str = "keep"       # 'keep' 主点保持原值 | 'sum' 把被压制量并入主点
    rest: float = 0.0              # 响应域的压制目标（0 = 压到该通道的静息水平）
    use_channel_rest: bool = True  # 用 calibrate 得到的逐通道静息响应水平，而非统一 rest
    verbose: bool = False


class BiasTable:
    """串扰偏置表：bias[(line_idx, src, dst)] = 该主点位置下目标位置的响应分布。

    实测表明串扰是「加法型偏置」（幅值与主点幅值弱相关），因此标定的是响应
    的绝对值分布，而不是比例。取低分位作为偏置中心 b，取高分位作为上界 u：
        观测 y_q <= u   →  视为纯串扰，可完全压制
        观测 y_q  > u   →  超出部分视为真实接触，保留
    按 (主点所在行/列, 源位置, 目标位置) 三元组建表，样本不足时回退到
    (源位置, 目标位置) 的汇总值。
    """

    def __init__(self, direction: str, rows: int, cols: int,
                 q_lo: float = 50.0, q_hi: float = 90.0, min_n: int = 10):
        self.direction = direction          # 'row' 横向压制用同行统计；'col' 纵向
        self.rows, self.cols = rows, cols
        self.q_lo, self.q_hi, self.min_n = q_lo, q_hi, min_n
        self.line_stats: dict[tuple[int, int, int], tuple[float, float, int]] = {}
        self.global_stats: dict[tuple[int, int], tuple[float, float, int]] = {}

    # ------------------------------------------------------------------ #

    def fit(self, frames: np.ndarray, sel) -> "BiasTable":
        """frames: (T, rows, cols)；sel: 用于标定的帧序号（串扰主导的帧）。"""
        tmp: dict[tuple[int, int, int], list[float]] = {}
        gtmp: dict[tuple[int, int], list[float]] = {}
        for t in sel:
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (self.rows, self.cols))
            src = c0 if self.direction == "row" else r0
            line = r0 if self.direction == "row" else c0
            L = self.cols if self.direction == "row" else self.rows
            for j in range(L):
                if j == src:
                    continue
                v = f[r0, j] if self.direction == "row" else f[j, c0]
                if not np.isfinite(v):
                    continue
                tmp.setdefault((line, src, j), []).append(float(v))
                gtmp.setdefault((src, j), []).append(float(v))
        for k, v in tmp.items():
            if len(v) >= self.min_n:
                a = np.asarray(v)
                self.line_stats[k] = (float(np.percentile(a, self.q_lo)),
                                      float(np.percentile(a, self.q_hi)), len(v))
        for k, v in gtmp.items():
            a = np.asarray(v)
            self.global_stats[k] = (float(np.percentile(a, self.q_lo)),
                                    float(np.percentile(a, self.q_hi)), len(v))
        return self

    def lookup(self, r0: int, c0: int, j: int):
        """返回 (偏置中心 b, 串扰上界 u)；无样本返回 (None, None)。"""
        if self.direction == "row":
            src, line = c0, r0
        else:
            src, line = r0, c0
        hit = self.line_stats.get((line, src, j))
        if hit is None:
            hit = self.global_stats.get((src, j))
        if hit is None:
            return None, None
        return hit[0], hit[1]

    # ------------------------------------------------------------------ #

    def to_json(self) -> dict:
        return dict(direction=self.direction, rows=self.rows, cols=self.cols,
                    q_lo=self.q_lo, q_hi=self.q_hi, min_n=self.min_n,
                    line_stats={f"{a}|{b}|{c}": v for (a, b, c), v in self.line_stats.items()},
                    global_stats={f"{a}|{b}": v for (a, b), v in self.global_stats.items()})

    @classmethod
    def from_json(cls, obj: dict) -> "BiasTable":
        t = cls(obj["direction"], obj["rows"], obj["cols"], obj["q_lo"], obj["q_hi"], obj["min_n"])
        for k, v in obj.get("line_stats", {}).items():
            a, b, c = (int(x) for x in k.split("|"))
            t.line_stats[(a, b, c)] = (v[0], v[1], v[2])
        for k, v in obj.get("global_stats", {}).items():
            a, b = (int(x) for x in k.split("|"))
            t.global_stats[(a, b)] = (v[0], v[1], v[2])
        return t


class CrosstalkSuppressor:
    """逐帧串扰压制器。

    用法::

        sup = CrosstalkSuppressor(rec)
        sup.calibrate(d_response)          # 可选：用一段数据估计噪声底/量程
        for t in range(T):
            y = sup.frame(d_response[t])   # 铺成 (rows, cols)
            y_out, info = sup.process(y)
    """

    def __init__(self, rec, config: SuppressConfig | None = None, bias: "BiasTable | None" = None):
        self.rec = rec
        self.cfg = config or SuppressConfig()
        self.bias = bias
        self.grid = rec.grid
        self.idx = rec.index_map
        self.rows, self.cols = rec.rows, rec.cols
        # 行/列的有效格数（用于能量归一化，避免列长不同造成偏差）
        self.row_n = np.array([np.sum(self.grid[r, :]) for r in range(self.rows)], dtype=float)
        self.col_n = np.array([np.sum(self.grid[:, c]) for c in range(self.cols)], dtype=float)
        self.ref_scale = 0.0
        self.noise = 0.0
        self.rest_ch = None            # (N_active,) 逐通道静息响应水平
        self.rest_frame = None         # (rows, cols) 同上，阵列化
        self.stats = dict(frames=0, single=0, suppressed_points=0, suppressed_energy=0.0,
                          kept_points=0, kept_energy=0.0,
                          passthrough_area=0, passthrough_low=0, passthrough_ambig=0,
                          by_direction={"row": 0, "col": 0})

    # ------------------------------------------------------------------ #
    # 标定
    # ------------------------------------------------------------------ #

    def calibrate(self, d: np.ndarray, quiet_q: float = 30.0, peak_q: float = 99.5):
        """用一段响应数据估计噪声底、参考量程与逐通道静息水平。

        d: (T, N_active) 已完成基线扣除的响应。
        """
        frames = np.full((d.shape[0], self.rows, self.cols), np.nan)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.idx[r, c] >= 0:
                    frames[:, r, c] = d[:, self.idx[r, c]]
        peak = np.nanmax(frames, axis=(1, 2))
        self.ref_scale = float(np.nanpercentile(peak, peak_q))
        # 噪声底：取"无触发帧"（峰值很低）的非主点响应的高分位
        quiet_mask = peak <= np.nanpercentile(peak, quiet_q)
        quiet = frames[quiet_mask]
        if quiet.size:
            self.noise = float(np.nanpercentile(quiet, 99))
        else:
            self.noise = 0.0
        # 逐通道静息响应水平：无触发帧上该通道响应的中位数，
        # 用于把被压制的位置恢复到"该通道本来该有的样子"，而不是一律归零。
        if quiet_mask.any():
            self.rest_ch = np.median(d[quiet_mask], axis=0)
        else:
            self.rest_ch = np.zeros(d.shape[1])
        self.rest_frame = np.full((self.rows, self.cols), np.nan)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.idx[r, c] >= 0:
                    self.rest_frame[r, c] = self.rest_ch[self.idx[r, c]]
        return self

    def _rest_at(self, r: int, c: int) -> float:
        """该格在响应域的静息目标值。"""
        if self.cfg.use_channel_rest and self.rest_frame is not None:
            v = self.rest_frame[r, c]
            if np.isfinite(v):
                return float(v)
        return float(self.cfg.rest)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #

    def frame(self, d_row: np.ndarray) -> np.ndarray:
        """把 (N_active,) 响应铺成 (rows, cols)。"""
        y = np.full((self.rows, self.cols), np.nan)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.idx[r, c] >= 0:
                    y[r, c] = d_row[self.idx[r, c]]
        return y

    def process(self, y: np.ndarray):
        """处理一帧。y 为 (rows, cols)，非活跃格为 nan。

        返回 (y_out, info)；info['mode'] ∈ {'low','single','area','ambiguous'}。
        """
        cfg = self.cfg
        y_out = np.array(y, dtype=float, copy=True)
        info = dict(mode="low", direction=None, peak=0.0, r0=-1, c0=-1,
                    suppressed=0, suppressed_energy=0.0, kept=0, e_row=0.0, e_col=0.0,
                    active_rows=0, active_cols=0, crosstalk=None)
        self.stats["frames"] += 1

        finite = np.isfinite(y)
        if not np.any(finite):
            return y_out, info
        P = float(np.nanmax(y))
        info["peak"] = P
        thr_abs = cfg.min_peak_abs + cfg.min_peak_ratio * self.ref_scale
        if not np.isfinite(P) or P < max(thr_abs, 3.0 * self.noise):
            self.stats["passthrough_low"] += 1
            return y_out, info

        r0, c0 = np.unravel_index(np.nanargmax(y), (self.rows, self.cols))
        info["r0"], info["c0"] = int(r0), int(c0)

        # ---- 1) 大面积 / 多点保护：整帧是否在 ≥2 行且 ≥2 列上都有显著能量 ----
        row_energy = np.array([np.nansum(y[r, :]) / max(self.row_n[r], 1) for r in range(self.rows)])
        col_energy = np.array([np.nansum(y[:, c]) / max(self.col_n[c], 1) for c in range(self.cols)])
        ar = int(np.sum(row_energy > cfg.area_active_frac * np.nanmax(row_energy)))
        ac = int(np.sum(col_energy > cfg.area_active_frac * np.nanmax(col_energy)))
        info["active_rows"], info["active_cols"] = ar, ac
        if ar >= cfg.area_min_rows and ac >= cfg.area_min_cols:
            info["mode"] = "area"
            self.stats["passthrough_area"] += 1
            return y_out, info

        # ---- 2) 方向判定：比较主点所在行 / 列上的"其它格"能量 ----
        row_line = y[r0, :]
        col_line = y[:, c0]
        row_mask = np.isfinite(row_line)
        row_mask[c0] = False
        col_mask = np.isfinite(col_line)
        col_mask[r0] = False
        e_row = float(np.nansum(row_line[row_mask]))
        e_col = float(np.nansum(col_line[col_mask]))
        info["e_row"], info["e_col"] = e_row, e_col

        if cfg.direction in ("row", "col"):
            direction = cfg.direction
        elif cfg.direction == "off":
            direction = None
        else:
            hi, lo = max(e_row, e_col), min(e_row, e_col)
            if hi <= 1e-9:
                direction = None
            elif hi < cfg.direction_ratio * max(lo, 1e-9):
                direction = None          # 两个方向能量相当 → 形态不明确 → 放行
            else:
                direction = "row" if e_row > e_col else "col"

        if direction is None:
            info["mode"] = "ambiguous"
            self.stats["passthrough_ambig"] += 1
            return y_out, info
        info["direction"] = direction

        # ---- 3) 取主方向线并做"多点"复核 ----
        if direction == "row":
            line, mask, key_self = row_line, row_mask, c0
        else:
            line, mask, key_self = col_line, col_mask, r0
        others = line[mask]
        s_others = float(np.nansum(others))
        p_other_max = float(np.nanmax(others)) if others.size else 0.0
        info["line_others_sum"] = s_others
        info["line_other_max"] = p_other_max

        if cfg.line_energy_max > 0 and s_others > cfg.line_energy_max * P:
            info["mode"] = "area"
            self.stats["passthrough_area"] += 1
            return y_out, info
        if cfg.line_other_max > 0 and p_other_max > cfg.line_other_max * P:
            info["mode"] = "area"
            self.stats["passthrough_area"] += 1
            return y_out, info

        # ---- 4) 压制：主方向线上除主点外逐格处理 ----
        # 语义说明：这里全部在"响应域"操作。压制的目标不是 0，而是
        # 「该通道在没有串扰时本该有的响应水平」——即逐通道静息水平 rest。
        # 写回数据域时再加回 baseline，得到该通道正常的静息底值。
        have_bias = self.bias is not None and self.bias.direction == direction
        mode = cfg.mode
        if mode == "auto":
            mode = "zero"      # 默认彻底清除线内串扰，恢复到该通道静息水平；
                               # 需要"保真优先"时显式选 'subtract' / 'bias'

        suppressed_map = np.zeros((self.rows, self.cols))
        idxs = range(self.cols) if direction == "row" else range(self.rows)
        for j in idxs:
            if j == (c0 if direction == "row" else r0):
                continue
            v = line[j]
            if not np.isfinite(v):
                continue
            rr, cc = (r0, j) if direction == "row" else (j, c0)
            rest = self._rest_at(rr, cc)
            b = u = None
            if have_bias:
                b, u = self.bias.lookup(r0, c0, j)

            if mode in ("subtract", "bias") and b is not None:
                if mode == "bias" and u is not None and v <= u:
                    # 未超出串扰上界：纯串扰 → 恢复到静息水平
                    out_val = rest
                else:
                    # 减去标定偏置，保留超出部分（可能是真实接触）
                    out_val = max(v - b, rest)
                    if out_val > rest + 1e-9:
                        info["kept"] = info.get("kept", 0) + 1
                        self.stats["kept_points"] += 1
                        self.stats["kept_energy"] += out_val
            else:
                out_val = rest

            if direction == "row":
                suppressed_map[r0, j] = v
                if cfg.compensate == "sum":
                    y_out[r0, c0] += max(v - rest, 0.0)
                y_out[r0, j] = out_val
            else:
                suppressed_map[j, c0] = v
                if cfg.compensate == "sum":
                    y_out[r0, c0] += max(v - rest, 0.0)
                y_out[j, c0] = out_val
            info["suppressed"] += 1
            info["suppressed_energy"] += float(v)

        info["mode"] = "single"
        info["direction"] = direction
        info["crosstalk"] = suppressed_map
        self.stats["single"] += 1
        self.stats["suppressed_points"] += info["suppressed"]
        self.stats["suppressed_energy"] += info["suppressed_energy"]
        self.stats["by_direction"][direction] += 1
        return y_out, info

    # ------------------------------------------------------------------ #
    # 批量处理
    # ------------------------------------------------------------------ #

    def process_batch(self, d: np.ndarray):
        """d: (T, N_active) 响应矩阵。返回 (y_out (T, rows, cols), infos list)。"""
        T = d.shape[0]
        out = np.full((T, self.rows, self.cols), np.nan)
        infos = []
        for t in range(T):
            y = self.frame(d[t])
            yo, info = self.process(y)
            out[t] = yo
            infos.append(info)
        return out, infos


def grid_to_row(rec, y: np.ndarray) -> np.ndarray:
    """(rows, cols) -> (N_active,)，nan -> 0。"""
    out = np.zeros(rec.n_active)
    idx = rec.index_map
    for r in range(rec.rows):
        for c in range(rec.cols):
            if idx[r, c] >= 0:
                v = y[r, c]
                out[idx[r, c]] = 0.0 if not np.isfinite(v) else v
    return out
