# -*- coding: utf-8 -*-
"""串扰压制核心 —— 无第三方依赖的移植蓝本。

设计约束（为了能逐行翻译成 C / C++ / Rust / Java）：
    * 不使用 numpy、不使用任何第三方库，只用定长 list 与标量运算
    * 不动态分配内存：所有缓冲在构造时按最大尺寸一次性分配
    * 全部在一维数组上操作（长度 = 有效通道数），宿主负责铺回/取出二维阵列
    * 无随机、无递归、无异常控制流；单帧计算量为 O(rows*cols)

数据约定（务必与宿主系统对齐）：
    输入/输出都是**响应域**值，即 resp = raw - baseline。
    宿主负责：
        1) 维护 baseline（见手册"基线/静息估计"一节）
        2) 调用方在使用输出前自行加回 baseline：raw_out = resp_out + baseline
    rest[] 是"该通道静息时的响应水平"，由标定阶段给出。

通道顺序：与录制 CSV 中"单元0..单元N-1"的列顺序完全一致，
          即按行优先遍历 layout_mask 为 '1' 的格子。
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #

class Params:
    """全部可调参数。默认值来自 7 台实测设备的统计结果。"""

    __slots__ = (
        "min_peak_abs",      # 主点绝对下限（响应单位）。0 = 不启用
        "min_peak_ratio",    # 主点下限 = 该系数 × ref_scale
        "direction_ratio",   # 方向显著性：#1方向能量 / #2方向能量，低于此值判为"形态不明"
        "area_active_frac",  # 活跃行/列判定阈值（相对最强行/列的能量）
        "area_min_rows",     # 活跃行数 >= 此值
        "area_min_cols",     # 且 活跃列数 >= 此值  → 判为"大面积/多点"，放行
        "line_energy_max",   # 主方向线非主点总能量 / 主点值 的上限，超出放行
        "line_other_max",    # 主方向线单个非主点 / 主点值 的上限，超出放行
        "rest_default",      # 未提供逐通道 rest 时使用的响应域压制目标
        "direction",         # 0=auto  1=row(横向)  2=col(纵向)  3=off
        "compensate_sum",    # 1 = 把被压制量并入主点（能量守恒），0 = 保持主点原值
    )

    def __init__(self):
        self.min_peak_abs = 0.0
        self.min_peak_ratio = 0.25
        self.direction_ratio = 2.0
        self.area_active_frac = 0.25
        self.area_min_rows = 3          # 见移植手册 §3.3：门槛为 3，只拦真正的面状触发
        self.area_min_cols = 3
        self.line_energy_max = 3.0
        self.line_other_max = 0.90
        self.rest_default = 0.0
        self.direction = 0
        self.compensate_sum = 0


# 与结果无关、仅用于外部读取的状态码
MODE_LOW = 0        # 未达峰值门槛，原样输出
MODE_SINGLE = 1     # 判定为单点+串扰，已压制
MODE_AREA = 2       # 大面积/多点，放行
MODE_AMBIG = 3      # 形态不明确（双轴能量相当），放行

DIR_NONE = 0
DIR_ROW = 1         # 横向：同行、列间
DIR_COL = 2         # 纵向：同列、行间


# --------------------------------------------------------------------------- #
# 主体
# --------------------------------------------------------------------------- #

class Suppressor:
    """逐帧串扰压制器。

    构造::

        sup = Suppressor(rows, cols, layout_mask, params)
        sup.set_calibration(ref_scale, noise, rest)   # 标定值（可来自产线）

    逐帧::

        info = sup.process(resp_in, resp_out)          # 就地写入 resp_out
    """

    def __init__(self, rows, cols, layout_mask, params=None):
        self.rows = rows
        self.cols = cols
        self.p = params or Params()

        # ---- 1) 展开掩码，建立通道 <-> 阵列 的双向映射 ----
        self.n = 0
        self.chan = [-1] * (rows * cols)          # 格 -> 通道号（-1 = 无）
        self.cell_row = []                        # 通道 -> 行
        self.cell_col = []                        # 通道 -> 列
        for r in range(rows):
            for c in range(cols):
                if layout_mask[r * cols + c] == "1":
                    self.chan[r * cols + c] = self.n
                    self.cell_row.append(r)
                    self.cell_col.append(c)
                    self.n += 1

        # ---- 2) 行 / 列 的通道清单（用于能量统计与线内遍历） ----
        self.row_chans = [[] for _ in range(rows)]
        self.col_chans = [[] for _ in range(cols)]
        for k in range(self.n):
            self.row_chans[self.cell_row[k]].append(k)
            self.col_chans[self.cell_col[k]].append(k)

        # ---- 3) 标定值（默认由外部覆盖） ----
        self.ref_scale = 0.0
        self.noise = 0.0
        self.rest = [self.p.rest_default] * self.n

        # 复用缓冲，避免逐帧分配
        self._row_energy = [0.0] * rows
        self._col_energy = [0.0] * cols

    # ------------------------------------------------------------------ #

    def set_calibration(self, ref_scale, noise, rest=None):
        """ref_scale: 参考量程（如全量程响应的 99.5 分位）
        noise:      响应域噪声底
        rest:       长度 = 通道数的逐通道静息响应水平；None 则用 rest_default
        """
        self.ref_scale = float(ref_scale)
        self.noise = float(noise)
        if rest is not None:
            self.rest = [float(v) for v in rest]

    # ------------------------------------------------------------------ #

    def process(self, resp_in, resp_out):
        """处理一帧。

        resp_in / resp_out: 长度 = self.n 的序列（响应域）。
        返回 info 字典（含 mode / direction / peak / 主点坐标 / 压制计数）。
        info = {
            "mode": MODE_*, "direction": DIR_*, "peak": float,
            "r0": int, "c0": int, "k0": int,
            "suppressed": int, "suppressed_energy": float,
        }
        """
        p = self.p
        n = self.n

        # 先拷贝，便于各放行分支直接返回原帧
        for k in range(n):
            resp_out[k] = resp_in[k]

        # ---- 1) 找主点 ----
        k0 = 0
        peak = resp_in[0]
        for k in range(1, n):
            if resp_in[k] > peak:
                peak = resp_in[k]
                k0 = k
        info = {"mode": MODE_LOW, "direction": DIR_NONE, "peak": peak,
                "r0": self.cell_row[k0], "c0": self.cell_col[k0], "k0": k0,
                "suppressed": 0, "suppressed_energy": 0.0}

        thr = p.min_peak_abs + p.min_peak_ratio * self.ref_scale
        if 3.0 * self.noise > thr:
            thr = 3.0 * self.noise
        if peak < thr or peak <= 0.0:
            return info

        r0 = self.cell_row[k0]
        c0 = self.cell_col[k0]

        # ---- 2) 大面积 / 多点保护 ----
        # 行能量、列能量都按"该行/列的有效格数"归一化，再取相对最强的比例
        max_re = 0.0
        for r in range(self.rows):
            s = 0.0
            for k in self.row_chans[r]:
                if resp_in[k] > 0.0:
                    s += resp_in[k]
            e = s / len(self.row_chans[r])
            self._row_energy[r] = e
            if e > max_re:
                max_re = e
        max_ce = 0.0
        for c in range(self.cols):
            s = 0.0
            for k in self.col_chans[c]:
                if resp_in[k] > 0.0:
                    s += resp_in[k]
            e = s / len(self.col_chans[c])
            self._col_energy[c] = e
            if e > max_ce:
                max_ce = e

        ar = 0
        if max_re > 0.0:
            lim = p.area_active_frac * max_re
            for r in range(self.rows):
                if self._row_energy[r] > lim:
                    ar += 1
        ac = 0
        if max_ce > 0.0:
            lim = p.area_active_frac * max_ce
            for c in range(self.cols):
                if self._col_energy[c] > lim:
                    ac += 1
        if ar >= p.area_min_rows and ac >= p.area_min_cols:
            info["mode"] = MODE_AREA
            return info

        # ---- 3) 方向判定：主点所在行 / 列的"其余格"能量 ----
        e_row = 0.0
        for k in self.row_chans[r0]:
            if k != k0 and resp_in[k] > 0.0:
                e_row += resp_in[k]
        e_col = 0.0
        for k in self.col_chans[c0]:
            if k != k0 and resp_in[k] > 0.0:
                e_col += resp_in[k]

        if p.direction == 1:
            direction = DIR_ROW
        elif p.direction == 2:
            direction = DIR_COL
        elif p.direction == 3:
            direction = DIR_NONE
        else:
            hi = e_row if e_row > e_col else e_col
            lo = e_col if e_row > e_col else e_row
            if hi <= 1e-9 or hi < p.direction_ratio * (lo if lo > 1e-9 else 1e-9):
                direction = DIR_NONE
            else:
                direction = DIR_ROW if e_row > e_col else DIR_COL

        if direction == DIR_NONE:
            info["mode"] = MODE_AMBIG
            return info
        info["direction"] = direction

        # ---- 4) 线内复核：线上能量或单点过高时，认为存在真实多点 ----
        line_chans = self.row_chans[r0] if direction == DIR_ROW else self.col_chans[c0]
        s_other = 0.0
        m_other = 0.0
        for k in line_chans:
            if k == k0:
                continue
            v = resp_in[k]
            if v > 0.0:
                s_other += v
            if v > m_other:
                m_other = v
        if p.line_energy_max > 0.0 and s_other > p.line_energy_max * peak:
            info["mode"] = MODE_AREA
            return info
        if p.line_other_max > 0.0 and m_other > p.line_other_max * peak:
            info["mode"] = MODE_AREA
            return info

        # ---- 5) 压制：主方向线上除主点外，恢复到该通道的静息响应水平 ----
        total = 0.0
        for k in line_chans:
            if k == k0:
                continue
            v = resp_in[k]
            rest = self.rest[k]
            if v > rest:
                total += v - rest
            resp_out[k] = rest
            info["suppressed"] += 1
            info["suppressed_energy"] += v
        if p.compensate_sum == 1:
            resp_out[k0] = resp_in[k0] + total

        info["mode"] = MODE_SINGLE
        return info

    # ------------------------------------------------------------------ #

    @staticmethod
    def mask_of(rec) -> str:
        """从录制对象取掩码字符串（等价于 rec.mask）。"""
        return rec.mask
