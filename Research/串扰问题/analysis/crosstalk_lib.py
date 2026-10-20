# -*- coding: utf-8 -*-
"""串扰分析基础库：加载录制片段、解析布局掩码、提供阵列化访问。

数据格式（device_001_seg000.csv）：
    前若干行为 '##Session' 键值元数据（中文键），随后一行 '##Data'，
    再一行中文表头，之后为帧数据：时间戳, 相对时间, 帧编号, 单元0..单元N-1
布局：rows x cols 的掩码字符串（行优先，长度 = rows*cols），'1' 表示该单元
      在物理阵列上存在（数据列里的单元序号只对应掩码中为 '1' 的格子）。
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Recording:
    """一段单设备录制。"""

    path: str
    label: str            # 例如 "大拇指指腹/device1"
    rows: int
    cols: int
    mask: str             # 行优先掩码字符串
    elapsed: np.ndarray   # (T,) 相对时间 s
    frame_index: np.ndarray
    data: np.ndarray      # (T, N_active) 原始值
    session_meta: dict
    device_meta: dict

    # ---- 派生 ----

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def n_active(self) -> int:
        return self.data.shape[1]

    @property
    def grid(self) -> np.ndarray:
        """(rows, cols) 布尔掩码。"""
        return np.array([c == "1" for c in self.mask], dtype=bool).reshape(self.rows, self.cols)

    @property
    def index_map(self) -> np.ndarray:
        """(rows, cols) int，元素为数据列下标，非活跃格为 -1。"""
        m = np.full((self.rows, self.cols), -1, dtype=int)
        k = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mask[r * self.cols + c] == "1":
                    m[r, c] = k
                    k += 1
        return m

    def to_grid(self, frame: np.ndarray) -> np.ndarray:
        """把一帧 (N_active,) 铺成 (rows, cols)，非活跃格为 nan。"""
        g = np.full((self.rows, self.cols), np.nan)
        m = self.index_map
        for r in range(self.rows):
            for c in range(self.cols):
                k = m[r, c]
                if k >= 0:
                    g[r, c] = frame[k]
        return g

    def grid_frames(self) -> np.ndarray:
        """(T, rows, cols) 阵列化，非活跃格 nan。"""
        out = np.full((self.n_frames, self.rows, self.cols), np.nan)
        m = self.index_map
        for r in range(self.rows):
            for c in range(self.cols):
                k = m[r, c]
                if k >= 0:
                    out[:, r, c] = self.data[:, k]
        return out

    def cell(self, r: int, c: int) -> np.ndarray:
        k = self.index_map[r, c]
        return self.data[:, k] if k >= 0 else np.array([])


def _parse_session_block(lines: list[str]) -> tuple[dict, dict]:
    """解析 ##Session 段。返回 (顶层键值, 设备段键值)。"""
    top: dict = {}
    dev: dict = {}
    for ln in lines:
        if "," not in ln:
            continue
        k, v = ln.split(",", 1)
        k, v = k.strip(), v.strip()
        if k in ("行数", "列数", "数据点数", "数据阶段", "有无原始ADC", "显示模式",
                 "录制频率", "时间基准", "行版本", "值阶段", "上报压强单位", "上报力值单位",
                 "显示压强单位", "显示力值单位", "布局掩码", "设备ID", "设备别名", "端口"):
            dev[k] = v
        else:
            top[k] = v
    return top, dev


def load_recording(csv_path: str, label: str | None = None) -> Recording:
    """读取一个 device_*_seg*.csv。"""
    with io.open(csv_path, "r", encoding="utf-8-sig") as f:
        raw_lines = f.read().splitlines()

    data_start = None
    header = None
    session_lines = []
    for i, ln in enumerate(raw_lines):
        if ln.startswith("##Data"):
            header = raw_lines[i + 1]
            data_start = i + 2
            break
        if ln.startswith("##Session"):
            continue
        session_lines.append(ln)

    assert data_start is not None, f"未找到 ##Data 段: {csv_path}"
    top, dev = _parse_session_block(session_lines)

    rows = int(dev["行数"])
    cols = int(dev["列数"])
    mask = dev["布局掩码"].strip()

    body = raw_lines[data_start:]
    arr = np.array([[float(x) for x in ln.split(",")] for ln in body if ln.strip()])
    names = [s.strip() for s in header.split(",")]
    # 时间戳,相对时间,帧编号,单元0.....
    elapsed = arr[:, 1]
    frame_index = arr[:, 2].astype(int)
    data = arr[:, 3:]

    n_active = int(np.sum([c == "1" for c in mask]))
    assert data.shape[1] == n_active, f"列数 {data.shape[1]} != 掩码活跃数 {n_active}"

    if label is None:
        label = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
    return Recording(csv_path, label, rows, cols, mask, elapsed, frame_index, data, top, dev)


def iter_all_recordings(root: str) -> Iterator[Recording]:
    """递归遍历 raw_data 下所有录制（兼容两种目录层级）。

    旧结构：raw_data/<类别>/<deviceN>/<录制会话>/device_001_seg000.csv
    新结构：raw_data/<类别>/<deviceN-时间戳>/device_001_seg000.csv
    label 统一取「相对 root 的目录路径」，例如 "四指指腹/device1-2026-10-10_17-14-00"。
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        csvs = sorted(fn for fn in filenames if fn.endswith(".csv"))
        if not csvs:
            continue
        rel = os.path.relpath(dirpath, root)
        label = rel.replace(os.sep, "/")
        for fn in csvs:
            yield load_recording(os.path.join(dirpath, fn), label)


def short_label(label: str) -> str:
    """"四指指腹/device1-2026-10-10_17-14-00" -> "四指指腹/device1"。"""
    parts = label.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[-1].split('-')[0]}"
    return label


def print_grid(rec: Recording, values: np.ndarray | None = None, fmt: str = "{:7.2f}") -> str:
    """把 (rows, cols) 数组打印成对齐文本（nan 显示为 .）。"""
    if values is None:
        values = np.where(rec.grid, 1.0, np.nan)
        fmt = "{:7.0f}"
    lines = []
    for r in range(rec.rows):
        cells = []
        for c in range(rec.cols):
            v = values[r, c]
            cells.append("      ." if np.isnan(v) else fmt.format(v))
        lines.append(" ".join(cells))
    return "\n".join(lines)
