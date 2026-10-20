# -*- coding: utf-8 -*-
"""端到端串扰压制工具。

用法示例::

    # 1) 用标定录制训练偏置表（可选，偏置模式必需）
    python run_suppress.py calib --raw "..\\raw_data" --out bias_table.json

    # 2) 处理单个录制（直接置零模式）
    python run_suppress.py run --csv <device_001_seg000.csv> --out out\\suppressed.csv

    # 3) 处理整个 raw_data 目录，并输出统计
    python run_suppress.py batch --raw "..\\raw_data" --out out\\batch --mode bias

设计要点见 crosstalk_suppress.py 顶部注释。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response, response_with_baseline, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings, load_recording  # noqa: E402
from crosstalk_suppress import BiasTable, CrosstalkSuppressor, SuppressConfig  # noqa: E402


def match_table_key(tables: dict, label: str):
    """把录制 label 匹配到偏置表 key。

    兼容两种目录层级与设备目录名后缀，且必须「类别 + 设备名」同时命中：
        label = "大拇指指腹/device1-2026-10-10_17-07-39"
        key   = "大拇指指腹/device1" 或 "大拇指指腹/device1-2026-10-10_17-07-39"
    """
    parts = label.split("/")
    if len(parts) < 2:
        return None
    cls, tail = parts[0], parts[-1]
    dev = tail.split("-")[0]
    for k in tables:
        kparts = k.split("/")
        if len(kparts) < 2:
            continue
        if kparts[0] == cls and kparts[-1].split("-")[0] == dev:
            return k
    return None


# --------------------------------------------------------------------------- #
# 偏置表标定
# --------------------------------------------------------------------------- #

def calibration_frames(rec, frames, min_peak_ratio=0.25, active_q=99.5):
    """选取"串扰主导"的帧用于标定：响应超过阈值、且形态由主点单点主导。

    判据只要求「主点在正交方向干净」，主方向沿线的一切（含串扰）都进入标定。
    """
    P_all = np.nanmax(frames, axis=(1, 2))
    P_f = np.where(np.isfinite(P_all), P_all, -np.inf)
    thr = min_peak_ratio * np.nanpercentile(P_all, active_q)
    sel = []
    for t in range(frames.shape[0]):
        if P_f[t] < thr:
            continue
        sel.append(t)
    return sel, thr


def cmd_calib(args):
    tables = {}
    for rec in iter_all_recordings(args.raw):
        d = response(rec.data)
        frames = to_frames(rec, d)
        sel, thr = calibration_frames(rec, frames)
        sup = CrosstalkSuppressor(rec).calibrate(d)
        # 自动判定主方向
        e_row = e_col = 0.0
        for t in sel:
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            P = f[r0, c0]
            e_row += np.nansum(np.delete(f[r0, :], c0)) / P
            e_col += np.nansum(np.delete(f[:, c0], r0)) / P
        direction = "row" if e_row >= e_col else "col"
        bt = BiasTable(direction, rec.rows, rec.cols).fit(frames, sel)
        tables[rec.label] = bt.to_json()
        print(f"{rec.label}: 方向={direction} 标定帧={len(sel)} "
              f"线级条目={len(bt.line_stats)} 汇总条目={len(bt.global_stats)}", file=sys.stderr)
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=1)
    print(f"偏置表已写入 {args.out}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# 单录制处理
# --------------------------------------------------------------------------- #

def process_recording(rec, cfg, bias_table=None, report=None):
    """返回 (sup, frames_resp, y_out_resp, infos, base)。

    注意：`frames_resp` / `y_out_resp` 都在**响应域**（已扣基线），
    写回数据域必须加回 `base`（见 write_csv）。
    """
    d, base = response_with_baseline(rec.data)
    frames = to_frames(rec, d)
    sup = CrosstalkSuppressor(rec, cfg, bias=bias_table)
    sup.calibrate(d)
    if cfg.direction == "auto" and bias_table is not None:
        cfg.direction = bias_table.direction          # 用标定方向，避免逐帧抖动
    y_out, infos = sup.process_batch(d)
    if report is not None:
        report(rec, sup, frames, y_out, infos)
    return sup, frames, y_out, infos, base


def write_csv(src_csv: str, y_out_resp: np.ndarray, base: np.ndarray, dst_csv: str, rec):
    """把压制结果写回与输入同构的 CSV。

    头部与时间列原样保留；数据区 = 响应域结果 + 基线，使未被改动的位置
    与输入逐位一致，被压制的位置恢复到该通道的静息底值（而不是 0）。
    """
    with io.open(src_csv, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    data_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("##Data"):
            data_start = i + 2
            break
    assert data_start is not None
    idx = rec.index_map
    out_lines = lines[:data_start]
    body = lines[data_start:]
    for k, ln in enumerate(body):
        if not ln.strip():
            continue
        parts = ln.split(",")
        cells = parts[:3]
        vals = []
        for r in range(rec.rows):
            for c in range(rec.cols):
                j = idx[r, c]
                if j < 0:
                    continue
                v = y_out_resp[k, r, c]
                if not np.isfinite(v):
                    vals.append(0.0)
                else:
                    vals.append(float(v) + float(base[k, j]))
        out_lines.append(",".join(cells + [f"{v:.6f}" for v in vals]))
    with io.open(dst_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def cmd_run(args):
    rec = load_recording(args.csv)
    cfg = SuppressConfig(direction=args.direction, mode=args.mode,
                         compensate=args.compensate, min_peak_ratio=args.min_peak_ratio)
    bt = None
    label = os.path.relpath(args.csv, args.raw).replace("\\", "/") if args.raw else None
    if args.bias_table and os.path.exists(args.bias_table):
        with io.open(args.bias_table, "r", encoding="utf-8") as f:
            tables = json.load(f)
        key = match_table_key(tables, label) if label else None
        if key is None:
            key = next(iter(tables))
            print(f"⚠ 未匹配到偏置表条目，回退使用 [{key}]", file=sys.stderr)
        bt = BiasTable.from_json(tables[key])
        print(f"使用偏置表 [{key}] 方向={bt.direction}", file=sys.stderr)

    sup, frames, y_out, infos, base = process_recording(rec, cfg, bt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_csv(args.csv, y_out, base, args.out, rec)
    s = sup.stats
    print(f"{rec.label}: 帧={s['frames']} 单点压制={s['single']} 大面积放行={s['passthrough_area']} "
          f"形态不明={s['passthrough_ambig']} 静默={s['passthrough_low']} "
          f"压制点={s['suppressed_points']} 保留点={s['kept_points']} "
          f"方向={s['by_direction']} → {args.out}", file=sys.stderr)


def cmd_batch(args):
    os.makedirs(args.out, exist_ok=True)
    tables = {}
    if args.bias_table and os.path.exists(args.bias_table):
        with io.open(args.bias_table, "r", encoding="utf-8") as f:
            tables = json.load(f)
    for rec in iter_all_recordings(args.raw):
        cfg = SuppressConfig(direction=args.direction, mode=args.mode,
                             min_peak_ratio=args.min_peak_ratio)
        bt = None
        key = match_table_key(tables, rec.label)
        if key is not None:
            bt = BiasTable.from_json(tables[key])
        sup, frames, y_out, infos, base = process_recording(rec, cfg, bt)
        s = sup.stats
        # 保护校验：大面积/静默帧必须与输入完全一致
        changed = 0
        for t, i in enumerate(infos):
            if i["mode"] in ("single",):
                continue
            a, b = frames[t], y_out[t]
            m = np.isfinite(a) & np.isfinite(b)
            if np.any(np.abs(a[m] - b[m]) > 1e-9):
                changed += 1
        print(f"{rec.label:>18s} 帧={s['frames']:6d} 单点压制={s['single']:6d} "
              f"面积放行={s['passthrough_area']:5d} 不明={s['passthrough_ambig']:5d} "
              f"静默={s['passthrough_low']:6d} 压制点={s['suppressed_points']:6d} "
              f"非单点帧改动={changed} 方向={s['by_direction']}", file=sys.stderr)
    print("\n完成。", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="触觉阵列单点串扰压制")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calib", help="从 raw_data 标定串扰偏置表")
    c.add_argument("--raw", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_calib)

    r = sub.add_parser("run", help="处理单个 CSV")
    r.add_argument("--csv", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--raw", default=None, help="raw_data 根目录（用于匹配偏置表 key）")
    r.add_argument("--bias-table", default=None)
    r.add_argument("--direction", default="auto", choices=["auto", "row", "col", "off"])
    r.add_argument("--mode", default="auto", choices=["auto", "zero", "bias", "subtract"])
    r.add_argument("--compensate", default="keep", choices=["keep", "sum"])
    r.add_argument("--min-peak-ratio", type=float, default=0.25)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="处理整个 raw_data")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--bias-table", default=None)
    b.add_argument("--direction", default="auto", choices=["auto", "row", "col", "off"])
    b.add_argument("--mode", default="auto", choices=["auto", "zero", "bias", "subtract"])
    b.add_argument("--min-peak-ratio", type=float, default=0.25)
    b.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
