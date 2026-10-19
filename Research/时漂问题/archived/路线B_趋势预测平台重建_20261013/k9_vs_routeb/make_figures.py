# -*- coding: utf-8 -*-
"""K9 vs 路线B 对比出图 —— 全部走数据解析工具（dptool）的绘图出口。

产物（figures/）：
  fig01~06  <会话>_总力对比.png            overlay：5 条曲线（未补偿/录制/两种 K9/路线B）
  fig07~12  <会话>_残余偏移对比.png        overlay：5 秒窗口中值偏移，0 = 加载后 15 s
  fig13     六份残余偏移总览.png           小倍数：6 面板 × 4 方法
  fig14~15  <10g 会话>_台阶局部.png        overlay：小台阶局部放大（考察是否被慢态吞掉）

数据来自 compare_k9_vs_routeb.py 落下的 compare_metrics.json 与 series/*.npz。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
if str(TOOLBOX) not in sys.path:
    sys.path.insert(0, str(TOOLBOX))

from dptool import merge as M                                  # noqa: E402
from dptool.figure_export import export_small_multiples        # noqa: E402
from dptool.snapshot import CurveSnapshot, FigureSnapshot, PanelSnapshot  # noqa: E402

FIG = HERE / "figures"
PALETTE = {                                                   # 与 snapshot.PALETTE 无关，自己定
    "pre": ("未补偿输入 pre", "#7f7f7f", "solid", 1.0),
    "out_rec": ("录制输出（现役 K9 r=0.1）", "#1f77b4", "solid", 1.0),
    "k9_rec": ("K9 复算 r=0.1", "#2ca02c", "dash", 1.0),
    "k9_0p4": ("K9 单改 r=0.4", "#ff7f0e", "solid", 1.6),
    "routeb": ("路线B r=0.4（含事件保护）", "#d62728", "solid", 1.6),
}
SERIES_ORDER = ("pre", "out_rec", "k9_rec", "k9_0p4", "routeb")
PREFIXES = {"pre": "src_pre", "out_rec": "src_outrec", "k9_rec": "src_k9rec",
            "k9_0p4": "src_k90p4", "routeb": "src_routeb"}
NAME2STYLE = {PALETTE[m][0]: PALETTE[m] for m in PALETTE}


def rolling_median(t: np.ndarray, y: np.ndarray, width: float = 5.0,
                   step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    centers, values = [], []
    for start in np.arange(float(t[0]), float(t[-1]) - width + 1e-9, step):
        chunk = y[(t >= start) & (t < start + width)]
        if chunk.size:
            centers.append(start + width / 2.0)
            values.append(float(np.median(chunk)))
    return np.asarray(centers), np.asarray(values)


def build_table(method: str, x: np.ndarray, y: np.ndarray,
                frame: np.ndarray) -> M.SourceTable:
    label, _color, _style, _width = PALETTE[method]
    return M.SourceTable.from_arrays(
        label, ["total"], y.reshape(-1, 1),
        prefix=PREFIXES[method], stream=M.STREAM_PROCESSED,
        path=str(HERE / "virtual" / ("%s.csv" % label)),
        time={"frame_index": frame, "elapsed": x})


def overlay_figure(entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
                   out_path: Path, *, suptitle: str, y_label: str,
                   xlim=None, ylim=None, smooth_s: float = 0.0) -> dict:
    """entries = [(method, frame, x, y)]，用 dptool 的 overlay 模式出图。"""
    sources = [build_table(method, x, y, frame) for method, frame, x, y in entries]
    table = M.merge_sources(sources, align="frame_index")
    snap = M.figure_snapshot(table, mode="overlay", series_by="file", signal="sum",
                             smooth_s=smooth_s, width_in=17.0, panel_height_in=6.2,
                             ncol=1, xlim=xlim, ylim=ylim)
    panel = snap.panels[0]
    panel.title = suptitle
    panel.y_label = y_label
    panel.x_label = "录制相对时间 elapsed (s)"
    for curve in panel.curves:
        style = NAME2STYLE.get(curve.name)
        if style:
            _label, color, line_style, width = style
            curve.color = color
            curve.style = line_style
            curve.width = width
    snap.suptitle = ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return export_small_multiples(snap, str(out_path), dpi=110, width_in=17.0,
                                  panel_height_in=6.2, check=True)


def figures() -> int:
    payload = json.loads((HERE / "compare_metrics.json").read_text(encoding="utf-8"))
    FIG.mkdir(exist_ok=True)
    made: list[str] = []

    panels_all = []
    for index, report in enumerate(payload["sessions"], start=1):
        label = report["label"].replace("/", "_")
        loaded = np.load(HERE / "series" / ("%s.npz" % label))
        t, frame = loaded["t"], loaded["frame"]
        load = report["load_time_s"]
        totals = {m: loaded["total_" + m] for m in SERIES_ORDER}

        # figA：原始总力对比
        entries = [(m, frame, t, totals[m]) for m in SERIES_ORDER]
        path = FIG / ("fig%02d_%s_总力对比.png" % (index, label))
        overlay_figure(entries, path,
                       suptitle="%s ｜ 总力对比（加载 %.1f s 起算，K9 与路线B 共用 57 通道）"
                                % (report["id"], load),
                       y_label="总力（57 通道求和, ADC）")
        made.append(path.name)

        # figB：5 秒窗口中值残余偏移
        entries = []
        residual: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in SERIES_ORDER:
            x, y = rolling_median(t, totals[method], 5.0, 0.5)
            base = float(np.median(totals[method][(t >= load + 15.) & (t < load + 20.)]))
            y = y - base
            residual[method] = (x, y)
            entries.append((method, np.arange(x.size), x, y))
        path = FIG / ("fig%02d_%s_残余偏移对比.png" % (index + 6, label))
        overlay_figure(entries, path,
                       suptitle="%s ｜ 5 秒窗口中值残余偏移（0 = 加载后 15 s，绝对值越小越好）"
                                % report["id"],
                       y_label="相对加载后基线偏移 (ADC)")
        made.append(path.name)

        # 汇总面板（用加载后时间轴）
        curves = []
        for method in ("out_rec", "k9_rec", "k9_0p4", "routeb"):
            x, y = residual[method]
            mask = x >= load + 15.
            _label, color, style, width = PALETTE[method]
            curves.append(CurveSnapshot(name=_label, x=x[mask] - load, y=y[mask],
                                        color=color, style=style, width=width))
        m = report["methods"]
        panels_all.append(PanelSnapshot(
            title="%s ｜ 净变化 pre=%+.0f / K9.1=%+.0f / K9.4=%+.0f / B=%+.0f ADC"
                  % (report["id"], m["pre"]["net_change_adc"], m["k9_rec"]["net_change_adc"],
                     m["k9_0p4"]["net_change_adc"], m["routeb"]["net_change_adc"]),
            curves=curves, x_label="加载后时间 (s)", y_label="残余偏移 (ADC)",
            show_legend=(index == 1)))

    summary = FigureSnapshot(
        suptitle="右拇指指腹 6 份录制 ｜ K9 现役/提高预算/路线B 残余偏移（5 s 窗口中值，0 = 加载后 15 s）",
        panels=panels_all, ncol=2, width_in=20.0, panel_height_in=3.4, dpi=110,
        legend_panel=0)
    path = FIG / "fig13_六份残余偏移总览.png"
    export_small_multiples(summary, str(path), dpi=110, width_in=20.0,
                           panel_height_in=3.4, check=True)
    made.append(path.name)

    # 10g 局部放大：小台阶是否被慢态吞掉
    for index, report in enumerate(payload["sessions"]):
        if "10g" not in report["id"]:
            continue
        label = report["label"].replace("/", "_")
        loaded = np.load(HERE / "series" / ("%s.npz" % label))
        t, frame = loaded["t"], loaded["frame"]
        total_pre = loaded["total_pre"]
        total_b = loaded["total_routeb"]
        total_04 = loaded["total_k9_0p4"]
        total_rec = loaded["total_out_rec"]
        steps = detect_local_steps(t, total_pre, 300.0)
        if not steps:
            continue
        first = steps[0]["t"]
        xlim = (max(0.0, first - 8.0), min(float(t[-1]), first + 55.0))
        keep = (t >= xlim[0]) & (t <= xlim[1])
        entries = [("pre", frame[keep], t[keep], total_pre[keep]),
                   ("out_rec", frame[keep], t[keep], total_rec[keep]),
                   ("k9_0p4", frame[keep], t[keep], total_04[keep]),
                   ("routeb", frame[keep], t[keep], total_b[keep])]
        out = FIG / ("fig%02d_%s_台阶局部.png" % (14 + index, label))
        overlay_figure(entries, out,
                       suptitle="%s ｜ 10 g 反复增减局部（台阶事件 %s）"
                                % (report["id"], ", ".join("%.1fs%+.0f" % (s["t"], s["adc"])
                                                           for s in steps[:8])),
                       y_label="总力（57 通道求和, ADC）")
        made.append(out.name)

    print("\n".join("  %s" % name for name in made))
    print("-> %s" % FIG)
    return 0


def detect_local_steps(t: np.ndarray, total: np.ndarray, min_adc: float,
                       refractory_s: float = 1.5) -> list[dict]:
    d = np.diff(total)
    idx = np.flatnonzero(np.abs(d) >= min_adc)
    steps: list[dict] = []
    for i in idx:
        if steps and t[i] - steps[-1]["t"] < refractory_s:
            step = steps[-1]
            step["adc"] = float(total[min(i + 1, total.size - 1)] - total[int(step["i"])])
            step["i"] = int(i)
            step["t"] = float(t[i])
            continue
        steps.append(dict(i=int(i), t=float(t[i]),
                          adc=float(total[min(i + 1, total.size - 1)] - total[i])))
    return steps


if __name__ == "__main__":
    raise SystemExit(figures())
