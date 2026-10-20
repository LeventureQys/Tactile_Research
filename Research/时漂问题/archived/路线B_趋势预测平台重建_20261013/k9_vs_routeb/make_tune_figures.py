# -*- coding: utf-8 -*-
"""参数空间实测出图（dptool）。

fig21 权衡曲线：漂移最差 vs 分散/集中注入保持量
fig22 cap 响应：cap 扫描下漂移与分散保持的反向走势
fig23 逐通道吸收时间：为什么分散加载必然被吞、集中加载必然保住
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
sys.path.insert(0, str(HERE))

from dptool.figure_export import export_small_multiples            # noqa: E402
from dptool.snapshot import CurveSnapshot, FigureSnapshot, PanelSnapshot  # noqa: E402
from compare_k9_vs_routeb import detect_steps, load_session, params_of    # noqa: E402
from creep_observer_k9 import CreepObserverK9                      # noqa: E402

FIG = HERE / "figures"
DISP, CONC = "分散57×100", "集中1×300"


def save(snap: FigureSnapshot, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    result = export_small_multiples(snap, str(FIG / name), dpi=115,
                                    width_in=snap.width_in,
                                    panel_height_in=snap.panel_height_in, check=True)
    print("  -> %-34s %d 字节" % (name, result["bytes"]))


def fig_tradeoff(rows: list[dict]) -> None:
    order = ["r=1.0", "r=0.6", "r=0.6 cap=0.01", "r=0.6 cap=0.005 confirm=8",
             "r=0.6 cap=0.005 unfreeze=8", "r=0.6 cap=0.002"]
    by_name = {r["name"]: r for r in rows}
    picked = [by_name[n] for n in order if n in by_name]
    picked.sort(key=lambda r: r["worst_abs_net"])

    x = np.array([r["worst_abs_net"] for r in picked])
    disp = np.array([r["disp"] for r in picked])
    conc = np.array([r["conc"] for r in picked])
    ref = by_name.get("现役 r=0.1")

    curves = [
        CurveSnapshot(name="分散 57×100 ADC 保持量（需求 70 → 70 ADC）", x=x, y=disp,
                      color="#d62728", width=1.8),
        CurveSnapshot(name="集中 1×300 ADC 保持量（需求 210 ADC）", x=x, y=conc,
                      color="#1f77b4", width=1.4),
        CurveSnapshot(name="需求线：分散 70 ADC", x=x, y=np.full_like(x, 70.0),
                      color="#2ca02c", style="dash", width=1.2),
    ]
    labels = []
    for row in picked:
        labels.append("%s: %.0f/%.1f%%" % (row["name"], row["worst_abs_net"],
                                          100.0 * row["disp"] / 100.0))
    panel = PanelSnapshot(
        title="K9 参数空间的权衡曲线 ｜ 每点：%s" % "；".join(labels[:3]),
        note="；".join(labels[3:]),
        curves=curves, x_label="漂移侧：四份恒载 |首尾净变化| 的最大值 (ADC)",
        y_label="注入保持量 (ADC)", show_legend=True)
    if ref is not None:
        panel.curves.append(CurveSnapshot(
            name="现役 r=0.1：漂移 %.0f，分散保持 %.1f" % (ref["worst_abs_net"], ref["disp"]),
            x=np.array([ref["worst_abs_net"]]), y=np.array([ref["disp"]]),
            color="#7f7f7f", style="dot", width=2.0))
    save(FigureSnapshot(suptitle="右拇指指腹 ｜ 漂移抑制 vs 分散小变化保护：现有结构下的权衡曲线",
                        panels=[panel], ncol=1, width_in=15.0, panel_height_in=6.0),
         "fig21_参数权衡曲线.png")


def fig_cap(rows: list[dict]) -> None:
    caps, drifts, disps = [], [], []
    for row in rows:
        override = row["override"]
        if "slope_cap_frac" not in override or override.get("r_slow_max") != 0.6:
            continue
        if set(override) - {"r_slow_max", "slope_cap_frac"}:
            continue
        caps.append(override["slope_cap_frac"])
        drifts.append(row["worst_abs_net"])
        disps.append(row["disp"])
    order = np.argsort(caps)
    caps = np.array(caps)[order]
    drifts = np.array(drifts)[order]
    disps = np.array(disps)[order]
    panel = PanelSnapshot(
        title="slope_cap_frac 扫描（r_slow_max 固定 0.6）",
        curves=[
            CurveSnapshot(name="漂移最差 |首尾净变化| (ADC)", x=caps, y=drifts,
                          color="#d62728", width=1.8),
            CurveSnapshot(name="分散保持量 (ADC) ×3 便于同图", x=caps, y=disps * 3.0,
                          color="#1f77b4", width=1.6),
        ], x_label="slope_cap_frac（慢态积分速率上限）", y_label="ADC（蓝线已 ×3）",
        show_legend=True)
    save(FigureSnapshot(suptitle="速率旋钮一压小：漂移急剧恶化，分散保持只缓慢上升",
                        panels=[panel], ncol=1, width_in=14.0, panel_height_in=5.5),
         "fig22_cap响应.png")


def fig_absorb() -> None:
    sess = load_session("设备1", "测试2-2026-10-13_16-43-59")
    t, values = sess["t"], sess["pre"].astype(np.float64)
    params = params_of(sess["session"], 0.6)
    stop = int(np.searchsorted(t, detect_steps(t, values.sum(1), 2000.0)[0]["t"] + 30.0))
    obs = CreepObserverK9(params)
    obs._trace_frame = lambda *a: None
    for i in range(stop):
        obs.process(float(t[i]), values[i])
    e = np.maximum(values[stop] - obs.zero - obs.x_fast - obs.x_slow, 0.0)
    cap = params.slope_cap_frac * np.maximum(e, 1.0)
    idx = np.argsort(cap)
    absorb_disp = (100.0 / 57.0) / cap[idx]
    absorb_conc = 300.0 / cap[idx]
    hold = np.full(cap.size, 15.0)

    panel = PanelSnapshot(
        title="按通道慢态吸收时间（57 通道，按吸收时间排序；注入保持窗 15 s）",
        curves=[
            CurveSnapshot(name="分散 57×100 ADC：每通道 1.75 ADC 被吸完所需时间 (s)（>20 s 截到 20）",
                          x=np.arange(cap.size), y=np.minimum(absorb_disp, 20.0),
                          color="#d62728", width=1.6),
            CurveSnapshot(name="注入保持窗 15 s", x=np.arange(cap.size), y=hold,
                          color="#2ca02c", style="dash", width=1.2),
        ], x_label="通道序号（按吸收时间升序）", y_label="吸收时间 (s)", ylim=(0.0, 20.0),
        show_legend=True)
    save(FigureSnapshot(
        suptitle="分散加载为何必然被吞：74%% 的通道在 1 秒内就吸收掉自己那份；集中加载需 %.0f s"
                 % float(np.median(absorb_conc)),
        panels=[panel], ncol=1, width_in=14.0, panel_height_in=5.5),
         "fig23_逐通道吸收时间.png")


def main() -> int:
    rows = json.loads((HERE / "tune_grid_merged.json").read_text(encoding="utf-8"))
    fig_tradeoff(rows)
    fig_cap(rows)
    fig_absorb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
