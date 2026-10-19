# -*- coding: utf-8 -*-
"""路线B 的保住率对「注入时刻」有多敏感？

同一份真实背景（设备1测试2）、同一个瞬时 100 ADC / 57 通道注入、保持 15 s，
只把注入起点在 26.9~40 s 之间平移几秒，看持载末 5 s 的配对响应如何变化。
最终探针 final_probe.py 用的起点是 33.674 s（= 首次大加载 + 30 s）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_k9_vs_routeb import load_session, params_of, run_k9  # noqa: E402
from route_b import route_b                                       # noqa: E402
from dptool.figure_export import export_small_multiples           # noqa: E402
from dptool.snapshot import CurveSnapshot, FigureSnapshot, PanelSnapshot  # noqa: E402

STARTS = (26.5, 30.0, 33.674, 37.44, 40.0)
HOLD_S = 15.0
COLORS = ("#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd")


def figure(t, curves, out_path: Path, k9_curve) -> None:
    x = t - 33.674
    mask = (x >= -2.0) & (x <= 28.0)
    items = [CurveSnapshot(name=label, x=x[mask], y=y[mask], color=color, width=1.3)
             for (label, y), color in zip(curves, COLORS)]
    items.append(CurveSnapshot(name=k9_curve[0], x=x[mask], y=k9_curve[1][mask],
                               color="0.35", style="dash", width=1.2))
    panel = PanelSnapshot(
        title="真实背景上的同一次注入，只把注入起点平移几秒（保持段 -5~0 s 为读数窗）",
        curves=items, x_label="相对 33.674 s 的时间 (s) ｜ 0 = 最终探针所用注入起点",
        y_label="配对响应（注入臂 − 无注入臂, ADC）", show_legend=True)
    snap = FigureSnapshot(suptitle="路线B 的持载保住率对注入时刻极度敏感；K9@0.4 不敏感",
                          panels=[panel], ncol=1, width_in=17.0, panel_height_in=6.2, dpi=110)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = export_small_multiples(snap, str(out_path), dpi=110, width_in=17.0,
                                    panel_height_in=6.2, check=True)
    print("  -> %s（%d 字节）" % (out_path.name, result["bytes"]))


def main() -> int:
    sess = load_session("设备1", "测试2-2026-10-13_16-43-59")
    t = sess["t"]
    background = sess["pre"].astype(np.float64)
    raw_params = dict(sess["session"]["algorithm"]["params"])
    profile = np.zeros((t.size, 57))
    rows = []
    curves: list[tuple[str, np.ndarray]] = []
    k9_curve: tuple[str, np.ndarray] | None = None
    for method in ("k9_0p4", "routeb"):
        def run(values):
            if method == "k9_0p4":
                return run_k9(values, t, params_of(sess["session"], 0.4)).sum(1)
            out, _ = route_b(values, t, dict(raw_params, r_slow_max=0.4))
            return out.sum(1)
        bg_out = run(background)
        for start in STARTS:
            inject = background.copy()
            mask = (t >= start) & (t < start + HOLD_S)
            inject[mask] += 100.0 / 57.0
            pair = run(inject) - bg_out
            hold = float(np.median(pair[(t >= start + 10) & (t < start + HOLD_S)]))
            rel = float(np.median(pair[(t >= start + HOLD_S + 7) & (t < start + HOLD_S + 12)]))
            rows.append(dict(method=method, start_s=start, hold_adc=hold,
                             release_7_12_adc=rel))
            if method == "routeb":
                curves.append(("路线B 注入起点 %.1f s → 保持 %+.1f ADC" % (start, hold), pair))
            elif abs(start - 33.674) < 1e-6:
                k9_curve = ("K9@0.4（注入起点 33.674 s）→ 保持 %+.1f ADC" % hold, pair)
            print("  %-8s 注入起点 %7.3f s → 持载末 5 s 响应 %7.1f ADC ｜ 撤载后 7~12 s %+7.1f"
                  % (method, start, hold, rel), flush=True)

    print("\n%-10s" % "注入起点s" + "".join("%18s" % m for m in ("K9@0.4", "路线B")))
    for start in STARTS:
        cells = [next(r for r in rows if r["method"] == m and r["start_s"] == start)["hold_adc"]
                 for m in ("k9_0p4", "routeb")]
        print("%-10.3f" % start + "".join("%17.1f " % c for c in cells))

    (HERE / "injection_timing.json").write_text(
        json.dumps({"starts": list(STARTS), "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n-> %s" % (HERE / "injection_timing.json"))
    if k9_curve is not None:
        figure(t, curves, HERE / "figures" / "fig20_注入时刻敏感性.png", k9_curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
