# -*- coding: utf-8 -*-
"""PPT 配图自检：画布越界 + 文本碰撞 + 缺字 + 文件完整性。

复用 make_ppt_figures 的绘图函数，函数返回 figure 而不落盘，
本脚本负责几何审计后再保存 PNG。
"""
from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_ppt_figures as M  # noqa: E402

FIG = M.FIG


def audit(fig) -> list[str]:
    issues: list[str] = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    boxes = []
    for ax in fig.axes:
        artists = [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            artists += list(legend.get_texts())
        for artist in artists:
            label = (artist.get_text() or "").strip()
            if not label:
                continue
            try:
                bb = artist.get_window_extent(renderer=renderer)
            except Exception:  # noqa: BLE001
                continue
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > width + 1 or bb.y1 > height + 1:
                issues.append(
                    f"越界 「{label[:24]}」 ({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}) / 画布 {width}x{height}")
            boxes.append((label, (bb.x0, bb.y0, bb.x1, bb.y1)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (t1, b1), (t2, b2) = boxes[i], boxes[j]
            ox = min(b1[2], b2[2]) - max(b1[0], b2[0])
            oy = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if ox > 3 and oy > 3:
                small = min((b1[2] - b1[0]) * (b1[3] - b1[1]),
                            (b2[2] - b2[0]) * (b2[3] - b2[1]))
                if small > 0 and ox * oy / small > 0.18:
                    issues.append(f"重叠 {ox * oy / small:.0%} 「{t1[:18]}」×「{t2[:18]}」")
    return issues


def main() -> int:
    warnings.simplefilter("error", UserWarning)

    session = json.loads((M.DATA / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    pre = M.load_stream(M.DATA, "device_001_pre_seg0.csv")
    seg = M.load_stream(M.DATA, "device_001_seg000.csv")
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame - frame[0]) / fps
    pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
    seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)
    before = (time >= 193.5) & (time < 194.45)
    after = (time >= 195.0) & (time < 196.0)
    shift = float(np.median(seg_total[after]) - np.median(seg_total[before]))
    bb0, ba0 = float(np.median(seg_total[before])), float(np.median(seg_total[after]))

    chan = M.load_json("channel_result.json")
    acc = M.load_json("guard_accept.json")
    cross = M.load_json("dip_crosscheck.json")

    jobs = [
        ("ppt_fig01_现象总览.png", lambda: M.fig01(time, pre_total, seg_total, shift)),
        ("ppt_fig02_局部放大.png", lambda: M.fig02(time, pre_total, seg_total, shift, bb0, ba0)),
        ("ppt_fig03_通道影响.png", lambda: M.fig03(chan)),
        ("ppt_fig04_受控复现.png", lambda: M.fig04(acc)),
        ("ppt_fig05_修复对比.png", lambda: M.fig05(acc)),
        ("ppt_fig06_出现次数.png", lambda: M.fig06(cross)),
        ("ppt_fig07_两种变化对比.png", lambda: M.fig07()),
        ("ppt_fig08_设计原则.png", lambda: M.fig08()),
    ]

    FIG.mkdir(parents=True, exist_ok=True)
    geometry, files = {}, {}
    for name, fn in jobs:
        fig = fn()
        issues = audit(fig)
        geometry[name] = issues
        fig.savefig(FIG / name, dpi=175, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with Image.open(FIG / name) as im:
            files[name] = {"size": im.size, "bytes": (FIG / name).stat().st_size}
        files[name]["edge_ink"] = edge_ink(FIG / name)

    print(json.dumps({"geometry": geometry, "files": files}, ensure_ascii=False, indent=1))
    bad = {k: v for k, v in geometry.items() if v}
    print("GEOMETRY_ISSUES:", sum(len(v) for v in bad.values()))
    return 1 if bad else 0


def edge_ink(path: Path, band: int = 2) -> dict:
    """检查落盘 PNG 的四条边缘：若边缘 2px 内有非白像素，说明内容被裁切。"""
    with Image.open(path).convert("L") as im:
        arr = np.asarray(im, dtype=np.uint8)
    h, w = arr.shape
    nonwhite = arr < 245
    return {
        "left": int(nonwhite[:, :band].sum()),
        "right": int(nonwhite[:, w - band:].sum()),
        "top": int(nonwhite[:band, :].sum()),
        "bottom": int(nonwhite[h - band:, :].sum()),
        "size": [w, h],
    }


if __name__ == "__main__":
    raise SystemExit(main())
