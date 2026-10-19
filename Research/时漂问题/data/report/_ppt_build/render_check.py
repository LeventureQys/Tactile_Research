# -*- coding: utf-8 -*-
"""渲染页像素级兜底检查：内容是否超出安全边距 / 图片是否越界。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SLIDE_W, SLIDE_H = 13.333, 7.5
PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("preview")


def analyze(path: Path) -> dict:
    with Image.open(path).convert("RGB") as im:
        arr = np.asarray(im, dtype=np.int16)
    h, w, _ = arr.shape
    px = w / SLIDE_W
    ink = (np.abs(arr - 255).sum(axis=2) > 36)

    def to_px(inches):
        return int(round(inches * px))

    top_bar = to_px(0.30)              # 顶部通栏色条属于设计
    body = ink[top_bar:h, :]
    rows = np.where(body.any(axis=1))[0]
    cols = np.where(body.any(axis=0))[0]
    bbox = None
    if rows.size:
        bbox = {
            "top_in": round((rows[0] + top_bar) / px, 3),
            "bottom_in": round((rows[-1] + top_bar) / px, 3),
            "left_in": round(cols[0] / px, 3),
            "right_in": round(cols[-1] / px, 3),
        }
    edge = {
        "left": int(ink[:, :4].sum()),
        "right": int(ink[:, w - 4:].sum()),
        "bottom": int(ink[h - 4:, :].sum()),
    }
    return {"size": [w, h], "bbox_in": bbox, "edge_ink": edge}


report = {p.name: analyze(p) for p in sorted(PREVIEW.glob("*.png"))}
issues = []
for name, r in report.items():
    e = r["edge_ink"]
    if e["left"] or e["right"] or e["bottom"]:
        issues.append(f"{name}: 左右/下边缘有墨迹 {e}")
    bb = r["bbox_in"]
    if bb:
        if bb["left_in"] < 0.15 or bb["right_in"] > SLIDE_W - 0.15:
            issues.append(f"{name}: 内容贴近左右边缘 bbox={bb}")
        if bb["bottom_in"] > SLIDE_H - 0.05:
            issues.append(f"{name}: 内容超出下边界 bbox={bb}")

print(json.dumps(report, ensure_ascii=False, indent=1))
print("\nEDGE_ISSUES:", len(issues))
for line in issues:
    print(" -", line)
