# -*- coding: utf-8 -*-
"""渲染图裁切 / 溢出兜底检查。

对每一页 PNG：
  · 页面四边 6px 内是否有非白、非底色像素（内容被裁切）；
  · 页脚分隔线（约 y = 7.03/7.5 * height）下方是否只有页脚文字；
  · 内容区（y 上界 1.55in 至下界 7.0in）之外是否存在大量墨迹。
仅用于兜底，最终判断以视觉模型读图为准。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SLIDE_W, SLIDE_H = 13.333, 7.5


def analyze(path: Path) -> dict:
    with Image.open(path).convert("RGB") as im:
        arr = np.asarray(im, dtype=np.int16)
    h, w, _ = arr.shape
    ink = (np.abs(arr - 255).sum(axis=2) > 36)

    def band_rows(y0, y1):
        a = int(y0 / SLIDE_H * h)
        b = int(y1 / SLIDE_H * h)
        return int(ink[a:b].sum()), a, b

    result = {
        "size": [w, h],
        "edge_ink": {
            "top": int(ink[:4].sum()),
            "bottom": int(ink[h - 4:].sum()),
            "left": int(ink[:, :4].sum()),
            "right": int(ink[:, w - 4:].sum()),
        },
        "row_ink": {},
    }
    for name, (y0, y1) in {
        "header": (0.0, 1.55),
        "body": (1.55, 6.05),
        "footer_zone": (6.05, 7.05),
        "below_footer": (7.35, 7.5),
    }.items():
        count, a, b = band_rows(y0, y1)
        result["row_ink"][name] = {"ink": count, "rows": [a, b],
                                   "density": round(count / max(1, (b - a) * w), 5)}
    return result


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "preview-v1"
    report = {p.name: analyze(p) for p in sorted(directory.glob("*.png"))}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    bad = []
    for name, item in report.items():
        edge = sum(item["edge_ink"].values())
        if edge > 0:
            bad.append(f"{name}: 边缘存在墨迹 {item['edge_ink']}")
        if item["row_ink"]["below_footer"]["ink"] > 0:
            bad.append(f"{name}: 页脚分隔线下方仍有内容")
    print("\nLAYOUT_ISSUES:", len(bad))
    for line in bad:
        print(" -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
