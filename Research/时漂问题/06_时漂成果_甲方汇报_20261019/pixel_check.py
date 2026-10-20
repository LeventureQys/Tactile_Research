"""渲染后的像素级自检（替代目视：当前模型无法读图）。

对每页 PNG 统计：
  ink      非白像素占比（页面是否有内容）
  blue     品牌蓝像素数（图表/标题条是否渲染出来）
  title    标题带（y 0.35–1.05 in）深色像素数
  edge     页面四边 4 px 内的非白像素数（内容被裁的迹象）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DPI = 110.0


def scan(path: Path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    h, w, _ = a.shape
    nonwhite = (a.sum(2) < 720)
    blue = (a[..., 2] > 150) & (a[..., 2] - a[..., 0] > 60)
    top = int(0.35 * DPI)
    bot = int(1.05 * DPI)
    title = nonwhite[top:bot, int(0.7 * DPI):int(9.8 * DPI)].sum()
    edge = (nonwhite[:4, :].sum() + nonwhite[-4:, :].sum()
            + nonwhite[:, :4].sum() + nonwhite[:, -4:].sum())
    return dict(page=path.name, size=[w, h],
                ink_pct=round(float(nonwhite.mean()) * 100, 2),
                blue_px=int(blue.sum()),
                title_ink=int(title),
                edge_ink=int(edge))


def main():
    report = {}
    for tag in ("preview_a", "preview_b"):
        d = HERE / tag
        if not d.exists():
            continue
        rows = [scan(p) for p in sorted(d.glob("page-*.png"))]
        report[tag] = rows
        print(f"== {tag} ({len(rows)} pages)")
        for r in rows:
            print(f"  {r['page']}  ink={r['ink_pct']:5.2f}%  blue={r['blue_px']:6d}  "
                  f"title={r['title_ink']:6d}  edge={r['edge_ink']:4d}")
    (HERE / "out" / "pixel_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
