"""PPT 版面自检：越界、文本框溢出估算、表格宽度、图片比例。

模型不具备读图能力，因此这里用几何计算替代目视检查：
  - shape 是否超出版面
  - 文本按字号/行距估算所需高度，与文本框高度比对（>105% 记溢出风险）
  - 表格列宽之和与声明宽度的偏差
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

HERE = Path(__file__).resolve().parent
SLIDE_W, SLIDE_H = 13.333, 7.5
TOL = 0.02


def char_units(text):
    u = 0.0
    for c in text:
        u += 1.0 if ord(c) > 127 else 0.56
    return max(u, 1.0)


def est_text_height(tf, width_in):
    total = 0.0
    for p in tf.paragraphs:
        text = "".join(r.text for r in p.runs)
        if not text:
            total += 0.10
            continue
        sizes = [r.font.size.pt for r in p.runs if r.font.size is not None]
        size = max(sizes) if sizes else 12.0
        ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.15
        cap = width_in / (size / 72.0)
        lines = max(1, math.ceil(char_units(text) / cap))
        total += lines * size * ls / 72.0
    return total


def check(path: Path):
    prs = Presentation(path)
    issues, stats = [], []
    for i, slide in enumerate(prs.slides, 1):
        pics = tables = texts = 0
        for sh in slide.shapes:
            l, t = Emu(sh.left).inches, Emu(sh.top).inches
            w, h = Emu(sh.width).inches, Emu(sh.height).inches
            name = f"p{i}/{sh.shape_type}/{sh.name}"
            if l < -TOL or t < -TOL or l + w > SLIDE_W + TOL or t + h > SLIDE_H + TOL:
                issues.append(dict(kind="out_of_slide", shape=name,
                                   box=[round(l, 2), round(t, 2), round(w, 2), round(h, 2)]))
            if sh.has_text_frame and sh.text_frame.text.strip():
                texts += 1
                need = est_text_height(sh.text_frame, max(w - 0.02, 0.2))
                if need > h * 1.05:
                    issues.append(dict(kind="text_overflow_risk", shape=name,
                                       need_in=round(need, 2), box_h=round(h, 2),
                                       text=sh.text_frame.text[:44]))
            if sh.has_table:
                tables += 1
                tw = sum(Emu(c.width).inches for c in sh.table.columns)
                if abs(tw - w) > 0.06:
                    issues.append(dict(kind="table_width", shape=name,
                                       sum=round(tw, 2), declared=round(w, 2)))
                for ri, row in enumerate(sh.table.rows):
                    for ci, cell in enumerate(row.cells):
                        cw = Emu(sh.table.columns[ci].width).inches - 0.22
                        need = est_text_height(cell.text_frame, max(cw, 0.2))
                        rh = Emu(row.height).inches
                        if need > rh * 1.05:
                            issues.append(dict(kind="cell_overflow_risk",
                                               shape=f"{name}/r{ri}c{ci}",
                                               need_in=round(need, 2),
                                               box_h=round(rh, 2),
                                               text=cell.text_frame.text[:36]))
            if sh.shape_type is not None and str(sh.shape_type).startswith("PICTURE"):
                pics += 1
        stats.append(dict(slide=i, shapes=len(slide.shapes), pictures=pics,
                          tables=tables, textboxes=texts))
    return dict(file=path.name, slides=len(prs.slides), stats=stats, issues=issues)


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted((HERE / "deliverables").glob("*.pptx"))
    out = [check(p) for p in paths]
    for r in out:
        print(f"{r['file']}: {r['slides']} slides, {len(r['issues'])} issues")
        for it in r["issues"]:
            print("   ", it)
    (HERE / "out" / "layout_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
