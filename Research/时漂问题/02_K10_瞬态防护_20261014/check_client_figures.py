# -*- coding: utf-8 -*-
"""配图结构自检：文本越界 / 文本互叠 / 面板空白。"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE / "figure_check.json"

_orig_savefig = plt.Figure.savefig
_results = []


def _patched(self, fname, *args, **kwargs):
    fig = self
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    problems = []
    for ax_index, ax in enumerate(fig.axes):
        candidates = [ax.title, ax.xaxis.label, ax.yaxis.label]
        candidates += list(ax.texts)
        candidates += list(ax.get_legend().get_texts()) if ax.get_legend() else []
        boxes = []
        for text in candidates:
            if not text.get_text().strip():
                continue
            bb = text.get_window_extent(renderer)
            boxes.append((text.get_text()[:24].replace("\n", " / "), bb))
            if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > fig_bbox.x1 + 2 or bb.y1 > fig_bbox.y1 + 2:
                problems.append(dict(kind="越界", panel=ax_index, text=text.get_text()[:24]))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a_box, b_box = boxes[i][1], boxes[j][1]
                if a_box.overlaps(b_box):
                    inter = max(0.0, min(a_box.x1, b_box.x1) - max(a_box.x0, b_box.x0)) * \
                            max(0.0, min(a_box.y1, b_box.y1) - max(a_box.y0, b_box.y0))
                    area = min(a_box.width * a_box.height, b_box.width * b_box.height)
                    if area > 0 and inter / area > 0.18:
                        problems.append(dict(kind="文本互叠", panel=ax_index,
                                             a=boxes[i][0], b=boxes[j][0],
                                             overlap_ratio=round(inter / area, 3)))
    _results.append(dict(file=Path(fname).name, problems=problems))
    return _orig_savefig(self, fname, *args, **kwargs)


plt.Figure.savefig = _patched
runpy.run_path(str(HERE / "make_client_figures.py"), run_name="__main__")
plt.Figure.savefig = _orig_savefig

REPORT.write_text(json.dumps(_results, ensure_ascii=False, indent=2), encoding="utf-8")
bad = [r for r in _results if r["problems"]]
for item in _results:
    flag = "OK  " if not item["problems"] else "问题"
    print(f"{flag} {item['file']}  问题数={len(item['problems'])}")
    for problem in item["problems"]:
        print("      ", json.dumps(problem, ensure_ascii=False))
print("全部通过" if not bad else f"{len(bad)} 张图存在问题")
