# -*- coding: utf-8 -*-
"""生成《瞬态干扰导致基线偏移·问题说明》甲方汇报 PPT（模量科技 Modutech）。

数据与结论全部来自：
  · 瞬态干扰基线偏移问题说明_甲方版.md
  · 02_K10_瞬态防护_20261014 下的现场录制与验证结果（*_result.json）
配图由 make_ppt_figures.py / check_ppt_figures.py 生成并通过几何自检。
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures_ppt"
ASSETS = HERE / "assets"
OUT = HERE.parent / "瞬态干扰基线偏移问题说明_甲方汇报.pptx"

# ── 视觉规范 ────────────────────────────────────────────────────────────────
INK = RGBColor(0x0B, 0x2E, 0x4F)        # 深蓝（主色）
INK_SOFT = RGBColor(0x16, 0x45, 0x72)
PAPER = RGBColor(0xF4, 0xF7, 0xFA)      # 浅底
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0xC8, 0x9A, 0x3C)       # 强调
TEXT = RGBColor(0x17, 0x21, 0x2B)
GREY = RGBColor(0x5A, 0x65, 0x72)
LINE = RGBColor(0xD8, 0xDF, 0xE8)
REPORT = RGBColor(0xA8, 0x32, 0x2B)
FIX = RGBColor(0x2E, 0x6B, 0x50)

CN = "微软雅黑"
EN = "Arial"
W, H = 13.333, 7.5

META = {
    "company_cn": "模量科技",
    "company_en": "Modutech",
    "project": "触觉阵列时漂补偿",
    "version": "K10 瞬态防护",
    "date": "2026-10-14",
    "client": "XX 项目组",
}


# ── 基础工具 ────────────────────────────────────────────────────────────────
def set_run(run, size, bold=False, color=TEXT, font=None, italic=False) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    name = font or (EN if all(ord(ch) < 128 for ch in run.text) else CN)
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = rPr.find(qn(tag))
        if node is None:
            node = rPr.makeelement(qn(tag), {})
            rPr.append(node)
        node.set("typeface", CN if tag == "a:ea" else name)


def rect(slide, x, y, w, h, fill=None, line=None, radius=None, shadow=False):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shape.adjustments[0] = radius
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    if not shadow:
        shape.shadow.inherit = False
    shape.text_frame.word_wrap = True
    return shape


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return frame


def para(frame, first=False, space_before=0, space_after=0, line=1.28, align=PP_ALIGN.LEFT):
    p = frame.paragraphs[0] if first else frame.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.line_spacing = line
    return p


def write(frame, blocks, first=True) -> None:
    """blocks: [(text, size, bold, color, extra_dict)]"""
    for index, item in enumerate(blocks):
        text, size, bold, color = item[0], item[1], item[2], item[3]
        extra = item[4] if len(item) > 4 else {}
        p = para(frame, first=(first and index == 0),
                 space_before=extra.get("before", 0),
                 space_after=extra.get("after", 0),
                 line=extra.get("line", 1.28),
                 align=extra.get("align", PP_ALIGN.LEFT))
        run = p.add_run()
        run.text = text
        set_run(run, size, bold, color, extra.get("font"))


def slide_base(prs, title=None, kicker=None, number=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, 0.115, fill=INK)
    rect(slide, 0, 0.115, 4.25, 0.045, fill=GOLD)
    if title:
        frame = textbox(slide, 0.75, 0.42, 9.8, 0.62, anchor=MSO_ANCHOR.MIDDLE)
        write(frame, [(title, 26, True, INK)])
    if kicker:
        frame = textbox(slide, 0.75, 1.06, 10.6, 0.34)
        write(frame, [(kicker, 11.5, False, GREY)])
    # 右上品牌区
    slide.shapes.add_picture(str(ASSETS / "modutech_logo.png"),
                             Inches(11.63), Inches(0.4), height=Inches(0.46))
    frame = textbox(slide, 10.15, 0.44, 1.36, 0.42, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [(META["company_cn"], 11.5, True, INK, {"align": PP_ALIGN.RIGHT})])
    # 页脚
    rect(slide, 0.75, 7.03, 11.83, 0.012, fill=LINE)
    frame = textbox(slide, 0.75, 7.11, 9.0, 0.28)
    write(frame, [(f"{META['company_cn']} {META['company_en']}　|　{META['project']}　|　{META['version']}",
                   9, False, GREY)])
    if number is not None:
        frame = textbox(slide, 11.4, 7.11, 1.18, 0.28)
        write(frame, [(f"{number:02d}", 10, True, INK, {"align": PP_ALIGN.RIGHT, "font": EN})])
    return slide


def card(slide, x, y, w, h, accent=INK, fill=PAPER):
    rect(slide, x, y, w, h, fill=fill, radius=0.06)
    rect(slide, x, y, 0.055, h, fill=accent)
    return (x + 0.3, y)


def kv_card(slide, x, y, w, h, label, value, note=None, accent=INK, value_size=24):
    bx, by = card(slide, x, y, w, h, accent=accent)
    frame = textbox(slide, bx, by + 0.14, w - 0.55, h - 0.2)
    write(frame, [(label, 11, True, accent)])
    frame = textbox(slide, bx, by + 0.46, w - 0.55, h - 0.5)
    blocks = [(value, value_size, True, TEXT, {"line": 1.18})]
    if note:
        blocks.append((note, 10, False, GREY, {"before": 4, "line": 1.2}))
    write(frame, blocks)


def bullet_card(slide, x, y, w, h, head, lines, accent=INK, fill=PAPER,
                head_size=14, body_size=11.5):
    bx, by = card(slide, x, y, w, h, accent=accent, fill=fill)
    frame = textbox(slide, bx, by + 0.16, w - 0.6, h - 0.28)
    write(frame, [(head, head_size, True, accent, {"after": 7})])
    for line in lines:
        write(frame, [(line, body_size, False, TEXT, {"after": 5, "line": 1.3})], first=False)


def figure(slide, name, x, y, w, h=None):
    path = FIG / name
    if h is None:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    else:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                 width=Inches(w), height=Inches(h))
    return path


def caption(slide, x, y, w, text, size=10):
    frame = textbox(slide, x, y, w, 0.5)
    write(frame, [(text, size, False, GREY, {"line": 1.25})])


def table(slide, x, y, w, rows, col_ratio, head_size=11, body_size=11, row_h=0.36,
          right_cols=1, left_cols=1):
    """right_cols / left_cols：末尾右对齐的列数、开头左对齐的列数，中间列居中。"""
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                   Inches(w), Inches(row_h * n_rows))
    tbl = shape.table
    tbl.first_row = True
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.12)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = INK
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else PAPER
            frame = cell.text_frame
            frame.word_wrap = True
            p = frame.paragraphs[0]
            if c < left_cols:
                p.alignment = PP_ALIGN.LEFT
            elif c >= n_cols - right_cols:
                p.alignment = PP_ALIGN.RIGHT
            else:
                p.alignment = PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(value)
            set_run(run, head_size if r == 0 else body_size, r == 0,
                    WHITE if r == 0 else TEXT)
    for c, ratio in enumerate(col_ratio):
        tbl.columns[c].width = Emu(int(Inches(w) * ratio))
    return tbl


# ── 各页 ────────────────────────────────────────────────────────────────────
def s01_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, H, fill=INK)
    rect(slide, 0, 0, 0.28, H, fill=GOLD)

    frame = textbox(slide, 1.15, 1.42, 6.4, 0.4)
    write(frame, [(f"{META['company_cn']} · {META['company_en']}", 13, True, GOLD)])

    frame = textbox(slide, 1.15, 2.0, 8.3, 2.0)
    write(frame, [("瞬态干扰导致的\n基线永久偏移", 44, True, WHITE, {"line": 1.16}),
                  ("问题说明与解决方案", 22, False, RGBColor(0xBF, 0xD4, 0xE8),
                   {"before": 8, "line": 1.2})])

    rect(slide, 1.15, 4.72, 2.1, 0.035, fill=GOLD)
    frame = textbox(slide, 1.15, 4.98, 10.2, 1.0)
    write(frame, [("数据来源：左拇指指腹恒载录制（长时间恒定加压）", 12, False,
                   RGBColor(0xA9, 0xC1, 0xD8), {"after": 4}),
                  (f"版本：{META['version']}　　日期：{META['date']}　　"
                   f"汇报对象：{META['client']}", 12, False,
                   RGBColor(0xA9, 0xC1, 0xD8))])

    slide.shapes.add_picture(str(ASSETS / "modutech_logo.png"),
                             Inches(10.62), Inches(4.95), height=Inches(1.05))
    frame = textbox(slide, 10.02, 6.16, 2.2, 0.4)
    write(frame, [("MODUTECH", 15, True, WHITE, {"align": PP_ALIGN.RIGHT, "font": EN})])
    return slide


def s02_summary(prs):
    slide = slide_base(prs, "结论摘要", "一页结论：现象、原因、方案与验证结果", 2)
    items = [
        ("现象", "一次约 0.3 秒的瞬时跳动，使补偿后读数基线永久抬升 61.5 ADC，此后不再回落。", INK, 17),
        ("对照", "同一时段传感器原始读数没有任何台阶变化，跳动前后实测差值 0.0 ADC。", GREY, 17),
        ("影响量级", "61.5 ADC 按项目既有粗略比例折算约 0.06 N，"
                     "相当于 10 g 级目标信号（约 100 ADC）的六成。", GOLD, 15),
        ("根本原因", "算法把瞬时跳动误判为“载荷已卸除”，据此执行了一次不会自动撤销的基线修正。", REPORT, 15),
        ("解决方式", "增加一层瞬态识别保护：只有“持续保持”的电平变化才被认定为真实载荷变化。", FIX, 15),
        ("验证结果", "同样跳动下长期偏移由 +24 ADC 降至接近 0；对真实加载、真实卸载、"
                     "小幅度卸载的长期结果无影响。", FIX, 15),
    ]
    x0, y0, w, h, gap = 0.75, 1.62, 3.87, 1.94, 0.2
    for i, (label, text, accent, size) in enumerate(items):
        col, row = i % 3, i // 3
        x = x0 + col * (w + gap)
        y = y0 + row * (h + 0.26)
        bx, by = card(slide, x, y, w, h, accent=accent)
        frame = textbox(slide, bx, by + 0.15, w - 0.5, 0.3)
        write(frame, [(label, 12, True, accent)])
        frame = textbox(slide, bx, by + 0.52, w - 0.5, h - 0.68)
        write(frame, [(text, size, False, TEXT, {"line": 1.32})])
    caption(slide, 0.75, 6.14, 11.83,
            "口径说明：ADC 为传感器阵列原始计数单位；数据来自 2026-10-14 左拇指指腹长时间恒载录制与受控复现试验。")
    return slide


def s03_background(prs):
    slide = slide_base(prs, "背景：为什么需要基线补偿",
                       "矩阵式触觉传感器的输出会随时间缓慢漂移，直接读数不能作为绝对压力使用", 3)
    bullet_card(slide, 0.75, 1.62, 3.76, 2.55, "传感器本身的特性", [
        "原始输出随温度、蠕变与接触状态缓慢变化；",
        "加载时间越长，基线漂移越明显；",
        "漂移是缓慢的、可跟踪的，与突然的载荷变化在时间尺度上完全不同。",
    ], accent=INK)
    bullet_card(slide, 4.72, 1.62, 3.76, 2.55, "补偿算法做了什么", [
        "持续跟踪一个随时间缓慢变化的基线电平；",
        "输出的是“相对于当前基线”的载荷量；",
        "目标是让长时间按压下的读数保持稳定可用。",
    ], accent=INK_SOFT)
    bullet_card(slide, 8.69, 1.62, 3.89, 2.55, "补偿成立的前提", [
        "必须正确判别“仍有载荷”与“载荷已经卸除”；",
        "判别的依据是电平的变化；",
        "一旦判别被瞬态干扰欺骗，基线就会在错误的位置重新建立，并长期保持下去。",
    ], accent=REPORT)
    rect(slide, 0.75, 4.42, 11.83, 0.62, fill=PAPER, radius=0.16)
    frame = textbox(slide, 1.0, 4.5, 11.3, 0.46, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [("本页说明的是“为什么需要这套补偿”，"
                   "下一页开始说明瞬时干扰如何让这套补偿产生永久偏差。", 12.5, True, INK)])
    table(slide, 0.75, 5.28, 11.83, [
        ["时间尺度", "典型幅度", "物理含义", "补偿算法应当如何对待"],
        ["缓慢漂移（数十秒至数分钟）", "数百至数千 ADC", "温度、蠕变、接触状态变化", "更新基线，消除漂移"],
        ["真实载荷变化（持续保持）", "按载荷而定", "加载或卸载真实发生", "确认新基准，正常输出"],
        ["瞬态干扰（亚秒级）", "与真实变化可比重叠", "接触抖动、毛刺、瞬时干扰", "不改变基线（本次问题点）"],
    ], [0.30, 0.18, 0.26, 0.26], row_h=0.44)
    return slide


def s04_phenomenon(prs):
    slide = slide_base(prs, "问题现象", "恒载录制第 194.5 秒附近出现一次约 0.3 秒的瞬时跳动", 4)
    figure(slide, "ppt_fig01_现象总览.png", 0.97, 1.58, 11.4)
    caption(slide, 0.97, 6.72, 11.4,
            "灰线为传感器原始读数，蓝线为补偿后输出。原始读数全程平滑，跳动前后没有任何台阶（实测差值 0.0 ADC）；"
            "补偿输出在跳动处下探，随后基线整体上抬 61.5 ADC 并保持不变。")
    return slide


def s05_zoom(prs):
    slide = slide_base(prs, "问题现象 · 局部放大",
                       "跳动前后存在一个稳定台阶，而原始读数没有变化", 5)
    figure(slide, "ppt_fig02_局部放大.png", 1.2, 1.58, 10.9)
    caption(slide, 1.2, 6.56, 10.9,
            "图中绿线与红线为跳动前后的基线电平，两者之间存在稳定台阶；灰线为传感器原始读数，"
            "事件前后没有变化，说明台阶不是物理载荷变化造成的。")
    return slide


def s06_cause(prs):
    slide = slide_base(prs, "问题原因", "算法把一次瞬时跳动误判为“载荷已卸除”", 6)
    steps = [
        ("干扰到来", "读数在约 0.3 秒内快速跌破原有基准电平。", REPORT),
        ("被误判", "算法在极短时间内看到读数低于基准，判定载荷已被卸除。", REPORT),
        ("执行修正", "按照卸除逻辑执行一次基线修正，把当时错误电平当作新基准。", REPORT),
        ("长期保留", "跳动早已恢复，修正却不会自动回退，偏差长期留在输出中。", INK),
    ]
    x0, y0, w, gap = 0.75, 1.6, 2.85, 0.19
    for i, (head, body, accent) in enumerate(steps):
        x = x0 + i * (w + gap)
        bx, by = card(slide, x, y0, w, 1.86, accent=accent,
                      fill=RGBColor(0xFA, 0xF0, 0xEF) if accent == REPORT else PAPER)
        frame = textbox(slide, bx, by + 0.16, w - 0.5, 0.32)
        write(frame, [(f"{i + 1}. {head}", 14, True, accent)])
        frame = textbox(slide, bx, by + 0.58, w - 0.5, 1.16)
        write(frame, [(body, 11.5, False, TEXT, {"line": 1.3})])
        if i < 3:
            frame = textbox(slide, x + w + 0.02, y0 + 0.7, 0.16, 0.4, anchor=MSO_ANCHOR.MIDDLE)
            write(frame, [("▶", 11, True, RGBColor(0xA8, 0xB2, 0xBE), {"align": PP_ALIGN.CENTER})])
    bullet_card(slide, 0.75, 3.72, 5.88, 2.2, "关键：修正不可撤销", [
        "补偿算法在运行中持续维护一个“基线基准”；",
        "判定为卸除状态时，会把当时的电平接纳为新的基准；",
        "本次事件中，接纳动作发生在跳动的谷底，而跳动随后立即恢复；",
        "基准一旦被改写，不会因为读数回到原水平而自动回退。",
    ], accent=REPORT, fill=RGBColor(0xFA, 0xF0, 0xEF))
    bullet_card(slide, 6.7, 3.72, 5.88, 2.2, "与真实卸载的区别", [
        "真实卸载：载荷确实改变，新基准是正确的结果；",
        "瞬时跳动：载荷并未改变，新基准是错误的产物；",
        "两者在数值幅度上可能相似，仅凭幅度无法区分；",
        "因此需要额外的判别依据，见第 9 页。",
    ], accent=INK)
    return slide


def s07_channels(prs):
    slide = slide_base(prs, "问题原因 · 通道级证据",
                       "补偿量被永久扣减的通道，全部落在跳动最深的几条通道上", 7)
    figure(slide, "ppt_fig03_通道影响.png", 1.2, 1.5, 10.9)
    caption(slide, 1.2, 6.66, 10.9,
            "深蓝柱为该通道的瞬时跳动深度，红柱为该通道补偿量被永久扣减的量。"
            "跳动最深的 CH13 / CH25 / CH8 三条通道合计扣减 62.6 ADC，与全部通道补偿量变化总量 −61.6 ADC 吻合，"
            "排除了整体漂移与随机噪声的解释。")
    return slide


def s08_frequency(prs):
    slide = slide_base(prs, "问题原因 · 为什么此前未被发现",
                       "同类瞬时跳动的出现次数与录制时长、恒压状态强相关", 8)
    figure(slide, "ppt_fig06_出现次数.png", 0.75, 1.6, 7.65)
    table(slide, 8.62, 1.78, 3.96, [
        ["测试类型", "检出次数"],
        ["设备1 · 10 g 反复增减荷载", "1 次"],
        ["设备2 · 10 g 反复增减荷载", "1 次"],
        ["设备1 · 恒载测试1", "11 次"],
        ["设备2 · 恒载测试1", "13 次"],
        ["右拇指 · 恒载测试1", "22 次"],
        ["左拇指 · 恒载测试1", "2 次"],
    ], [0.62, 0.38], row_h=0.42)
    caption(slide, 8.62, 4.72, 3.96,
            "长时间恒压过程中，这类干扰出现得更频繁，每一次都可能留下一份不会消失的偏差，逐渐累积。")
    rect(slide, 0.75, 5.74, 11.83, 0.72, fill=PAPER, radius=0.12)
    frame = textbox(slide, 1.0, 5.82, 11.33, 0.56, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [("这也解释了为什么该问题在短时的 10 g 测试中不明显、"
                   "却在长时间恒压测试中凸显。", 12.5, True, INK)])
    return slide


def s09_solution(prs):
    slide = slide_base(prs, "解决方案 · 设计思路",
                       "区分二者的依据不是“幅度”，而是“变化是否持续保持”", 9)
    figure(slide, "ppt_fig07_两种变化对比.png", 0.75, 1.58, 11.83)
    rect(slide, 0.75, 6.18, 11.83, 0.68, fill=PAPER, radius=0.14)
    frame = textbox(slide, 1.0, 6.25, 11.33, 0.54, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [("两类变化在幅度上互相重叠，无法用阈值可靠区分；差别只在于："
                   "变化是否持续保持。", 12.5, True, INK)])
    return slide


def s10_principles(prs):
    slide = slide_base(prs, "解决方案 · 设计原则与功能定位",
                       "保护层只改变“疑似卸除”这一条判定路径，不改变原有补偿行为", 10)
    bullet_card(slide, 0.75, 1.6, 3.87, 2.3, "一、保护“形态”，不设幅度门槛", [
        "不以“偏移超过某个幅度就拒绝”为判据；",
        "避免小幅度真实操作被误伤；",
        "保护对象是变化的形态，不是幅度。",
    ], accent=INK)
    bullet_card(slide, 4.72, 1.6, 3.87, 2.3, "二、必须持续保持才被采纳", [
        "电平变化需在短时间内持续保持；",
        "才被确认为新的基线基准；",
        "抖动、毛刺、接触干扰这类“来了又走”的变化不会被采纳。",
    ], accent=INK_SOFT)
    bullet_card(slide, 8.7, 1.6, 3.88, 2.3, "三、大幅度真实卸载仍立即响应", [
        "明显卸载（例如手指抬起）不受保护层影响；",
        "卸载响应速度与原有算法一致；",
        "保持现有操作手感不变。",
    ], accent=FIX, fill=RGBColor(0xEE, 0xF6, 0xF1))
    bullet_card(slide, 0.75, 4.12, 5.88, 2.48, "功能定位", [
        "在原有补偿逻辑之上新增的一层识别与保护机制；",
        "作为可开关选项启用，关闭时与原有算法结果完全一致；",
        "不引入新的外部依赖，计算开销可忽略；",
        "在长时间恒压场景下建议优先启用。",
    ], accent=GOLD, fill=RGBColor(0xFA, 0xF5, 0xE9), head_size=15, body_size=12)
    rect(slide, 6.7, 4.12, 5.88, 2.48, fill=PAPER, radius=0.06)
    rect(slide, 6.7, 4.12, 0.055, 2.48, fill=INK)
    frame = textbox(slide, 7.0, 4.3, 5.3, 2.1)
    write(frame, [("开关带来的可控性", 15, True, INK, {"after": 8}),
                  ("· 现场可做 A/B 对照：同一段数据分别用开、关两种配置回放；", 12, False, TEXT,
                   {"after": 6, "line": 1.3}),
                  ("· 出现任何疑虑时，可立即回退到原有行为，不影响既有交付；", 12, False, TEXT,
                   {"after": 6, "line": 1.3}),
                  ("· 保护参数（确认时长等）与传感器噪声水平、实际操作速度相关，"
                   "需按现场标定后定版。", 12, False, TEXT, {"line": 1.3})])
    return slide


def s11_effects(prs):
    slide = slide_base(prs, "验证结果 · 受控复现与修复效果",
                       "把现场波形提取出来在受控条件下注入，对比三种扰动形态的长期结果", 11)
    figure(slide, "ppt_fig04_受控复现.png", 0.75, 1.58, 6.9)
    figure(slide, "ppt_fig05_修复对比.png", 7.88, 1.58, 4.7)
    caption(slide, 0.75, 4.62, 6.9,
            "受控复现：只有瞬时跳动留下永久偏移（约 +24 ADC），持续加载与无扰动对照都在噪声量级。")
    caption(slide, 7.88, 4.98, 4.7,
            "防护启用后，同样强度的跳动造成的长期偏移由约 +24 ADC 降至接近 0。")
    rect(slide, 0.75, 5.62, 11.83, 1.22, fill=PAPER, radius=0.1)
    frame = textbox(slide, 1.05, 5.78, 11.2, 0.94)
    write(frame, [("受控复现确认了触发条件：保护层只需针对“瞬时跳动”这一种形态即可。", 12.5, True, INK,
                   {"after": 7}),
                  ("真实卸载 90% 会引起约 −2.3 万 ADC 的载荷变化，量级与上述三者不可比，"
                   "其长期偏移经单独核算为 −0.05 ADC，与无扰动对照等同，说明保护层不会带来额外偏移。",
                   11.5, False, GREY, {"line": 1.3})])
    return slide


def s12_regression(prs):
    slide = slide_base(prs, "验证结果 · 对正常功能无影响",
                       "防护启用后，对真实加载、真实卸载与小信号检测的实测影响", 12)
    rows = [
        ["验证项", "结果"],
        ["持续加载（+2000 ADC，20 秒）", "长期结果与原来完全一致"],
        ["真实卸载（卸载 90%）", "长期结果与原来完全一致"],
        ["小幅度真实卸载（每通道 20 ~ 400 ADC）", "长期稳态差异 0；卸载瞬间约 17 ADC 短暂差异，5 秒内收敛"],
        ["长时间恒载录制的漂移指标", "变化 ≤ 0.7%"],
        ["小信号（10 g 级）保留能力", "无变化"],
        ["保护层关闭时与原有算法的一致性", "完全一致（逐帧比对最大差异 0.0 ADC）"],
    ]
    table(slide, 0.75, 1.66, 11.83, rows, [0.42, 0.58], row_h=0.53, body_size=11.5)
    bullet_card(slide, 0.75, 5.5, 5.88, 1.42, "结论一", [
        "保护层只抑制“瞬时跳动”这一类错误修正；",
        "对承载能力的判定、卸载响应、小信号检测均无实质影响。",
    ], accent=FIX, fill=RGBColor(0xEE, 0xF6, 0xF1), head_size=13, body_size=11.5)
    bullet_card(slide, 6.7, 5.5, 5.88, 1.42, "结论二", [
        "关闭保护层时行为与现有算法完全一致；",
        "可作为可选功能安全引入，不存在对既有交付的回退风险。",
    ], accent=INK, head_size=13, body_size=11.5)
    return slide


def s13_roadmap(prs):
    slide = slide_base(prs, "建议", "从验证到现场定版的落地路径", 13)
    steps = [
        ("① 固件引入", "在固件中引入该保护层并保留开关，便于现场对照验证。", INK),
        ("② 场景启用", "长时间恒压场景优先启用：这类场景干扰频率高，累积影响明显。", INK_SOFT),
        ("③ 参数标定", "确认时长等参数与噪声水平、操作速度相关，需在不同传感器部位分别确认后定版。", GOLD),
        ("④ 后续补充", "补充“带操作时刻标记”的测试数据，量化保护层对真实小载荷操作的响应延迟。", GREY),
    ]
    x0, w, gap = 0.75, 2.85, 0.19
    for i, (head, body, accent) in enumerate(steps):
        x = x0 + i * (w + gap)
        bx, by = card(slide, x, 1.66, w, 2.0, accent=accent)
        frame = textbox(slide, bx, by + 0.18, w - 0.5, 0.34)
        write(frame, [(head, 15, True, accent)])
        frame = textbox(slide, bx, by + 0.64, w - 0.5, 1.24)
        write(frame, [(body, 11.5, False, TEXT, {"line": 1.32})])
    bullet_card(slide, 0.75, 3.98, 11.83, 2.6, "结论与当前状态", [
        "该保护层有效解决了瞬时干扰导致的基线永久偏移；对真实载荷变化、卸载响应、小信号检测均无实质影响；"
        "关闭保护层时行为与现有算法完全一致。",
        "当前已验证内容：现场数据的通道级归因、受控复现（三种扰动形态）、防护启用前后的长期偏移对比、"
        "小幅度卸载与长时间恒载的回归对比。",
        "待确认事项：保护参数的现场标定口径（确认时长与噪声水平、操作速度的对应关系）；"
        "不同传感器部位的分别确认结果。",
        "数据与图件均可本地复现：全部图件由现场录制数据直接计算生成，脚本与原始数据随交付一并提供。",
    ], accent=INK, head_size=15, body_size=12)
    return slide


def s14_appendix(prs):
    slide = slide_base(prs, "附录 · 数据与图件清单", "全部图件由现场录制数据直接计算生成，可在本地复现", 14)
    rows = [
        ["编号", "内容", "本说明中的位置"],
        ["图 1", "现象总览：原始读数与补偿输出对比，显示跳动位置与基线抬升", "第 4 页"],
        ["图 2", "局部放大：事件前后基线的台阶对比", "第 5 页"],
        ["图 3", "通道影响：跳动幅度与补偿量变化的通道级对应", "第 7 页"],
        ["图 4", "出现次数：同类跳动在不同测试中的分布", "第 8 页"],
        ["图 5", "两种变化的形态对比：来了又走 / 来了就留下", "第 9 页"],
        ["图 6", "受控复现：不同扰动形态是否造成永久偏移", "第 11 页"],
        ["图 7", "修复对比：保护层启用前后的长期偏移", "第 11 页"],
    ]
    table(slide, 0.75, 1.66, 11.83, rows, [0.10, 0.62, 0.28], row_h=0.47, body_size=11.5,
          right_cols=1)
    rect(slide, 0.75, 5.66, 11.83, 0.92, fill=PAPER, radius=0.1)
    frame = textbox(slide, 1.0, 5.78, 11.33, 0.7)
    write(frame, [("配套文档：《瞬态干扰基线偏移问题说明（甲方版）》；"
                   "原始数据：左拇指指腹恒载录制 2026-10-14；验证脚本与结果文件随交付提供。",
                   11.5, False, GREY, {"line": 1.3})])
    return slide


def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for fn in (s01_cover, s02_summary, s03_background, s04_phenomenon, s05_zoom,
               s06_cause, s07_channels, s08_frequency, s09_solution, s10_principles,
               s11_effects, s12_regression, s13_roadmap, s14_appendix):
        fn(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(json.dumps({"out": str(path), "bytes": path.stat().st_size,
                      "slides": len(Presentation(str(path)).slides._sldIdLst)},
                     ensure_ascii=False, indent=1))
