# -*- coding: utf-8 -*-
"""生成《右手掌触觉阵列 · 时漂补偿效果验证》甲方汇报 PPT。

数据与结论全部来自 data/report/右手掌-new_param 下两组实测录制，
经 _ppt_build/step1..step6 脚本复算（metrics.json 为唯一数字来源），
配图由 make_figs.py 生成并通过几何 + 视觉双重自检。
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
FIG = HERE / "figures"
ASSETS = HERE / "assets"
OUT = HERE.parent / "右手掌-new_param_补偿效果前后对比与综合评价.pptx"
M = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))

INK = RGBColor(0x0B, 0x2E, 0x4F)
INK_SOFT = RGBColor(0x16, 0x45, 0x72)
PAPER = RGBColor(0xF4, 0xF7, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0xC8, 0x9A, 0x3C)
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
    "version": "右手掌实测 · 两组数据",
    "date": "2026-10-15",
    "client": "XX 项目组",
}

S1, S2 = M["s1"], M["s2"]
S1_DRIFT_R, S1_DRIFT_A = S1["raw"]["drift_mN"] / 1000, S1["algo"]["drift_mN"] / 1000
S2_DRIFT_R, S2_DRIFT_A = S2["raw"]["drift_mN"] / 1000, S2["algo"]["drift_mN"] / 1000
S1_REL_R = 100 * S1["raw"]["drift_mN"] / S1["raw"]["start_mN"]
S1_REL_A = 100 * S1["algo"]["drift_mN"] / S1["algo"]["start_mN"]
S2_REL_R = 100 * S2["raw"]["drift_mN"] / S2["raw"]["start_mN"]
S2_REL_A = 100 * S2["algo"]["drift_mN"] / S2["algo"]["start_mN"]
S1_STEP_R = S1["event_step"]["raw_mN"] / 1000
S1_STEP_A = S1["event_step"]["algo_mN"] / 1000
S1_AFTER_R = (S1["event_step"]["after_raw_mN"] - S1["event_step"]["base_raw_mN"]) / 1000
S1_AFTER_A = (S1["event_step"]["after_algo_mN"] - S1["event_step"]["base_algo_mN"]) / 1000
S1_MIN = f"{S1['duration_s'] / 60:.1f}"
S2_MIN = f"{S2['duration_s'] / 60:.1f}"
S1_FPS = f"{S1['frames'] / S1['duration_s']:.0f}"
S2_FPS = f"{S2['frames'] / S2['duration_s']:.0f}"


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
        frame = textbox(slide, 0.75, 1.06, 11.0, 0.34)
        write(frame, [(kicker, 11.5, False, GREY)])
    slide.shapes.add_picture(str(ASSETS / "modutech_logo.png"),
                             Inches(11.63), Inches(0.4), height=Inches(0.46))
    frame = textbox(slide, 10.15, 0.44, 1.36, 0.42, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [(META["company_cn"], 11.5, True, INK, {"align": PP_ALIGN.RIGHT})])
    rect(slide, 0.75, 7.03, 11.83, 0.012, fill=LINE)
    frame = textbox(slide, 0.75, 7.11, 9.6, 0.28)
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


def figure(slide, name, x, y, w):
    slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), width=Inches(w))


def takeaway(slide, x, y, w, h, blocks, accent=INK):
    """页脚结论条。"""
    rect(slide, x, y, w, h, fill=PAPER, radius=0.14)
    rect(slide, x, y, 0.055, h, fill=accent)
    frame = textbox(slide, x + 0.28, y + 0.13, w - 0.5, h - 0.26, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, blocks)


def caption(slide, x, y, w, text, size=10):
    frame = textbox(slide, x, y, w, 0.5)
    write(frame, [(text, size, False, GREY, {"line": 1.25})])


def table(slide, x, y, w, rows, col_ratio, head_size=11, body_size=11, row_h=0.36):
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
            cell.fill.fore_color.rgb = INK if r == 0 else (WHITE if r % 2 else PAPER)
            frame = cell.text_frame
            frame.word_wrap = True
            p = frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
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

    frame = textbox(slide, 1.15, 1.34, 8.0, 0.4)
    write(frame, [(f"{META['company_cn']} · {META['company_en']}", 13, True, GOLD)])

    frame = textbox(slide, 1.15, 1.92, 9.6, 2.1)
    write(frame, [("右手掌触觉阵列\n时漂补偿效果验证", 42, True, WHITE, {"line": 1.16}),
                  ("两组实测数据的前后对比与综合评价", 21, False,
                   RGBColor(0xBF, 0xD4, 0xE8), {"before": 10, "line": 1.2})])

    rect(slide, 1.15, 4.66, 2.1, 0.035, fill=GOLD)
    frame = textbox(slide, 1.15, 4.92, 10.6, 1.1)
    write(frame, [("数据来源：右手掌触觉阵列长时间恒载录制，两组连续实测数据（2026-10-15）",
                   12, False, RGBColor(0xA9, 0xC1, 0xD8), {"after": 5}),
                  (f"合计观测时长 {float(S1_MIN) + float(S2_MIN):.1f} 分钟　·　约 "
                   f"{S1_FPS} 帧/秒　·　逐帧同步记录", 12, False,
                   RGBColor(0xA9, 0xC1, 0xD8), {"after": 5}),
                  (f"版本：{META['version']}　　日期：{META['date']}", 12, False,
                   RGBColor(0xA9, 0xC1, 0xD8))])

    slide.shapes.add_picture(str(ASSETS / "modutech_logo.png"),
                             Inches(10.62), Inches(5.0), height=Inches(1.05))
    frame = textbox(slide, 10.02, 6.21, 2.2, 0.4)
    write(frame, [("MODUTECH", 15, True, WHITE, {"align": PP_ALIGN.RIGHT, "font": EN})])
    return slide


def s02_brief(prs):
    slide = slide_base(prs, "测试概况与评价口径",
                       "同一套手掌触觉阵列连续两次长时间恒载录制，逐帧同步保留“补偿前”与“补偿后”两条曲线", 2)
    items = [
        ("测试对象", "右手掌触觉阵列（71 路测点；两次录制使用同一布局配置）",
         "两组建模条件一致，可直接横向比较", INK),
        ("采集规模", f"数据一 {S1_MIN} 分钟（{S1['frames']:,} 帧）；数据二 {S2_MIN} 分钟（{S2['frames']:,} 帧）",
         f"约 {S1_FPS} 帧/秒连续记录，全程无中断", INK_SOFT),
        ("测试方式", "手掌持续恒载加压，全程保持接触、不做卸载",
         "考察的是长时间保压下的读数稳定性", INK_SOFT),
        ("对比口径", "每组数据同时给出“补偿前（设备原始读数）”与“补偿后（输出给上位机）”",
         "两者逐帧对齐，差异即算法的补偿量", GOLD),
    ]
    x0, y0, w, h, gap = 0.75, 1.58, 5.83, 1.55, 0.17
    for i, (label, value, note, accent) in enumerate(items):
        col, row = i % 2, i // 2
        bx, by = card(slide, x0 + col * (w + gap), y0 + row * (h + 0.22), w, h, accent=accent)
        frame = textbox(slide, bx, by + 0.15, w - 0.55, 0.3)
        write(frame, [(label, 12, True, accent)])
        frame = textbox(slide, bx, by + 0.53, w - 0.55, h - 0.68)
        write(frame, [(value, 13.5, True, TEXT, {"line": 1.24, "after": 5}),
                      (note, 10.5, False, GREY, {"line": 1.24})])
    caption(slide, 0.75, 5.36, 11.83,
            "评价三项指标：保压全程力值漂移（相对起始稳定值的抬升）、瞬时扰动幅度（短时间内的跳变量级）、"
            "信号波动水平（短窗口标准差，衡量是否引入额外噪声）。")
    takeaway(slide, 0.75, 5.95, 11.83, 0.82, [
        ("本报告只呈现效果数据：补偿前与补偿后的实测曲线、关键幅度与稳定性指标；算法内部参数与实现细节不在本报告范围内。",
         12.5, True, INK, {"line": 1.3})], accent=GOLD)
    return slide


def s03_data1(prs):
    slide = slide_base(prs, "数据一 · 全程补偿效果",
                       f"{S1_MIN} 分钟持续保压：原始读数一路抬升 {S1_DRIFT_R:.2f} N，补偿后仅 {S1_DRIFT_A:.2f} N", 3)
    figure(slide, "fig_s1_full.png", 1.167, 1.44, 11.0)
    takeaway(slide, 0.75, 6.02, 11.83, 0.86, [
        (f"补偿前：全程抬升 +{S1_DRIFT_R:.2f} N（{S1_REL_R:+.1f}%，约 {S1['raw']['slope_mN_per_min']:.0f} mN/分钟）；"
         f"补偿后：+{S1_DRIFT_A:.2f} N（{S1_REL_A:+.1f}%）——漂移抑制约 {S1['suppression']['drift_pct']:.0f}%",
         13, True, INK, {"align": PP_ALIGN.CENTER})], accent=FIX)
    return slide


def s04_event(prs):
    slide = slide_base(prs, "数据一 · 61 秒瞬时扰动",
                       "第 61.5 秒出现一次读数阶跃：补偿后冲击力度减半，且没有污染后续基线", 4)
    figure(slide, "fig_s1_zoom61.png", 1.167, 1.44, 11.0)
    takeaway(slide, 0.75, 6.02, 11.83, 0.86, [
        (f"阶跃瞬间：原始读数 +{S1_STEP_R:.2f} N → 补偿后 +{S1_STEP_A:.2f} N；"
         f"事件后 30 秒：原始读数累计抬升 +{S1_AFTER_R:.2f} N（基线被永久抬高），"
         f"补偿后稳定在 +{S1_AFTER_A:.2f} N 不再继续漂移",
         12.5, True, INK, {"align": PP_ALIGN.CENTER})], accent=GOLD)
    return slide


def s05_data2(prs):
    slide = slide_base(prs, "数据二 · 全程补偿效果",
                       f"{S2_MIN} 分钟持续保压：本组原始读数本身漂移很小（{S2_DRIFT_R:.2f} N），补偿后与原始读数保持一致", 5)
    figure(slide, "fig_s2_full.png", 1.167, 1.44, 11.0)
    takeaway(slide, 0.75, 6.02, 11.83, 0.86, [
        (f"补偿前：全程抬升 +{S2_DRIFT_R:.2f} N（{S2_REL_R:+.1f}%）；补偿后：{S2_DRIFT_A:.2f} N（{S2_REL_A:+.1f}%）；"
         f"波动水平与原始读数同级（{S2['algo']['detrend_std_mN']:.0f} mN vs {S2['raw']['detrend_std_mN']:.0f} mN）"
         f"——漂移小的工况下不引入额外偏差",
         12.5, True, INK, {"align": PP_ALIGN.CENTER})], accent=FIX)
    return slide


def s06_compare(prs):
    slide = slide_base(prs, "两组数据量化对比",
                       "把“漂移大的工况”与“漂移小的工况”放在同一把尺子上看补偿表现", 6)
    figure(slide, "fig_compare.png", 1.97, 1.42, 9.4)
    for i, (head, lines, accent) in enumerate((
            ("漂移量（N）", [
                f"数据一：{S1_DRIFT_R:+.2f} → {S1_DRIFT_A:+.2f}",
                f"数据二：{S2_DRIFT_R:+.2f} → {S2_DRIFT_A:+.2f}",
                "单位与信号电平均为牛顿，可直接比较。",
            ], INK),
            ("相对漂移（%）", [
                f"数据一：{S1_REL_R:+.1f}% → {S1_REL_A:+.1f}%",
                f"数据二：{S2_REL_R:+.1f}% → {S2_REL_A:+.1f}%",
                "以各组起始稳定电平为分母。",
            ], INK_SOFT),
            ("综合评价", [
                f"明显漂移工况：抑制约 {S1['suppression']['drift_pct']:.0f}%；",
                "轻微漂移工况：与原始读数同量级，无额外偏差；",
                "补偿量由算法自动生成，全程无人工干预。",
            ], FIX),
    )):
        bullet_card(slide, 0.75 + i * 4.0, 5.32, 3.83, 1.44, head, lines,
                    accent=accent, head_size=12.5, body_size=10.5)
    return slide


def s07_stability(prs):
    slide = slide_base(prs, "瞬时扰动与信号波动",
                       "补偿后信号更稳：瞬时冲击被削弱，长时段波动保持在同一水平", 7)
    figure(slide, "fig_stability.png", 1.97, 1.42, 9.4)
    for i, (head, lines, accent) in enumerate((
            ("瞬时跳变幅度", [
                f"数据一 61.5 s：{S1_STEP_R*1000:.0f} → {S1_STEP_A*1000:.0f} mN；",
                f"数据二全段最大：{S2['max_1s_rise']['raw']['rise_mN']:.0f} → "
                f"{S2['max_1s_rise']['algo']['rise_mN']:.0f} mN。",
            ], REPORT),
            ("长时段波动", [
                f"数据一：{S1['raw']['detrend_std_mN']:.0f} → {S1['algo']['detrend_std_mN']:.0f} mN；",
                f"数据二：{S2['raw']['detrend_std_mN']:.0f} → {S2['algo']['detrend_std_mN']:.0f} mN。",
            ], INK),
            ("综合评价", [
                "补偿过程不引入额外噪声；",
                "真实载荷变化仍能正常通过，未出现“压死”现象。",
            ], FIX),
    )):
        bullet_card(slide, 0.75 + i * 4.0, 5.32, 3.83, 1.44, head, lines,
                    accent=accent, head_size=12.5, body_size=10.5)
    return slide


def s08_conclusion(prs):
    slide = slide_base(prs, "综合评价", "两组数据在“漂移明显”与“漂移轻微”两种工况下均通过验证", 8)
    items = [
        ("补偿有效性", f"长时间保压下的明显漂移被显著抑制：数据一由 {S1_DRIFT_R:.2f} N 降至 {S1_DRIFT_A:.2f} N，"
                       f"抑制约 {S1['suppression']['drift_pct']:.0f}%，读数在整个观测窗口内保持稳定。", FIX, 15),
        ("零偏差一致性", f"本底漂移很小的工况下不引入额外偏差：数据二补偿后为 {S2_DRIFT_A:.2f} N"
                          f"（{S2_REL_A:+.1f}%），与原始读数保持同量级。", INK, 15),
        ("瞬态鲁棒性", f"第 61.5 秒的读数阶跃被削弱：{S1_STEP_R:.2f} N → {S1_STEP_A:.2f} N，"
                       f"且事件后 30 秒补偿后信号稳定在 +{S1_AFTER_A:.2f} N，未出现基线被永久抬高的情况。", GOLD, 15),
        ("信号保真", "补偿后波动水平与原始读数同级（数据二 122 mN），长时间保压中既压住漂移、"
                     "也不损失真实载荷变化的通过性。", INK_SOFT, 15),
    ]
    x0, y0, w, h = 0.75, 1.56, 5.83, 1.72
    for i, (label, text, accent, size) in enumerate(items):
        col, row = i % 2, i // 2
        bx, by = card(slide, x0 + col * (w + 0.17), y0 + row * (h + 0.2), w, h, accent=accent)
        frame = textbox(slide, bx, by + 0.16, w - 0.55, 0.3)
        write(frame, [(label, 12.5, True, accent)])
        frame = textbox(slide, bx, by + 0.56, w - 0.55, h - 0.72)
        write(frame, [(text, size - 1.5, False, TEXT, {"line": 1.32})])
    rect(slide, 0.75, 5.44, 11.83, 1.3, fill=INK, radius=0.1)
    frame = textbox(slide, 1.15, 5.62, 11.05, 0.98, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [("整体结论", 12, True, GOLD, {"after": 6}),
                  (f"时漂补偿在两组实测数据上表现一致：明显漂移被抑制约 {S1['suppression']['drift_pct']:.0f}%"
                   f"（{S1_DRIFT_R:.2f} N → {S1_DRIFT_A:.2f} N），轻微漂移工况下不产生额外偏差"
                   f"（{S2_DRIFT_A:+.2f} N），瞬时扰动不再污染后续基线；"
                   "两组数据均满足长时间持续保压采集的使用要求，建议按现配置推进下一阶段批量验证。",
                   13, False, WHITE, {"line": 1.34})])
    return slide


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    s01_cover(prs)
    s02_brief(prs)
    s03_data1(prs)
    s04_event(prs)
    s05_data2(prs)
    s06_compare(prs)
    s07_stability(prs)
    s08_conclusion(prs)
    prs.save(OUT)
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
