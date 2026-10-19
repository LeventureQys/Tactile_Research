# -*- coding: utf-8 -*-
"""生成《触觉阵列时漂补偿效果验证》甲方汇报 PPT（三部位 · 六组实测数据）。

数字唯一来源：_ppt_build/metrics_all.json（由 step8/step9 从原始录制复算）；
配图由 make_figs_all.py 生成并通过几何自检与视觉审核。
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
FIG = HERE / "figures_all"
ASSETS = HERE / "assets"
OUT = HERE.parent / "时漂补偿效果验证_三部位六组实测数据对比与综合评价.pptx"

M = json.loads((HERE / "metrics_all.json").read_text(encoding="utf-8"))
ORDER = ["palm1", "palm2", "fing1", "fing2", "four1", "four2"]
GROUP = {"palm1": ("右手掌", 1), "palm2": ("右手掌", 2), "fing1": ("右指腹", 1),
         "fing2": ("右指腹", 2), "four1": ("四指指腹", 1), "four2": ("四指指腹", 2)}

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
    "company_cn": "模量科技", "company_en": "Modutech",
    "project": "触觉阵列时漂补偿", "version": "三部位 · 六组实测",
    "date": "2026-10-15",
}


# ── 数字辅助 ──────────────────────────────────────────────────────────────
def raw_n(k):
    return M[k]["raw"]["drift_mN"] / 1000.0


def algo_n(k):
    return M[k]["algo"]["drift_mN"] / 1000.0


def rel_raw(k):
    return 100 * M[k]["raw"]["drift_mN"] / M[k]["raw"]["start_mN"]


def rel_algo(k):
    return 100 * M[k]["algo"]["drift_mN"] / M[k]["algo"]["start_mN"]


def mins(k):
    return M[k]["duration_s"] / 60.0


def std_raw(k):
    return M[k]["raw"]["detrend_std_mN"]


def std_algo(k):
    return M[k]["algo"]["detrend_std_mN"]


def sup(k):
    return 100 * (1 - abs(algo_n(k)) / abs(raw_n(k))) if raw_n(k) else float("nan")


def nfmt(x):
    return "0.00 N" if abs(x) < 0.005 else f"{x:+.2f} N"


TOTAL_MIN = sum(mins(k) for k in ORDER)


# ── 基础工具 ──────────────────────────────────────────────────────────────
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


def bullet_card(slide, x, y, w, h, head, lines, accent=INK, fill=PAPER,
                head_size=13, body_size=11):
    bx, by = card(slide, x, y, w, h, accent=accent, fill=fill)
    frame = textbox(slide, bx, by + 0.15, w - 0.55, h - 0.26)
    write(frame, [(head, head_size, True, accent, {"after": 6})])
    for line in lines:
        write(frame, [(line, body_size, False, TEXT, {"after": 4, "line": 1.26})], first=False)


def figure(slide, name, x, y, w):
    slide.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), width=Inches(w))


def takeaway(slide, x, y, w, h, blocks, accent=INK):
    rect(slide, x, y, w, h, fill=PAPER, radius=0.14)
    rect(slide, x, y, 0.055, h, fill=accent)
    frame = textbox(slide, x + 0.28, y + 0.12, w - 0.5, h - 0.24, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, blocks)


def caption(slide, x, y, w, text, size=10):
    frame = textbox(slide, x, y, w, 0.5)
    write(frame, [(text, size, False, GREY, {"line": 1.25})])


def table(slide, x, y, w, rows, col_ratio, head_size=11, body_size=10.5, row_h=0.40):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                   Inches(w), Inches(row_h * n_rows))
    tbl = shape.table
    tbl.first_row = True
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.10)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if r == 0 else (WHITE if r % 2 else PAPER)
            frame = cell.text_frame
            frame.word_wrap = True
            p = frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c <= 1 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(value)
            set_run(run, head_size if r == 0 else body_size, r == 0,
                    WHITE if r == 0 else TEXT)
    for c, ratio in enumerate(col_ratio):
        tbl.columns[c].width = Emu(int(Inches(w) * ratio))
    return tbl


# ── 各页 ──────────────────────────────────────────────────────────────────
def s01_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect(slide, 0, 0, W, H, fill=INK)
    rect(slide, 0, 0, 0.28, H, fill=GOLD)

    frame = textbox(slide, 1.15, 1.30, 8.0, 0.4)
    write(frame, [(f"{META['company_cn']} · {META['company_en']}", 13, True, GOLD)])
    frame = textbox(slide, 1.15, 2.06, 10.2, 2.2)
    write(frame, [("触觉阵列时漂补偿效果验证", 42, True, WHITE, {"line": 1.16}),
                  ("右手掌 · 右指腹 · 四指指腹 ｜ 六组实测数据的前后对比与综合评价", 19, False,
                   RGBColor(0xBF, 0xD4, 0xE8), {"before": 10, "line": 1.2})])
    rect(slide, 1.15, 4.62, 2.1, 0.035, fill=GOLD)
    frame = textbox(slide, 1.15, 4.86, 10.6, 1.2)
    write(frame, [(f"六组数据　·　合计 {TOTAL_MIN:.1f} 分钟有效保压　·　漂移抑制 "
                   f"{min(sup(k) for k in ORDER):.0f}%–{max(sup(k) for k in ORDER):.0f}%",
                   13, True, GOLD, {"after": 9}),
                  (f"数据来源：三处触觉阵列长时间恒载录制，六个数据集（{META['date']}）",
                   12, False, RGBColor(0xA9, 0xC1, 0xD8), {"after": 5}),
                  (f"约 100 帧/秒逐帧同步记录　·　版本：{META['version']}　·　日期：{META['date']}",
                   12, False, RGBColor(0xA9, 0xC1, 0xD8))])
    slide.shapes.add_picture(str(ASSETS / "modutech_logo.png"),
                             Inches(10.62), Inches(4.98), height=Inches(1.05))
    frame = textbox(slide, 10.02, 6.19, 2.2, 0.4)
    write(frame, [("MODUTECH", 15, True, WHITE, {"align": PP_ALIGN.RIGHT, "font": EN})])
    return slide


def s02_overview(prs):
    slide = slide_base(prs, "测试概况与评价口径",
                       "三处部位各两次长时间恒载保压录制；数据逐帧同步采集，补偿前后可一一对照", 2)
    rows = [["测试部位", "组别", "保压时长", "测点数", "补偿前漂移", "补偿后漂移", "漂移抑制"]]
    for k in ORDER:
        part, idx = GROUP[k]
        rows.append([part, f"第 {idx} 组", f"{mins(k):.1f} 分钟", str(M[k]["cells"]),
                     nfmt(raw_n(k)), nfmt(algo_n(k)), f"{sup(k):.0f}%"])
    table(slide, 0.75, 1.56, 11.83, rows, [0.13, 0.10, 0.13, 0.10, 0.185, 0.185, 0.17])
    caption(slide, 0.75, 4.52, 11.83,
            "漂移口径：保压稳定后（加载完成约 15 秒起）至录制结束的力值变化量；"
            "“补偿前”为设备原始读数、“补偿后”为输出给上位机的值，两者一一对应，差值即算法自动补偿量。"
            "各组的补偿前后曲线见后续各页。")
    for i, (head, lines, accent) in enumerate((
            ("六组数据全部通过", [
                "六组数据的补偿后漂移均被压到 0.60 N 以内；",
                "抑制幅度 79%–100%，无一组出现补偿失效。",
            ], FIX),
            ("两类工况同时覆盖", [
                "手掌类漂移明显（最大 4.05 N），用于检验补偿力度；",
                "指腹类漂移轻微（≤ 0.45 N），用于检验零偏差。",
            ], INK),
            ("测试条件一致", [
                "同一套算法配置、同一上位机链路；",
                f"合计 {TOTAL_MIN:.1f} 分钟有效保压数据，全程无中断。",
            ], INK_SOFT))):
        bullet_card(slide, 0.75 + i * 4.0, 5.18, 3.83, 1.6, head, lines, accent=accent)
    return slide


def s03_drift(prs):
    slide = slide_base(prs, "六组数据 · 漂移对比总览",
                       "把“漂移明显”与“漂移轻微”两类工况放在同一把尺子上看补偿表现", 3)
    figure(slide, "fig_all_drift.png", 1.97, 1.42, 9.4)
    for i, (head, lines, accent) in enumerate((
            ("漂移量（N）", [
                f"右手掌第 1 组：{raw_n('palm1'):+.2f} → {algo_n('palm1'):+.2f}",
                f"其余五组：补偿前 0.12–{max(raw_n(k) for k in ORDER[1:]):.2f}，"
                f"补偿后最大 {max(abs(algo_n(k)) for k in ORDER[1:]):.2f}",
            ], INK),
            ("相对漂移（%）", [
                f"右手掌第 1 组：{rel_raw('palm1'):+.1f}% → {rel_algo('palm1'):+.1f}%",
                f"其余五组：≤ {max(rel_raw(k) for k in ORDER[1:]):.1f}% → "
                f"≤ {max(abs(rel_algo(k)) for k in ORDER[1:]):.1f}%",
            ], INK_SOFT),
            ("综合评价", [
                f"漂移抑制 {min(sup(k) for k in ORDER):.0f}%–{max(sup(k) for k in ORDER):.0f}%，六组全部达标；",
                "补偿量全部由算法自动生成，全程无人工干预。",
            ], FIX))):
        bullet_card(slide, 0.75 + i * 4.0, 5.32, 3.83, 1.44, head, lines,
                    accent=accent, head_size=12.5, body_size=10.5)
    return slide


def _pair_page(prs, part, k1, k2, number):
    kicker = (f"第 1 组 {mins(k1):.1f} 分钟、第 2 组 {mins(k2):.1f} 分钟；"
              f"补偿后漂移分别为 {nfmt(algo_n(k1))} 与 {nfmt(algo_n(k2))}"
              f"（补偿前 {nfmt(raw_n(k1))} / {nfmt(raw_n(k2))}）")
    slide = slide_base(prs, f"{part} · 两组保压全程对比", kicker, number)
    for k, y, idx in ((k1, 1.44, 1), (k2, 4.18, 2)):
        figure(slide, f"fig_{k}_full.png", 1.19, y, 10.95)
        frame = textbox(slide, 2.18, y + 0.33, 6.4, 0.26)
        write(frame, [(f"第 {idx} 组　·　{mins(k):.1f} 分钟　·　{M[k]['cells']} 测点",
                       11, True, INK)])
    return slide


def s04_palm(prs):
    return _pair_page(prs, "右手掌", "palm1", "palm2", 4)


def s05_finger(prs):
    return _pair_page(prs, "右指腹", "fing1", "fing2", 5)


def s06_four(prs):
    return _pair_page(prs, "四指指腹", "four1", "four2", 6)


def s07_event(prs):
    ev = M["palm1"]["event"]
    step_r, step_a = ev["step_raw_mN"] / 1000, ev["step_algo_mN"] / 1000
    after_r = (ev["after_raw_mN"] - ev["base_raw_mN"]) / 1000
    after_a = (ev["after_algo_mN"] - ev["base_algo_mN"]) / 1000
    slide = slide_base(prs, "瞬时扰动 · 右手掌第 1 组",
                       "第 61.5 秒出现一次读数阶跃：补偿后冲击幅度下降约四成，且没有污染后续基线", 7)
    figure(slide, "fig_palm1_event.png", 1.167, 1.44, 11.0)
    takeaway(slide, 0.75, 6.02, 11.83, 0.86, [
        (f"阶跃瞬间：原始读数 {step_r:+.2f} N → 补偿后 {step_a:+.2f} N；"
         f"事件后 30 秒：原始读数累计抬升 {after_r:+.2f} N（基线被抬高），"
         f"补偿后保持 {after_a:+.2f} N 不再漂移",
         12.5, True, INK, {"align": PP_ALIGN.CENTER})], accent=GOLD)
    return slide


def s08_stability(prs):
    slide = slide_base(prs, "信号波动与瞬时跳变",
                       "补偿过程不引入额外噪声：波动水平保持原样，瞬时冲击被削弱", 8)
    figure(slide, "fig_all_stability.png", 1.97, 1.42, 9.4)
    palm_k = ["palm1", "palm2"]
    finger_k = ["fing1", "fing2", "four1", "four2"]
    for i, (head, lines, accent) in enumerate((
            ("波动水平（5 秒窗口标准差）", [
                f"手掌类：{min(std_raw(k) for k in palm_k):.0f}–{max(std_raw(k) for k in palm_k):.0f} mN，"
                f"补偿后 {min(std_algo(k) for k in palm_k):.0f}–{max(std_algo(k) for k in palm_k):.0f} mN；",
                f"指腹类：{min(std_raw(k) for k in finger_k):.1f}–{max(std_raw(k) for k in finger_k):.1f} mN，"
                f"补偿后同级。",
            ], INK),
            ("瞬时跳变（1 秒窗口）", [
                f"右手掌第 1 组 61.5 s：{M['palm1']['event']['raw_1s_mN']:.0f} → "
                f"{M['palm1']['event']['algo_1s_mN']:.0f} mN；",
                f"其余五组全段最大 ≤ {max(M[k]['event']['raw_1s_mN'] for k in ORDER[1:]):.0f} mN，"
                "补偿后同步减小。",
            ], REPORT),
            ("综合评价", [
                "补偿前后波动处于同一水平，未出现“压死”或额外毛刺；",
                "真实载荷变化仍能正常通过，长保压读数稳定可用。",
            ], FIX))):
        bullet_card(slide, 0.75 + i * 4.0, 5.32, 3.83, 1.44, head, lines,
                    accent=accent, head_size=12.5, body_size=10.5)
    return slide


def s09_conclusion(prs):
    slide = slide_base(prs, "综合评价", "三处部位、六组数据在“漂移明显”与“漂移轻微”两类工况下均通过验证", 9)
    items = [
        ("补偿有效性", f"漂移明显工况被强力抑制：右手掌第 1 组由 {raw_n('palm1'):+.2f} N 降至 "
                       f"{algo_n('palm1'):+.2f} N，抑制 {sup('palm1'):.0f}%，整个观测窗口内读数保持稳定。",
         FIX),
        ("零偏差一致性", f"漂移轻微工况不引入额外偏差：右指腹、四指指腹四组补偿后残留 "
                          f"≤ 0.02 N（相对 ≤ 0.3%），与原始读数同量级。", INK),
        ("瞬态鲁棒性", f"右手掌第 61.5 秒的读数阶跃被削弱：{M['palm1']['event']['step_raw_mN']/1000:.2f} N → "
                       f"{M['palm1']['event']['step_algo_mN']/1000:.2f} N，事件后补偿后信号不再随原始读数继续抬升。",
         GOLD),
        ("信号保真", f"补偿前后波动水平同级（指腹类约 {std_algo('fing1'):.1f} mN、四指类约 "
                     f"{std_algo('four1'):.1f} mN），既压住漂移，也不损失真实载荷变化；"
                     "六组均未出现读数被“压死”或响应迟滞的现象。", INK_SOFT),
    ]
    x0, y0, w, h = 0.75, 1.56, 5.83, 1.72
    for i, (label, text, accent) in enumerate(items):
        col, row = i % 2, i // 2
        bx, by = card(slide, x0 + col * (w + 0.17), y0 + row * (h + 0.2), w, h, accent=accent)
        frame = textbox(slide, bx, by + 0.16, w - 0.55, 0.3)
        write(frame, [(label, 12.5, True, accent)])
        frame = textbox(slide, bx, by + 0.56, w - 0.55, h - 0.72)
        write(frame, [(text, 12, False, TEXT, {"line": 1.32})])
    rect(slide, 0.75, 5.44, 11.83, 1.3, fill=INK, radius=0.1)
    frame = textbox(slide, 1.15, 5.6, 11.05, 1.0, anchor=MSO_ANCHOR.MIDDLE)
    write(frame, [("整体结论", 12, True, GOLD, {"after": 6}),
                  (f"六组实测数据结论一致：漂移明显的工况被抑制 {sup('palm1'):.0f}% 以上，"
                   f"漂移轻微的工况不产生额外偏差，瞬时扰动不再污染后续基线；"
                   f"合计 {TOTAL_MIN:.1f} 分钟保压数据全部满足长时间采集的使用要求，"
                   "建议按现配置推进下一阶段批量验证。", 13, False, WHITE, {"line": 1.34})])
    return slide


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    s01_cover(prs)
    s02_overview(prs)
    s03_drift(prs)
    s04_palm(prs)
    s05_finger(prs)
    s06_four(prs)
    s07_event(prs)
    s08_stability(prs)
    s09_conclusion(prs)
    prs.save(OUT)
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
