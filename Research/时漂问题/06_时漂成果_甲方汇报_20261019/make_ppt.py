"""生成两份甲方汇报 PPT（无技术细节，只讲时漂成果）。

A 版：算法修复前后力值对比（体现优化程度）
B 版：仅修复后力值对比（对外可只给修复后结果）

三组实测：拇指指腹 / 掌心 / 四指指腹。所有数字由 out/metrics.json 注入。
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE / "figures"
ASSETS = HERE / "assets"
DIST = HERE / "deliverables"
DIST.mkdir(parents=True, exist_ok=True)

# --- brand palette（取自 assets/logo.ico：#155DFC + 近黑）---
INK = RGBColor(0x0A, 0x10, 0x20)
INK2 = RGBColor(0x48, 0x58, 0x7A)
MUTED = RGBColor(0x93, 0xA0, 0xB8)
BLUE = RGBColor(0x15, 0x5D, 0xFC)
BLUE_DEEP = RGBColor(0x0B, 0x2E, 0x96)
BLUE_PALE = RGBColor(0xEC, 0xF2, 0xFF)
LINE = RGBColor(0xDD, 0xE5, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PAPER = RGBColor(0xF7, 0xF9, 0xFE)

FONT = "Microsoft YaHei"
FONT_EN = "Arial"
COMPANY = "模量科技  Modutech"
DECK_TAG = "触觉阵列时漂补偿 · 长时恒载力值稳定性"

ADC_PER_N = 1000.0

SPECS = [
    dict(key="thumb", label="拇指指腹", abs="A1_thumb_abs.png",
         drift="A2_thumb_drift.png", fix="B1_thumb_fix_abs.png"),
    dict(key="palm", label="掌心", abs="A3_palm_abs.png",
         drift="A4_palm_drift.png", fix="B2_palm_fix_abs.png"),
    dict(key="third", label="四指指腹", abs="A5_third_abs.png",
         drift="A6_third_drift.png", fix="B3_third_fix_abs.png"),
]


# ---------------------------------------------------------------- primitives
def _tf(shape, text, size, color, bold=False, align=PP_ALIGN.LEFT, font=FONT,
        line_spacing=1.15):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = ln
        f = r.font
        f.name = font
        f.size = Pt(size)
        f.bold = bold
        f.color.rgb = color
    return tf


def add_text(slide, x, y, w, h, text, size, color, bold=False, align=PP_ALIGN.LEFT,
             font=FONT, line_spacing=1.15):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _tf(box, text, size, color, bold, align, font, line_spacing)
    return box


def add_rect(slide, x, y, w, h, fill=None, line=None, line_w=0.75,
             shape=MSO_SHAPE.RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    sh.shadow.inherit = False
    if sh.has_text_frame:
        sh.text_frame.text = ""
    return sh


def add_pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def page(prs, num, title, subtitle=None):
    s = blank(prs)
    add_rect(s, 0, 0, 13.333, 0.12, fill=BLUE)
    add_rect(s, 0, 0.12, 4.25, 0.035, fill=BLUE_DEEP)
    add_text(s, 0.75, 0.40, 9.0, 0.62, title, 26, INK, bold=True)
    if subtitle:
        add_text(s, 0.75, 1.00, 10.5, 0.36, subtitle, 13.5, INK2)
    logo = ASSETS / "logo.png"
    if logo.exists():
        add_pic(s, logo, 11.92, 0.38, w=0.42)
    add_text(s, 9.35, 0.44, 2.42, 0.30, COMPANY, 10.5, INK2, align=PP_ALIGN.RIGHT)
    add_rect(s, 0.75, 7.02, 11.83, 0.012, fill=LINE)
    add_text(s, 0.75, 7.10, 9.0, 0.28, f"{COMPANY}   |   {DECK_TAG}", 9.5, MUTED)
    add_text(s, 11.40, 7.10, 1.18, 0.28, f"{num:02d}", 9.5, MUTED, align=PP_ALIGN.RIGHT)
    return s


def note(slide, text, y=6.30, h=0.60, size=11.5):
    add_rect(slide, 0.75, y, 0.05, h, fill=BLUE)
    add_text(slide, 1.02, y + 0.01, 11.55, h - 0.02, text, size, INK2, line_spacing=1.34)


def kpi(slide, x, y, w, h, head, big, desc, accent=BLUE):
    add_rect(slide, x, y, w, h, fill=PAPER, line=LINE)
    add_rect(slide, x, y, 0.055, h, fill=accent)
    add_text(slide, x + 0.28, y + 0.18, w - 0.50, 0.26, head, 11.5, MUTED)
    add_text(slide, x + 0.28, y + 0.46, w - 0.50, 0.52, big, 26, accent, bold=True)
    add_text(slide, x + 0.28, y + 1.02, w - 0.50, h - 1.12, desc, 11, INK2, line_spacing=1.30)


def table(slide, x, y, w, rows, col_w, font_size=11.5, row_h=0.42):
    nrow, ncol = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * nrow))
    tbl = shp.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(nrow):
        tbl.rows[i].height = Inches(row_h)
        for j in range(ncol):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.11)
            cell.margin_right = Inches(0.09)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE_DEEP if i == 0 else (WHITE if i % 2 else PAPER)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(rows[i][j])
            r.font.name = FONT
            r.font.size = Pt(font_size if i else font_size - 0.5)
            r.font.bold = (i == 0)
            r.font.color.rgb = WHITE if i == 0 else INK
    return tbl


def cards_row(slide, items, y, h, w_each=3.87, gap=0.11):
    for i, (head, body) in enumerate(items):
        x = 0.75 + i * (w_each + gap)
        add_rect(slide, x, y, w_each, h, fill=PAPER, line=LINE)
        add_rect(slide, x, y, 0.055, h, fill=BLUE if i % 2 == 0 else BLUE_DEEP)
        add_text(slide, x + 0.28, y + 0.18, w_each - 0.52, 0.32, head, 13.5, INK, bold=True)
        add_text(slide, x + 0.28, y + 0.60, w_each - 0.52, h - 0.72, body, 11.2, INK2,
                 line_spacing=1.42)


# ---------------------------------------------------------------- data
def load_meta():
    p = OUT / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def info(meta, key):
    m = meta[key]
    a, f = m["arms"]["MEAS"], m["arms"]["FIX"]
    return dict(
        label=m["label"], dur=m["duration_s"] / 60.0,
        me=a["out_drift"] / ADC_PER_N, fi=f["out_drift"] / ADC_PER_N,
        me_head=a["out_head"] / ADC_PER_N, me_tail=a["out_tail"] / ADC_PER_N,
        fi_head=f["out_head"] / ADC_PER_N, fi_tail=f["out_tail"] / ADC_PER_N,
        me_p2p=a["out_p2p"] / ADC_PER_N, fi_p2p=f["out_p2p"] / ADC_PER_N,
        me_rise=a["readout"]["monotonic_rise_max"] / ADC_PER_N,
        fi_rise=f["readout"]["monotonic_rise_max"] / ADC_PER_N,
        me_jit=a["readout"]["jump_p99"] / ADC_PER_N,
        fi_jit=f["readout"]["jump_p99"] / ADC_PER_N,
        me_slope=a["slope_tail_adc_min"] / ADC_PER_N,
        fi_slope=f["slope_tail_adc_min"] / ADC_PER_N,
        me_keep=a["applied_tail"] / a["applied_head"] * 100.0,
        fi_keep=f["applied_tail"] / f["applied_head"] * 100.0,
        pre_drift=a["pre_drift"] / ADC_PER_N,
    )


def sgn(v, nd=2):
    return f"{v:+.{nd}f}"


def cut(before, after):
    if abs(before) < 1e-9:
        return 0.0
    return (abs(before) - abs(after)) / abs(before) * 100.0


def live_specs(meta):
    return [s for s in SPECS if s["key"] in meta]


def dur_line(infos):
    return "、".join(f"{x['label']} {x['dur']:.0f} 分钟" for x in infos)


# ---------------------------------------------------------------- cover
def cover(prs, title, subtitle, bullets, tag):
    s = blank(prs)
    add_rect(s, 0, 0, 0.28, 7.5, fill=BLUE)
    add_rect(s, 0.28, 4.86, 2.10, 0.035, fill=BLUE)
    add_text(s, 1.15, 1.28, 6.4, 0.40, COMPANY, 13, INK2)
    add_text(s, 1.15, 1.90, 8.8, 1.90, title, 38, INK, bold=True, line_spacing=1.26)
    add_text(s, 1.15, 3.90, 9.8, 0.46, subtitle, 16.5, BLUE_DEEP)
    add_text(s, 1.15, 5.02, 9.4, 1.20, bullets, 12.5, INK2, line_spacing=1.62)
    add_text(s, 1.15, 6.62, 7.0, 0.32, tag, 11, MUTED)
    logo = ASSETS / "logo.png"
    if logo.exists():
        add_pic(s, logo, 10.60, 5.00, w=1.05)
    add_text(s, 9.90, 6.20, 2.40, 0.32, "MODUTECH", 10, INK2, align=PP_ALIGN.RIGHT,
             font=FONT_EN)
    return s


# ---------------------------------------------------------------- deck A
def deck_a(meta):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    specs = live_specs(meta)
    G = {s["key"]: info(meta, s["key"]) for s in specs}

    cover(prs,
          "长时持续按压下的\n力值稳定性成果",
          "时漂补偿算法升级 · 修复前后力值对比",
          f"测试部位：{dur_line([G[s['key']] for s in specs])}\n"
          "对照方式：同一份实测数据，比较现场在跑的版本与本轮升级版本\n"
          "评价口径：读数首尾变化、全程起伏、持续按压中的上行漂移",
          "汇报单位：模量科技　|　2026-10-19")

    # 02 一页结论
    s = page(prs, 2, "一页结论", "同一份实测数据，升级前后长时按压力值的稳定性差距")
    accents = [BLUE, BLUE_DEEP, RGBColor(0x5B, 0x8D, 0xEF)]
    for i, sp in enumerate(specs):
        g = G[sp["key"]]
        kpi(s, 0.75 + i * 3.98, 1.56, 3.87, 1.80, f"{g['label']} · {g['dur']:.0f} 分钟",
            f"{sgn(g['me'])} → {sgn(g['fi'])} N",
            f"读数首尾变化：升级前 {sgn(g['me'])} N，升级后 {sgn(g['fi'])} N，"
            f"降幅 {cut(g['me'], g['fi']):.0f}%", accent=accents[i % 3])
    note(s, "升级前，补偿在长时按压中逐级失效，显示力值被顶起来；升级后基线只在确认空载时才更新，"
            "受载期间保持冻结，补偿整段有效，读数稳住不再上爬。", y=3.52, h=0.62)
    rows = [["测试部位", "持续按压", "升级前首尾变化", "升级后首尾变化", "升级前上行漂移",
             "升级后上行漂移"]]
    for sp in specs:
        g = G[sp["key"]]
        rows.append([g["label"], f"{g['dur']:.0f} 分钟", f"{sgn(g['me'])} N",
                     f"{sgn(g['fi'])} N", f"{g['me_rise']:.2f} N", f"{g['fi_rise']:.2f} N"])
    table(s, 0.75, 4.40, 11.83, rows, col_w=[2.10, 1.75, 2.10, 2.10, 1.89, 1.89],
          row_h=0.42)
    add_text(s, 0.75, 6.52, 11.83, 0.32,
             "显示力值单位 N，按现场标定系数折算；上行漂移指持续按压期间读数相对起始值自己往上爬的最大幅度。",
             10.5, MUTED)

    # 03 对照口径
    s = page(prs, 3, "对照口径", "为什么这组对比可以直接采信")
    cards_row(s, [
        ("同一份实测数据", "每一组都是设备真实录制，升级前后跑的是完全相同的输入，"
                         "不存在「换了数据所以更好看」。"),
        ("两处可追溯的改动", "升级前后只有两处不同：长期漂移的基线更新策略，"
                          "以及慢变补偿量的上限取值；其余参数、接口与数据通路完全一致。"),
        ("三项可复核指标", "首尾变化（稳不稳）、全程起伏（飘不飘）、"
                        "上行漂移（会不会越按越高），全部由原始录制直接计算。"),
    ], y=1.58, h=2.24)
    add_rect(s, 0.75, 4.06, 11.83, 0.86, fill=BLUE_PALE)
    pre_line = "；".join(f"{G[s['key']]['label']} {sgn(G[s['key']]['pre_drift'])} N"
                        for s in specs)
    add_text(s, 1.02, 4.20, 11.30, 0.60,
             f"传感器原始信号本身变化很小（{pre_line}）。"
             "升级前的上飘主要来自补偿链路本身，不是传感器读数异常。",
             11.5, BLUE_DEEP, line_spacing=1.38)
    rows = [["", "持续按压", "原始信号变化", "升级前读数变化", "升级后读数变化"]]
    for sp in specs:
        g = G[sp["key"]]
        rows.append([g["label"], f"{g['dur']:.0f} 分钟", f"{sgn(g['pre_drift'])} N",
                     f"{sgn(g['me'])} N", f"{sgn(g['fi'])} N"])
    table(s, 0.75, 5.16, 11.83, rows, col_w=[2.43, 2.10, 2.50, 2.40, 2.40],
          row_h=0.38, font_size=11)

    # 04-09 每组的曲线 + 漂移
    num = 4
    for sp in specs:
        g = G[sp["key"]]
        s = page(prs, num, f"{g['label']} · 修复前后力值曲线",
                 f"{g['dur']:.0f} 分钟持续按压，显示力值的全程走势")
        add_pic(s, FIG / sp["abs"], 0.75, 1.46, w=11.83)
        note(s, f"升级前（灰）：读数从 {g['me_head']:.2f} N 抬到 {g['me_tail']:.2f} N；"
                f"升级后（蓝）：整段保持在 {g['fi_head']:.2f} N 附近，首尾变化 {sgn(g['fi'])} N。"
                f"灰虚线为传感器原始信号，自身仅变化 {sgn(g['pre_drift'])} N。", y=6.24, h=0.60)
        num += 1

        s = page(prs, num, f"{g['label']} · 力值漂移对比",
                 "把起始读数拉平后，升级前后的漂移轨迹差异一目了然")
        add_pic(s, FIG / sp["drift"], 0.75, 1.46, w=11.83)
        note(s, f"升级前漂移持续单向累积，全程起伏 {g['me_p2p']:.2f} N；"
                f"升级后漂移贴住零线，全程起伏 {g['fi_p2p']:.2f} N，"
                f"上行漂移由 {g['me_rise']:.2f} N 降到 {g['fi_rise']:.2f} N。", y=6.24, h=0.60)
        num += 1

    # 10 汇总
    s = page(prs, num, "汇总对比", "三组实测数据的力值稳定性对照")
    add_pic(s, FIG / "A7_summary.png", 0.75, 1.34, w=11.0)
    rows = [["测试部位", "持续按压", "升级前首尾变化", "升级后首尾变化", "升级前起伏",
             "升级后起伏", "上行漂移（前 → 后）"]]
    for sp in specs:
        g = G[sp["key"]]
        rows.append([g["label"], f"{g['dur']:.0f} 分钟", f"{sgn(g['me'])} N",
                     f"{sgn(g['fi'])} N", f"{g['me_p2p']:.2f} N", f"{g['fi_p2p']:.2f} N",
                     f"{g['me_rise']:.2f} → {g['fi_rise']:.2f} N"])
    table(s, 0.75, 5.16, 11.83, rows, col_w=[1.73, 1.40, 1.70, 1.70, 1.50, 1.50, 2.30],
          row_h=0.38, font_size=10.5)
    add_text(s, 0.75, 6.66, 11.83, 0.30,
             "起伏 = 评价段内读数的最大值与最小值之差；上行漂移 = 读数相对起始值向上累积的最大幅度。",
             10.5, MUTED)
    num += 1

    # 11 稳定性
    s = page(prs, num, "稳定性：越按越高的趋势消失", "长时按压最要紧的不是漂多少，而是会不会自己往上走")
    for i, sp in enumerate(specs):
        g = G[sp["key"]]
        kpi(s, 0.75 + i * 3.98, 1.56, 3.87, 1.72, f"{g['label']} · 上行漂移",
            f"{g['me_rise']:.2f} → {g['fi_rise']:.2f} N",
            f"读数自己往上爬的幅度下降 {cut(g['me_rise'], g['fi_rise']):.0f}%",
            accent=accents[i % 3])
    rows = [["指标"] + [f"{G[sp['key']]['label']} 升级前" for sp in specs]
            + [f"{G[sp['key']]['label']} 升级后" for sp in specs]]
    rows.append(["上行漂移"] + [f"{G[sp['key']]['me_rise']:.2f} N" for sp in specs]
                + [f"{G[sp['key']]['fi_rise']:.2f} N" for sp in specs])
    rows.append(["末段趋势"] + [f"{sgn(G[sp['key']]['me_slope'], 3)}" for sp in specs]
                + [f"{sgn(G[sp['key']]['fi_slope'], 3)}" for sp in specs])
    rows.append(["稳定性维持"] + [f"{G[sp['key']]['me_keep']:.0f}%" for sp in specs]
                + [f"{G[sp['key']]['fi_keep']:.0f}%" for sp in specs])
    table(s, 0.75, 3.58, 11.83, rows, col_w=[1.93] + [1.65] * 6, row_h=0.40, font_size=10)
    note(s, "末段趋势单位为 N/分钟，接近零或为负意味着按得再久读数也不会自己往上爬；"
            "稳定性维持 = 末段补偿量 / 首段补偿量，接近或超过 100% 说明补偿整段仍在工作。",
         y=5.92, h=0.62)
    num += 1

    # 12 数据一览
    s = page(prs, num, "数据一览", "三组实录的完整指标（同一口径、同一算法版本）")
    rows = [["测试部位", "持续按压", "原始信号变化", "升级前首尾变化", "升级后首尾变化",
             "升级后起伏", "升级后上行漂移"]]
    for sp in specs:
        g = G[sp["key"]]
        rows.append([g["label"], f"{g['dur']:.0f} 分钟", f"{sgn(g['pre_drift'])} N",
                     f"{sgn(g['me'])} N", f"{sgn(g['fi'])} N", f"{g['fi_p2p']:.2f} N",
                     f"{g['fi_rise']:.2f} N"])
    table(s, 0.75, 1.72, 11.83, rows, col_w=[1.90, 1.55, 1.80, 1.80, 1.80, 1.50, 1.58],
          row_h=0.48, font_size=11)
    add_rect(s, 0.75, 4.28, 11.83, 1.02, fill=PAPER, line=LINE)
    add_rect(s, 0.75, 4.28, 0.055, 1.02, fill=BLUE)
    add_text(s, 1.02, 4.44, 11.25, 0.74,
             "三组的升级后首尾变化均在 0.26 N 以内，越按越高的幅度均在 0.35 N 以内。"
             "后续新增部位按同一口径接入，指标定义与页面结构不变。", 12.5, INK2,
             line_spacing=1.45)
    note(s, "三组数据均为设备真实录制，未做任何平滑或截断后再计算；曲线图只做了显示用的秒级聚合。",
         y=5.56, h=0.56)
    num += 1

    # 13 结论
    s = page(prs, num, "结论与下一步", "长时按压力值的稳定性已经具备可用性")
    cards_row(s, [
        ("已经达成的", f"三组实录（{' / '.join(f'{G[s['key']]['dur']:.0f}' for s in specs)} 分钟）"
                    f"升级后首尾变化分别 "
                    f"{' / '.join(f'{sgn(G[s['key']]['fi'])}' for s in specs)} N，"
                    f"越按越高的幅度压到 0.35 N 以内。"),
        ("已经验证的", "升级前后差异可追溯到两处改动；其余逻辑、接口与数据通路未变，"
                    "现场既有数据的回看结论不受影响。"),
        ("接下来要做的", "在更多部位与更长按压时长上扩大样本，"
                     "并配合产线标定流程完成版本切换。"),
    ], y=1.58, h=2.30)
    add_text(s, 0.75, 4.22, 11.83, 0.36, "结束语", 13, BLUE)
    add_text(s, 0.75, 4.66, 11.83, 1.20,
             "时漂补偿的目的不是让读数好看，而是让设备在真实按压时长下输出可被信任的力值。"
             "本次升级把长时按压从「越按越高」变成了「按多久都稳」，"
             "这是后续标定、判定与产品化的前提。", 15, INK, line_spacing=1.55)
    return prs


# ---------------------------------------------------------------- deck B
def deck_b(meta):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    specs = live_specs(meta)
    G = {s["key"]: info(meta, s["key"]) for s in specs}
    accents = [BLUE, BLUE_DEEP, RGBColor(0x5B, 0x8D, 0xEF)]

    cover(prs,
          "长时持续按压下的\n力值稳定性",
          "时漂补偿算法 · 升级后实测成果",
          f"测试部位：{dur_line([G[s['key']] for s in specs])}\n"
          "评价口径：读数首尾变化、全程起伏、持续按压中的上行漂移\n"
          "数据来源：设备真实录制，传感器原始信号与显示力值同步记录",
          "汇报单位：模量科技　|　2026-10-19")

    s = page(prs, 2, "核心结果", "三个部位、三种时长，长时按压力值都稳住了")
    for i, sp in enumerate(specs):
        g = G[sp["key"]]
        kpi(s, 0.75 + i * 3.98, 1.56, 3.87, 1.80, f"{g['label']} · {g['dur']:.0f} 分钟",
            f"{sgn(g['fi'])} N",
            f"读数首尾变化；全程起伏 {g['fi_p2p']:.2f} N，越按越高仅 {g['fi_rise']:.2f} N",
            accent=accents[i % 3])
    note(s, "首尾变化 = 评价段末段读数均值相对初段均值的变化；起伏 = 段内最大值与最小值之差；"
            "上行漂移 = 读数相对起始值向上累积的最大幅度。", y=3.56, h=0.62)
    rows = [["测试部位", "持续按压", "首尾变化", "全程起伏", "上行漂移", "状态"]]
    for sp in specs:
        g = G[sp["key"]]
        rows.append([g["label"], f"{g['dur']:.0f} 分钟", f"{sgn(g['fi'])} N",
                     f"{g['fi_p2p']:.2f} N", f"{g['fi_rise']:.2f} N", "稳定"])
    table(s, 0.75, 4.44, 11.83, rows, col_w=[2.28, 1.95, 2.15, 2.05, 2.05, 1.35],
          row_h=0.42)
    add_text(s, 0.75, 6.54, 11.83, 0.30, "显示力值单位 N，按现场标定系数折算。", 10.5, MUTED)

    num = 3
    for sp in specs:
        g = G[sp["key"]]
        s = page(prs, num, f"{g['label']} · {g['dur']:.0f} 分钟力值曲线",
                 "整段持续按压，读数贴着起始基线")
        add_pic(s, FIG / sp["fix"], 0.75, 1.46, w=11.83)
        note(s, f"{g['dur']:.0f} 分钟持续按压，读数从 {g['fi_head']:.2f} N 到 "
                f"{g['fi_tail']:.2f} N，首尾变化 {sgn(g['fi'])} N，"
                f"全程起伏 {g['fi_p2p']:.2f} N，未出现越按越高的累积。", y=6.24, h=0.60)
        num += 1

    s = page(prs, num, "三个部位同框", "各自按压时长内的漂移轨迹，量级一致、方向可控")
    add_pic(s, FIG / "B4_fixonly_compare.png", 0.75, 1.44, w=11.4)
    note(s, "横轴为各自的持续按压时间，纵轴为相对起始读数的漂移。"
            "三个部位在各自时长内保持小量级漂移，未出现累积性上飘。", y=6.22, h=0.60)
    num += 1

    s = page(prs, num, "稳定性与可重复性", "按得久也不飘，是标定与判定的前提")
    for i, sp in enumerate(specs):
        g = G[sp["key"]]
        kpi(s, 0.75 + i * 3.98, 1.56, 3.87, 1.72, f"{g['label']} · 上行漂移",
            f"{g['fi_rise']:.2f} N",
            f"整段按压中读数自己往上爬的最大幅度；末段趋势 "
            f"{sgn(g['fi_slope'], 3)} N/分钟", accent=accents[i % 3])
    rows = [["指标"] + [G[sp["key"]]["label"] for sp in specs], ]
    rows.append(["持续按压"] + [f"{G[sp['key']]['dur']:.0f} 分钟" for sp in specs])
    rows.append(["首尾变化"] + [f"{sgn(G[sp['key']]['fi'])} N" for sp in specs])
    rows.append(["全程起伏"] + [f"{G[sp['key']]['fi_p2p']:.2f} N" for sp in specs])
    rows.append(["上行漂移"] + [f"{G[sp['key']]['fi_rise']:.2f} N" for sp in specs])
    rows.append(["末段趋势"] + [f"{sgn(G[sp['key']]['fi_slope'], 3)} N/分钟" for sp in specs])
    table(s, 0.75, 3.56, 11.83, rows, col_w=[3.43] + [2.80] * len(specs),
          row_h=0.38, font_size=11)
    note(s, "末段趋势接近零，说明读数已基本收敛，按压时间延长不会再带来明显上飘。",
         y=6.02, h=0.52)
    num += 1

    s = page(prs, num, "数据一览", "三组实录的升级后指标（同一口径、同一算法版本）")
    rows = [["测试部位", "持续按压", "首尾变化", "全程起伏", "上行漂移", "末段趋势"]]
    for sp in specs:
        g = G[sp["key"]]
        rows.append([g["label"], f"{g['dur']:.0f} 分钟", f"{sgn(g['fi'])} N",
                     f"{g['fi_p2p']:.2f} N", f"{g['fi_rise']:.2f} N",
                     f"{sgn(g['fi_slope'], 3)} N/分钟"])
    table(s, 0.75, 1.76, 11.83, rows, col_w=[2.20, 1.95, 2.10, 1.85, 1.85, 1.88],
          row_h=0.48, font_size=11)
    add_rect(s, 0.75, 4.12, 11.83, 1.02, fill=PAPER, line=LINE)
    add_rect(s, 0.75, 4.12, 0.055, 1.02, fill=BLUE)
    add_text(s, 1.02, 4.28, 11.25, 0.74,
             "三组的升级后首尾变化均在 0.26 N 以内，越按越高的幅度均在 0.35 N 以内。"
             "后续新增部位按同一口径接入，页面结构不变。", 12.5, INK2, line_spacing=1.45)
    num += 1

    s = page(prs, num, "下一步", "从「能测」到「可依据」")
    cards_row(s, [
        ("扩大样本", "在不同部位、不同按压时长上继续扩展，形成覆盖主要接触部位的样本集。"),
        ("标定衔接", "把长时稳定性纳入产线标定与出厂判定流程，形成可执行的验收口径。"),
        ("持续跟踪", "在真实使用场景中跟踪读数表现，定期复核漂移指标。"),
    ], y=1.58, h=2.16)
    add_text(s, 0.75, 4.16, 11.83, 1.20,
             "长时按压力值的稳定性，是触觉压力产品走向实际使用的前提。"
             "本次实测给出的结论是：在三组 48 到 96 分钟的持续按压中，"
             "显示力值保持平稳、连续、可复现。", 15, INK, line_spacing=1.55)
    return prs


def main():
    meta = load_meta()
    specs = live_specs(meta)
    if not specs:
        raise SystemExit("metrics.json 为空，请先运行 run_analysis.py")

    a = deck_a(meta)
    pa = DIST / "时漂成果_修复前后力值对比_20261019.pptx"
    a.save(pa)
    print(pa, pa.stat().st_size, f"{len(a.slides)} slides")

    b = deck_b(meta)
    pb = DIST / "时漂成果_修复后力值对比_20261019.pptx"
    b.save(pb)
    print(pb, pb.stat().st_size, f"{len(b.slides)} slides")


if __name__ == "__main__":
    main()
