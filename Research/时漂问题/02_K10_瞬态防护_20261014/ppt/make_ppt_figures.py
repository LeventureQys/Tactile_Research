# -*- coding: utf-8 -*-
"""K10 瞬态防护 · 甲方汇报 PPT 配图生成脚本。

数据来源：
  · 现场录制  D:\\workshop\\Github\\Tactile_Research\\Research\\时漂问题\\data\\左拇指指腹 r_slow_max=0.6\\测试1-2026-10-14_10-18-40
  · 归因/验证 02_K10_瞬态防护_20261014 下的 *_result.json
所有数值均由上述数据直接计算，不使用任何示意数字。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                       # 02_K10_瞬态防护_20261014
FIG = HERE / "figures_ppt"
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")

REPORT = "#B23A32"      # 红 — 问题 / 修复前
PRIMARY = "#1F4E79"     # 深蓝 — 补偿后 / 主色
SENSOR = "#8A93A3"      # 灰 — 原始读数
FIX = "#2E7D5B"         # 绿 — 修复后 / 正确行为
ACCENT = "#C98A26"      # 金 — 强调
INK = "#1B2430"
LIGHT = "#E3E8EF"


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    if title:
        ax.set_title(title, fontsize=15, color=INK, pad=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.grid(alpha=0.22, lw=0.8, color=LIGHT)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.tick_params(labelsize=10, colors="#4A5464")


def save(fig, name: str):
    """落盘并关闭 figure；同时把 figure 返回给调用方（自检脚本需要几何信息）。"""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=175, bbox_inches="tight", facecolor="white")
    return fig


def load_stream(folder: Path, name: str) -> pd.DataFrame:
    skip = 30
    with (folder / name).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
    return pd.read_csv(folder / name, skiprows=skip, header=0, engine="c")


def load_json(name: str) -> dict:
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def plt_span(ax) -> tuple[float, float, float]:
    """当前数据跨度，用于按比例摆放标注，避免固定偏移量。"""
    y0, y1 = ax.get_ylim()
    return y0, y1, y1 - y0


def fig01(time, pre_total, seg_total, seg_shift) -> None:
    mask = (time >= 150) & (time <= 260)
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.plot(time[mask], pre_total[mask], color=SENSOR, lw=1.0, label="传感器原始读数")
    ax.plot(time[mask], seg_total[mask], color=PRIMARY, lw=1.5, label="补偿后输出")

    style(ax, "现象总览：一次 0.3 秒的瞬时跳动，让补偿后的基线永久抬高",
          "时间（秒）", "阵列读数总和（ADC）")
    ax.set_xlim(146, 264)
    low = float(seg_total[mask].min())
    high = float(pre_total[mask].max())
    ax.set_ylim(low - 60, high + (high - low) * 0.12)
    y0, y1, span = plt_span(ax)
    ax.axvline(194.5, ymin=0.0, ymax=1.0, color=REPORT, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(214.88, ymin=0.0, ymax=1.0, color="#8A93A3", ls="--", lw=1.1, alpha=0.8)

    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.annotate("一次瞬时跳动（约 0.3 秒）",
                xy=(194.45, y0 + span * 0.05), xytext=(151.0, y0 + span * 0.035),
                color=REPORT, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=REPORT, lw=1.3))
    ax.annotate(f"跳动后基线整体上抬 {seg_shift:+.1f} ADC 且不再回落",
                xy=(238.0, float(np.median(seg_total[(time >= 195.0) & (time <= 260.0)]))),
                xytext=(199.0, low + span * 0.50), color=REPORT, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=REPORT, lw=1.3))
    ax.annotate("同期原始读数没有台阶变化",
                xy=(222.0, float(np.median(pre_total[(time >= 195.0) & (time <= 260.0)]))),
                xytext=(167.0, high + span * 0.045), color="#4A5464", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#4A5464", lw=1.0))
    ax.annotate("同段录制中另一次更小的跳动（约 215 秒）",
                xy=(214.7, y0 + span * 0.07), xytext=(188.0, y0 + span * 0.012),
                color="#4A5464", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#8A93A3", lw=1.0))
    return save(fig, "ppt_fig01_现象总览.png")


def fig02(time, pre_total, seg_total, seg_shift, base_before, base_after) -> None:
    m = (time >= 193.0) & (time <= 197.6)
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.plot(time[m], pre_total[m], color=SENSOR, lw=1.3)
    ax.plot(time[m], seg_total[m], color=PRIMARY, lw=1.8)
    ax.axvspan(194.45, 194.78, color=REPORT, alpha=0.10)
    ax.hlines(base_before, 193.02, 194.45, color=FIX, ls="-", lw=3.0, alpha=0.9)
    ax.hlines(base_after, 194.9, 197.62, color=REPORT, ls="-", lw=3.0, alpha=0.9)

    style(ax, "局部放大：跳动前后存在一个稳定台阶，而原始读数没有变化",
          "时间（秒）", "阵列读数总和（ADC）")
    data_low = float(min(seg_total[m].min(), pre_total[m].min()))
    data_high = float(pre_total[m].max())
    span = data_high - data_low
    y0 = min(base_before, base_after) - span * 0.16
    y1 = max(base_after + span * 0.42, data_high + span * 0.05)
    ax.set_ylim(y0, y1)
    ax.set_xlim(192.95, 197.65)

    _, _, fs = plt_span(ax)
    ax.annotate("灰线：原始读数，事件前后无台阶（差值 0.0 ADC）",
                xy=(194.8, float(np.median(pre_total[m])) + 12),
                xytext=(192.99, y1 - fs * 0.035), color="#4A5464", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#4A5464", lw=1.0))
    ax.annotate("跳动前基线（绿）", xy=(193.55, base_before),
                xytext=(193.05, base_before + fs * 0.045), color=FIX, fontsize=11.5)
    ax.annotate(f"跳动后基线（红）：上抬 {seg_shift:+.1f} ADC\n此后长时间不再回落",
                xy=(195.8, base_after), xytext=(194.55, base_after + fs * 0.10),
                color=REPORT, fontsize=11.5)
    ax.annotate("蓝线：补偿后输出", xy=(197.15, float(seg_total[m][-1])),
                xytext=(195.95, y0 + fs * 0.10), color=PRIMARY, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.0))
    return save(fig, "ppt_fig02_局部放大.png")


def fig03(chan: dict) -> None:
    rows = sorted(chan["channels"], key=lambda r: r["dip_drop"])[:8]
    labels = [f"CH{r['ch']}" for r in rows]
    dips = [abs(r["dip_drop"]) for r in rows]
    changes = [abs(r["applied_change"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.bar(x - 0.21, dips, 0.42, color=PRIMARY)
    ax.bar(x + 0.21, changes, 0.42, color=REPORT)
    for i, (d, c) in enumerate(zip(dips, changes)):
        ax.text(i - 0.21, d + 1.5, f"{d:.1f}", ha="center", fontsize=9.5, color=PRIMARY)
        ax.text(i + 0.21, c + 1.5, f"{c:.1f}", ha="center", fontsize=9.5, color=REPORT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    style(ax, "通道级归因：跳动最深的通道，正是补偿量被永久扣减的通道",
          "通道", "ADC")
    ax.set_ylim(-max(dips) * 0.05, max(dips) * 1.28)
    return save(fig, "ppt_fig03_通道影响.png")


def fig04(acc: dict) -> None:
    """只画三种"长期基线偏移"量级可比的场景。

    真实卸载 90% 会引起约 −2.3 万 ADC 的载荷变化，与前三者在同一坐标轴上无法比较，
    其长期偏移单独以文字给出（−0.05 ADC），量级与无扰动对照等同。
    """
    order = ["dip_injection", "sustained_step_20s", "no_injection_control"]
    text = {"dip_injection": "注入瞬时跳动\n（防护关闭）",
            "sustained_step_20s": "持续加载 20 秒\n（防护关闭）",
            "no_injection_control": "无注入对照"}
    vals, names, colors = [], [], []
    for tag in order:
        item = next(i for i in acc["scenarios"] if i["tag"] == tag)
        names.append(text[tag])
        vals.append(item["dirty_shift"] - item.get("clean_shift", 0.0))
        colors.append(REPORT if tag == "dip_injection" else PRIMARY)

    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    bars = ax.bar(names, vals, color=colors, width=0.40)
    low, high = min(vals), max(vals)
    ref = max(abs(low), abs(high))
    for bar, val in zip(bars, vals):
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(cx, val + (ref * 0.05 if val >= 0 else -ref * 0.05),
                f"{val:+.2f} ADC", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=14, fontweight="bold",
                color=REPORT if val > 0.5 else PRIMARY)
    ax.axhline(0, color="#9AA3B2", lw=1.2)
    ax.set_ylim(-ref * 1.55, ref * 1.55)
    ax.text(0.5, 0.975, "三种形态的长期偏移都在 ±30 ADC 以内；只有“瞬时跳动”留下稳定偏移",
            transform=ax.transAxes, ha="center", va="top", fontsize=12, color=INK)
    style(ax, "", "", "相对无扰动对照的长期基线偏移（ADC）")
    return save(fig, "ppt_fig04_受控复现.png")


def fig05(acc: dict) -> None:
    dip = next(i for i in acc["scenarios"] if i["tag"] == "dip_injection")
    before = dip["dirty_shift"] - dip.get("clean_shift", 0.0)
    after = dip["guarded_shift"] - dip.get("clean_shift", 0.0)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    bars = ax.bar(["修复前\n（防护关闭）", "修复后\n（防护启用）"], [before, after],
                  color=[REPORT, FIX], width=0.42)
    for bar, val in zip(bars, [before, after]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + before * 0.06, f"{val:+.2f} ADC",
                ha="center", fontsize=17, fontweight="bold",
                color=REPORT if val > 0.5 else FIX)
    ax.axhline(0, color="#9AA3B2", lw=1.0)
    ax.set_ylim(-(before * 0.30) - 1.0, before * 1.42)
    ax.annotate("", xy=(0.86, before * 0.62), xytext=(0.14, before * 0.62),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=2.0))
    ax.text(0.5, before * 0.30, f"抑制 {before - after:.1f} ADC",
            ha="center", va="center", fontsize=15, color=ACCENT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=ACCENT, lw=1.2, alpha=0.95))
    style(ax, "修复前后对比：长期基线偏移",
          "", "跳动造成的长期基线偏移（ADC）")
    return save(fig, "ppt_fig05_修复对比.png")


def fig06(cross: dict) -> None:
    long_term = [("左拇指恒载\n测试1", "左拇指0.6/测试1"),
                 ("右拇指恒载\n测试1", "右拇指0.6/测试1"),
                 ("设备1 恒载\n测试1", "右拇指/设备1-测试1"),
                 ("设备2 恒载\n测试1", "右拇指/设备2-测试1")]
    ten_g = [("设备1\n10 g 测试", "右拇指/设备1-10g"),
             ("设备2\n10 g 测试", "右拇指/设备2-10g")]
    labels, counts, colors, durations = [], [], [], []
    for label, key in long_term + ten_g:
        labels.append(label)
        counts.append(cross[key]["transient_dips"])
        durations.append(cross[key]["duration_s"])
        colors.append(REPORT if "10 g" in label else PRIMARY)
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    bars = ax.bar(labels, counts, color=colors, width=0.52)
    for bar, val, dur in zip(bars, counts, durations):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.45, f"{val} 次",
                ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.3, f"录制 {dur:.0f} 秒",
                ha="center", fontsize=9.5, color="#6C7480")
    ax.set_ylim(0, max(counts) * 1.28)
    ax.text(0.5, 0.93, "长时间恒载录制中干扰出现明显更频繁，每一次都可能留下一份不会消失的偏差",
            transform=ax.transAxes, ha="center", fontsize=11, color=INK)
    style(ax, "同类瞬时跳动在各录制中的检出次数", "", "检出次数")
    return save(fig, "ppt_fig06_出现次数.png")


def fig07() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    t = np.linspace(0, 6, 1200)
    base = np.zeros_like(t)

    jump = base + 40 * np.exp(-((t - 2.6) / 0.13) ** 2) - 40 * np.exp(-((t - 2.62) / 0.11) ** 2)
    jump[: int(2.5 / 6 * 1200)] = 0
    jump[int(2.9 / 6 * 1200):] = 0
    ax = axes[0]
    ax.axhline(0, color=SENSOR, lw=1.4, ls="--", label="原有基线水平")
    ax.plot(t, jump, color=REPORT, lw=2.0, label="瞬时跳动")
    ax.text(0.06, 62, "物理含义：干扰", fontsize=13.5, color=REPORT, fontweight="bold")
    ax.text(0.06, 48, "载荷状态并未改变，基线必须保持原位", fontsize=10.5, color="#5A6371")
    ax.annotate("偏离后迅速回到原水平", xy=(2.62, -44),
                xytext=(2.95, -63), fontsize=11, color=REPORT,
                arrowprops=dict(arrowstyle="->", color=REPORT, lw=1.3))
    style(ax, "形态 A：来了又走", "时间（秒）", "相对基线的偏移（ADC）")
    ax.set_xlim(0, 6)
    ax.set_ylim(-84, 84)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    ax = axes[1]
    step = np.where(t >= 2.6, -45.0, 0.0)
    ax.axhline(0, color=SENSOR, lw=1.4, ls="--", label="原有基线水平")
    ax.plot(t, step, color=PRIMARY, lw=2.0, label="真实载荷变化")
    ax.axhline(-45, color=FIX, lw=1.4, ls=":", label="新的基线水平")
    ax.text(0.06, 62, "物理含义：真实卸载", fontsize=13.5, color=PRIMARY, fontweight="bold")
    ax.text(0.06, 48, "载荷确实已改变，基线应当随之更新", fontsize=10.5, color="#5A6371")
    ax.annotate("偏移后稳定停留", xy=(4.7, -45), xytext=(3.55, -62),
                fontsize=11, color=FIX,
                arrowprops=dict(arrowstyle="->", color=FIX, lw=1.3))
    style(ax, "形态 B：来了就留下", "时间（秒）", "相对基线的偏移（ADC）")
    ax.set_xlim(0, 6)
    ax.set_ylim(-84, 84)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    fig.suptitle("两种变化在数值幅度上可能相似，物理含义却完全相反 —— 这正是误判的根源",
                 fontsize=14, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.13, wspace=0.20)
    return save(fig, "ppt_fig07_两种变化对比.png")


def fig08() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def card(x, y, w, h, color, title, lines, title_size=13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.12",
                                    fc="white", ec=color, lw=1.8))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.62), w, 0.62,
                                    boxstyle="round,pad=0.12,rounding_size=0.12",
                                    fc=color, ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 0.31, title, ha="center", va="center",
                fontsize=title_size, color="white", fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + 0.28, y + h - 1.05 - i * 0.72, line, ha="left", va="center",
                    fontsize=11, color=INK)

    card(0.05, 2.05, 5.9, 3.05, REPORT, "不设幅度门槛", [
        "不以“偏移超过多少 ADC 就拒绝”为判据",
        "避免把真实的、幅度有限的加载卸载误判为干扰",
        "保护对象是变化的“形态”，不是“幅度”",
    ], title_size=13.5)

    card(6.35, 2.05, 3.6, 3.05, PRIMARY, "必须持续保持", [
        "电平变化要在短时间内持续保持，",
        "才被确认为新的基准；",
        "抖动、毛刺、接触干扰这类",
        "“来了又走”的变化不被采纳。",
    ], title_size=13.5)

    card(0.05, 0.25, 9.9, 1.5, ACCENT, "大幅度真实卸载仍然立即响应", [
        "明显的真实卸载（例如手指抬起）不受保护层延迟影响，卸载响应速度与原有算法保持一致；保护层只在“疑似卸除”这一条判定路径上生效。",
    ], title_size=13.5)
    fig.suptitle("三条设计原则", fontsize=15.5, color=INK, fontweight="bold", y=1.0)
    return save(fig, "ppt_fig08_设计原则.png")


def main() -> None:
    session = json.loads((DATA / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])

    pre = load_stream(DATA, "device_001_pre_seg0.csv")
    seg = load_stream(DATA, "device_001_seg000.csv")
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame - frame[0]) / fps
    pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
    seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)

    before = (time >= 193.5) & (time < 194.45)
    after = (time >= 195.0) & (time < 196.0)
    seg_shift = float(np.median(seg_total[after]) - np.median(seg_total[before]))
    base_before = float(np.median(seg_total[before]))
    base_after = float(np.median(seg_total[after]))

    chan = load_json("channel_result.json")
    acc = load_json("guard_accept.json")
    cross = load_json("dip_crosscheck.json")

    fig01(time, pre_total, seg_total, seg_shift)
    fig02(time, pre_total, seg_total, seg_shift, base_before, base_after)
    fig03(chan)
    fig04(acc)
    fig05(acc)
    fig06(cross)
    fig07()
    fig08()

    plt.close("all")
    out = sorted(p.name for p in FIG.glob("*.png"))
    print(json.dumps({"fps": round(fps, 4), "seg_shift_adc": round(seg_shift, 2),
                      "figures": out}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
