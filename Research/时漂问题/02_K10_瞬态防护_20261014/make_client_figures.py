# -*- coding: utf-8 -*-
"""生成面向甲方的说明文档配图：跳变导致基线上飘的现象、归因与修复验证。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures_甲方"
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
PREV = HERE.parent / "02_K10_瞬态防护_20261014"
SENSOR_LEVEL = 300.0        # ADC 域 / 力值域换算参考（仅用于量级提示）
ADC_PER_N = 300.0           # 需求文档口径：1000 ADC ≈ 1 N 为粗略折算，此处按现场标定另注

BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f855a"
GREY = "#718096"


def load_stream(folder: Path, name: str, skip: int = 30, nrows: int | None = None) -> pd.DataFrame:
    with (folder / name).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
    return pd.read_csv(folder / name, skiprows=skip, header=0, nrows=nrows, engine="c")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    session = json.loads((DATA / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])

    pre = load_stream(DATA, "device_001_pre_seg0.csv")
    seg = load_stream(DATA, "device_001_seg000.csv")
    frame = np.asarray(pre.iloc[:, 2], dtype=float)
    time = (frame - frame[0]) / fps
    pre_total = np.asarray(pre.iloc[:, 3:], dtype=float).sum(axis=1)
    seg_total = np.asarray(seg.iloc[:, 3:], dtype=float).sum(axis=1)

    # 用事件紧邻的前后窗，避免把蠕变自然上升混入"跳动造成的跳跃"
    before = (time >= 193.5) & (time < 194.45)
    after = (time >= 195.0) & (time < 196.0)
    pre_shift = float(np.median(pre_total[after]) - np.median(pre_total[before]))
    seg_shift = float(np.median(seg_total[after]) - np.median(seg_total[before]))
    seg_base_before = float(np.median(seg_total[before]))
    seg_base_after = float(np.median(seg_total[after]))

    # ── 图 1：现象总览 ──
    mask = (time >= 150) & (time <= 260)
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    ax.plot(time[mask], pre_total[mask], color=GREY, lw=1.0, label="传感器原始读数（算法输入）")
    ax.plot(time[mask], seg_total[mask], color=BLUE, lw=1.4, label="补偿后输出")
    ax.axvline(194.5, color=RED, ls="--", lw=1.2)
    ymin_visible = float(seg_total[mask].min())
    ymax_visible = float(pre_total[mask].max())
    ax.annotate("一次瞬时跳动\n（约 0.3 秒）",
                xy=(194.6, float(seg_total[np.argmin(np.abs(time - 194.6))])),
                xytext=(155.0, ymin_visible + 20), color=RED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.annotate(f"跳动之后基线整体上抬 {seg_shift:+.1f} ADC",
                xy=(234.0, seg_base_after), xytext=(207.0, ymin_visible + 150),
                color=RED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.annotate("原始读数同期无明显变化",
                xy=(222.0, float(np.median(pre_total[after]))),
                xytext=(176.0, ymax_visible - 130), color=GREY, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("阵列读数总和（ADC）")
    ax.set_title("现象总览：一次瞬时跳动让补偿后的基线永久抬高")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_现象总览.png")
    plt.close(fig)

    # ── 图 2：局部放大 ──
    m2 = (time >= 193.0) & (time <= 197.5)
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    ax.plot(time[m2], pre_total[m2], color=GREY, lw=1.0, label="传感器原始读数（算法输入）")
    ax.plot(time[m2], seg_total[m2], color=BLUE, lw=1.4, label="补偿后输出")
    ax.hlines(seg_base_before, 193.0, 194.5, color=GREEN, ls=":", lw=1.6)
    ax.hlines(seg_base_after, 194.9, 197.5, color=RED, ls=":", lw=1.6)
    ax.annotate("跳动前基线", xy=(193.2, seg_base_before),
                xytext=(193.02, seg_base_before - 105),
                color=GREEN, fontsize=9)
    ax.annotate(f"跳动后基线（上抬 {seg_shift:+.1f} ADC）", xy=(195.6, seg_base_after),
                xytext=(195.55, seg_base_after + 60),
                color=RED, fontsize=9)
    ax.axvspan(194.45, 194.78, color=RED, alpha=0.10)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("阵列读数总和（ADC）")
    ax.set_title("局部放大：跳动约持续 0.3 秒，其后基线未回落")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_局部放大.png")
    plt.close(fig)

    # ── 图 3：通道级影响（从已有归因结果读取） ──
    chan_path = HERE.parent / "02_K10_瞬态防护_20261014" / "channel_result.json"
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=160)
    if chan_path.exists():
        chan = json.loads(chan_path.read_text(encoding="utf-8"))
        rows = sorted(chan["channels"], key=lambda r: r["dip_drop"])[:8]
        labels = [f"通道 {r['ch']}" for r in rows]
        dips = [abs(r["dip_drop"]) for r in rows]
        changes = [abs(r["applied_change"]) for r in rows]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, dips, 0.4, color=BLUE, label="该通道瞬时跳动幅度")
        ax.bar(x + 0.2, changes, 0.4, color=RED, label="该通道补偿量永久减少")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ADC")
        ax.set_title("通道级归因：跳动最深的通道，正是补偿量被永久扣减的通道")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_通道影响.png")
    plt.close(fig)

    # ── 图 4：修复验证（受控复现实验） ──
    bed_path = HERE.parent / "02_K10_瞬态防护_20261014" / "guard_accept.json"
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=160)
    if bed_path.exists():
        acc = json.loads(bed_path.read_text(encoding="utf-8"))
        names, values, colors = [], [], []
        for item in acc["scenarios"]:
            label = {"dip_injection": "注入跳动\n（防护关闭）",
                     "sustained_step_20s": "持续加载\n（防护关闭）",
                     "real_unload_90pct": "真实卸载\n（防护关闭）",
                     "no_injection_control": "无注入对照"}.get(item["tag"], item["tag"])
            names.append(label)
            values.append(item["dirty_shift"] - item.get("clean_shift", 0.0))
            colors.append(RED if item["tag"] == "dip_injection" else BLUE)
        ax.bar(names, values, color=colors, width=0.55)
        for i, v in enumerate(values):
            ax.text(i, v + (1.2 if v >= 0 else -2.6), f"{v:+.2f}", ha="center", fontsize=9)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_ylabel("相对无扰动对照的长期偏移（ADC）")
        ax.set_title("受控复现：只有瞬时跳动会留下永久偏移（约 +24 ADC），持续加载与真实卸载不会")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_受控复现.png")
    plt.close(fig)

    # ── 图 5：修复前后对比 ──
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=160)
    if bed_path.exists():
        acc = json.loads(bed_path.read_text(encoding="utf-8"))
        dip = next(i for i in acc["scenarios"] if i["tag"] == "dip_injection")
        before_val = dip["dirty_shift"] - dip.get("clean_shift", 0.0)
        after_val = dip["guarded_shift"] - dip.get("clean_shift", 0.0)
        bars = ax.bar(["修复前", "修复后"], [before_val, after_val],
                      color=[RED, GREEN], width=0.45)
        for bar, val in zip(bars, [before_val, after_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + (1.0 if val >= 0 else -1.8),
                    f"{val:+.2f} ADC", ha="center", fontsize=11)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_ylabel("跳动造成的长期基线偏移（ADC）")
        ax.set_title("修复效果：同样强度的瞬时跳动，长期偏移由约 +24 ADC 降至接近 0")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_修复对比.png")
    plt.close(fig)

    # ── 图 6：同类跳动在不同录制中的出现次数 ──
    cross_path = HERE.parent / "02_K10_瞬态防护_20261014" / "dip_crosscheck.json"
    fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=160)
    if cross_path.exists():
        cross = json.loads(cross_path.read_text(encoding="utf-8"))
        order = ["左拇指0.6/测试1", "右拇指0.6/测试1",
                 "右拇指/设备1-测试1", "右拇指/设备2-测试1",
                 "右拇指/设备1-10g", "右拇指/设备2-10g"]
        labels, counts, colors = [], [], []
        for key in order:
            if key not in cross or "transient_dips" not in cross[key]:
                continue
            labels.append(key.replace("/", "\n"))
            counts.append(cross[key]["transient_dips"])
            colors.append(GREEN if "10g" in key else BLUE)
        ax.bar(labels, counts, color=colors, width=0.55)
        for i, v in enumerate(counts):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
        ax.set_ylabel("检出次数")
        ax.set_title("同类瞬时跳动在各录制中的出现次数（长时间恒载录制明显更多）")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_出现次数.png")
    plt.close(fig)

    summary = dict(fps=round(fps, 4), frames=int(time.size),
                   pre_shift_adc=round(pre_shift, 1), seg_shift_adc=round(seg_shift, 1),
                   figures=sorted(p.name for p in FIG.glob("*.png")))
    (HERE / "figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
