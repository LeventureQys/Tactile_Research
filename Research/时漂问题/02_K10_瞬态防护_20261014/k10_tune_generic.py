# -*- coding: utf-8 -*-
"""K10 调参通用扫描器：任意「单流 ADC 长保压」会话。

用法：
    python k10_tune_generic.py --data-root <目录> --tag <标签>

指标（口径对齐 161747 报告与 tune_right_thumb.py）：
    onset    = 输入总值首个 > 500 ADC 帧间跳变的时刻（加载沿）
    E        = 输入在 [onset+0.35, onset+1.0] 的最小值（弹性电平）
    rise     = 输出(末 10 s 中值) − 输出(10~20 s 中值)      ← 读数还涨多少（用户体感）
    landing  = 输出(末 10 s 中值) − E                        ← 准度（落点）
    tail     = 输出末 120 s 拟合斜率                          ← 是否已冻结
    offset30 = 自 t=30 s 起 10 s 窗口中值相对参考值的最大绝对偏移
    ratio    = 输出末 / 输入末
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from guarded_k9 import GUARD_DEFAULTS, compensate  # noqa: E402

GUARD_ON = dict(GUARD_DEFAULTS, enabled=True)

# 数据预设写死在此，避免命令行传递非 ASCII 路径
PRESETS = {
    # 本次任务的数据：右手掌-2026-10-15_15-22-01（670 s 恒载）
    # 该目录已被移入回收站（2026-10-15 15:43），此处直接只读原件，不做还原
    "palm_152201": Path(r"D:\$RECYCLE.BIN\S-1-5-21-3129820652-4156655561-3743858824-1001\$RKJ92E8"),
    # 交叉验证：手掌 b2d896 单台阶 429 s 长保压（ADC 显示、算法关、单流）
    "b2d896": Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\archived\手掌数据"
                   r"\20260927_161747_single_device_b2d896"),
}

# 现场可改写 8 参的既有档位（其余 19 项取工作树默认）
WORKSHEET = {}
FIRST_ARTICLE_PALM_R = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
                            slope_cap_frac=0.09, r_slow_max=0.2, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)
LIVE_CAP009 = dict(r_fast=0.02, tau_c_fast_s=1.0, slow_confirm_s=1.5, soft_unfreeze_s=1.5,
                   slope_cap_frac=0.009, r_slow_max=0.2, tau_r_fast_s=2.0, tau_r_slow_idle_s=0.5)
P161747_REC = dict(r_fast=0.06, tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=1.5,
                   slope_cap_frac=0.005, r_slow_max=0.15, tau_r_fast_s=6.0, tau_r_slow_idle_s=0.5)
P161747_MIN = dict(slope_cap_frac=0.005, r_slow_max=0.15)
FORCE_B = dict(r_fast=0.05, tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=2.0,
               slope_cap_frac=0.009, r_slow_max=0.35, tau_r_fast_s=6.0, tau_r_slow_idle_s=0.5)
PLAN_V4 = dict(r_fast=0.03, tau_c_fast_s=2.0, tau_r_fast_s=0.1, r_slow_max=0.1, slope_cap_frac=0.02,
               soft_unfreeze_s=2.0, slow_confirm_s=3.0, tau_r_slow_idle_s=0.2)


def load_stream(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def med_in(t, sig, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else None


def metrics(t: np.ndarray, total: np.ndarray, onset: float, e_level: float, analysis_start: float = 30.0) -> dict:
    early_lo = max(onset + 8.0, 10.0)
    early = med_in(t, total, early_lo, early_lo + 10.0)
    end = med_in(t, total, t[-1] - 10.0, t[-1])
    grid = np.arange(t[-1] - 120.0, t[-1] - 10.0, 10.0)
    late = [(g, med_in(t, total, g, g + 10.0)) for g in grid]
    late = [(q, v) for q, v in late if v is not None]
    slope = float(np.polyfit([q for q, _ in late], [v for _, v in late], 1)[0]) if len(late) > 1 else None
    ref = med_in(t, total, analysis_start, analysis_start + 10.0)
    wins = []
    a = analysis_start
    while a + 10.0 <= t[-1]:
        v = med_in(t, total, a, a + 10.0)
        if v is not None:
            wins.append(v)
        a += 10.0
    off = float(np.max(np.abs(np.asarray(wins) - ref))) if wins else None
    return dict(early_adc=round(early, 1), end_adc=round(end, 1), rise_adc=round(end - early, 1),
                landing_adc=round(end - e_level, 1),
                tail_slope_adc_s=round(slope, 4) if slope is not None else None,
                offset30_adc=round(off, 1) if off is not None else None)


def candidates() -> list[dict]:
    rows = [
        dict(tag="X0_worksheet_default", params=WORKSHEET),
        dict(tag="X1_palm_first_article", params=FIRST_ARTICLE_PALM_R),
        dict(tag="X2_live_cap009", params=LIVE_CAP009),
        dict(tag="X3_161747_recommended", params=P161747_REC),
        dict(tag="X4_161747_minimal", params=P161747_MIN),
        dict(tag="X5_force_B_crossmode", params=FORCE_B),
        dict(tag="X6_plan_v4_k9", params=PLAN_V4),
    ]
    for cap in (0.02, 0.03, 0.05):
        for rsm in (0.25, 0.35):
            for rf in (0.02, 0.05):
                rows.append(dict(tag=f"Y_cap{cap:.2f}_rsm{rsm:.2f}_rf{rf:.2f}",
                                 params=dict(r_fast=rf, slope_cap_frac=cap, r_slow_max=rsm,
                                             tau_c_fast_s=2.0, slow_confirm_s=2.0, soft_unfreeze_s=1.5,
                                             tau_r_fast_s=6.0, tau_r_slow_idle_s=0.5)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="palm_152201")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--main", default="device_001_seg000.csv")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    folder = args.data_root or PRESETS[args.preset]
    tag = args.tag or args.preset
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    dev = meta["devices"][0]
    fps = float(dev["observed_fps"])
    frame = load_stream(folder, args.main)
    x = np.asarray(frame.iloc[:, 3:], dtype=float)
    t = np.arange(x.shape[0], dtype=float) / fps
    input_total = x.sum(1)

    diff = np.diff(input_total)
    onset = float(t[int(np.argmax(diff > 500.0))])
    e_level = float(np.min(input_total[(t >= onset + 0.35) & (t <= onset + 1.0)]))
    base = metrics(t, input_total, onset, e_level)

    print(f"[{args.tag}] {folder.name}: {x.shape[0]} 帧 × {x.shape[1]} 通道, {t[-1]:.1f} s, fps {fps:.3f}",
          flush=True)
    print(f"[{args.tag}] 加载沿 {onset:.2f} s, E = {e_level:.1f} ADC | 输入: rise {base['rise_adc']:.0f}, "
          f"landing {base['landing_adc']:.0f}, tail {base['tail_slope_adc_s']:+.3f}, offset30 {base['offset30_adc']:.0f}",
          flush=True)
    header = f"{'tag':28s} {'rise':>8s} {'landing':>9s} {'tail':>8s} {'offset30':>9s} {'ratio':>7s}"
    print(header, flush=True)

    rows = [dict(tag="INPUT_reference", params="input", **base, compression_ratio=1.0)]
    for candidate in candidates():
        started = time.perf_counter()
        out, info = compensate(x.copy(), t, candidate["params"], GUARD_ON)
        m = metrics(t, out.sum(1), onset, e_level)
        row = dict(tag=candidate["tag"], params=json.dumps(candidate["params"], ensure_ascii=False),
                   **m, compression_ratio=round(m["end_adc"] / base["end_adc"], 4),
                   guard_frames=info.get("guard_frames"), seconds=round(time.perf_counter() - started, 2))
        rows.append(row)
        print(f"{row['tag']:28s} {row['rise_adc']:8.1f} {row['landing_adc']:9.1f} "
              f"{row['tail_slope_adc_s']:+8.3f} {row['offset30_adc']:9.1f} {row['compression_ratio']:7.4f}",
              flush=True)

    out_dir = args.out_dir or (HERE / "palm_tune_20261015")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"generic_{args.tag}.json").write_text(
        json.dumps(dict(source=str(folder), frames=int(x.shape[0]), channels=int(x.shape[1]),
                        fps=fps, onset_s=onset, E_adc=e_level, rows=rows), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"输出: {out_dir / f'generic_{args.tag}.json'}")


if __name__ == "__main__":
    main()
