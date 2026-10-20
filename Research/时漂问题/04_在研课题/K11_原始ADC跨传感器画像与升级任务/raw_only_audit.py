from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "data"
MANIFEST = json.loads((HERE.parent.parent / "07_原始ADC跨传感器画像_20261020" / "manifest.json").read_text(encoding="utf-8"))
OUT = HERE / "raw_only_results"
OUT.mkdir(exist_ok=True)


def read_raw(path: Path):
    lines = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if line.strip() == "##Data":
                break
    frame = pd.read_csv(path, skiprows=len(lines), encoding="utf-8-sig")
    values = frame.to_numpy(dtype=float)
    return values[:, 0], values[:, 1], values[:, 2], values[:, 3:], lines


def robust_slope(t, y):
    if len(t) < 5 or np.ptp(t) <= 0:
        return np.nan
    x = t - np.median(t)
    return float(np.sum(x * (y - np.median(y))) / np.sum(x * x) * 60.0)


def summarize(sid, group, rel):
    path = SOURCE / rel / "device_001_raw_seg000.csv"
    ts, elapsed, frame, raw, header = read_raw(path)
    t = ts - ts[0]
    total = raw.sum(axis=1)
    dt = np.diff(ts)
    edt = np.diff(elapsed)
    pos = dt[dt > 0]
    seconds = np.floor(np.maximum(t, 0)).astype(int)
    med = pd.Series(total).groupby(seconds).median()
    bt = med.index.to_numpy(float) + 0.5
    bv = med.to_numpy(float)
    def at(age):
        m = np.abs(bt - age) <= 0.5
        return float(np.median(bv[m])) if m.any() else np.nan
    q05, q50, q95 = np.percentile(total, [5, 50, 95])
    first = float(np.median(total[t < min(1.0, max(t[-1] / 2, 0.01))]))
    last = float(np.median(total[t >= max(t[-1] - 1.0, 0)]))
    q = raw.shape[1]
    # Equal-age values are descriptive only; no physical event is inferred.
    early = robust_slope(bt[(bt >= 0) & (bt <= min(30, t[-1]))], bv[(bt >= 0) & (bt <= min(30, t[-1]))])
    late_start = max(0.0, t[-1] - 30.0)
    late = robust_slope(bt[bt >= late_start], bv[bt >= late_start])
    active = np.maximum(np.median(raw[t >= max(t[-1]-1, 0)], axis=0) - np.median(raw[t < min(1, max(t[-1]/2, .01))], axis=0), 0)
    eff = float(active.sum() ** 2 / np.sum(active ** 2)) if np.sum(active ** 2) else np.nan
    return dict(id=sid, group=group, relative_path=rel, frames=len(t), channels=q, duration_s=float(t[-1]),
        first_total_adc=first, last_total_adc=last, end_minus_start_adc=last-first,
        total_q05_adc=float(q05), total_q50_adc=float(q50), total_q95_adc=float(q95),
        q95_q05_span_adc=float(q95-q05), first_1s_median_adc=at(.5),
        t30_total_adc=at(30), t60_total_adc=at(60), t120_total_adc=at(120),
        slope_first30_adc_min=early, slope_last30_adc_min=late,
        active_effective_channels=eff,
        timestamp_nonpositive=int(np.sum(dt <= 0)), timestamp_lt1ms_frac=float(np.mean(dt < .001)),
        timestamp_gt100ms=int(np.sum(dt > .1)), timestamp_max_gap_s=float(np.max(dt)),
        timestamp_positive_median_ms=float(np.median(pos)*1000) if len(pos) else np.nan,
        elapsed_nonpositive=int(np.sum(edt <= 0)), elapsed_repeat_frac=float(np.mean(edt == 0)),
        frame_missing=int(np.sum(np.maximum(np.diff(frame)-1, 0))),
        raw_min_adc=float(raw.min()), raw_max_adc=float(raw.max()))


rows = [summarize(*item) for item in MANIFEST["sessions"]]
df = pd.DataFrame(rows)
df.to_csv(OUT / "raw_only_audit.csv", index=False, encoding="utf-8-sig")

lines = [
    "# K11 原始 ADC 只读审计（raw-only）",
    "",
    "分析范围：manifest 固定的 21 份 `device_001_raw_seg000.csv`；仅读取数据阶段明确为 `raw_adc` 的文件。没有读取 pre、seg、algoadc、mapforce、forcetotal 作为数值证据。",
    "",
    "## 可复核事实",
    f"- 会话数 {len(df)}，总帧数 {int(df.frames.sum()):,}，总时长 {df.duration_s.sum():.1f} s；阵列通道数为 {sorted(df.channels.unique().tolist())}。",
    f"- timestamp 非正间隔合计 {int(df.timestamp_nonpositive.sum())}；最大间隔 {df.timestamp_max_gap_s.max():.3f} s；小于 1 ms 的间隔比例为 {df.timestamp_lt1ms_frac.min():.1%}–{df.timestamp_lt1ms_frac.max():.1%}。因此斜率只按真实 timestamp 的 1 s 中值序列估计，不能把原始帧当作等间隔采样。",
    f"- elapsed 重复比例为 {df.elapsed_repeat_frac.min():.1%}–{df.elapsed_repeat_frac.max():.1%}，不能单独把 elapsed 当作连续时间轴。",
    "",
    "## 观察到的形态",
    "- 同一阵列内部的会话总 ADC 量级和慢态形态差异很大。57 通道右拇指长记录 T11/T12/T21/T22 的首尾差分别为 " + ", ".join(f"{r.id} {r.end_minus_start_adc:.0f} ADC" for r in df[df.id.isin(['T11','T12','T21','T22'])].itertuples()) + "；这不是同一套固定慢态参数可以自然覆盖的证据。",
    "- 同部位重复记录并不等价：T11/T12 的 30–60 s 后仍有明显正斜率，T21/T22 的幅度和后段速度不同；NP1/NP2（71 通道）在后段分别保持正向变化与近平台。",
    "- 跨阵列比较必须归一化。原始总和受 52/57/71 通道数、空间激活数量和布局影响；本审计同时输出 `active_effective_channels`，不把不同通道数的总 ADC 直接视为同一物理量。",
    "- 部分会话首秒已接近高电平，部分会话首秒接近低电平；首秒不能被统一命名为空载基线。",
    "- 逐通道变化存在集中与分散两种形态；因此单一逐通道绝对阈值无法覆盖“总量小但多通道分散”的变化。",
    "",
    "## 对 K11 的直接判断",
    "1. 保留统一的因果结构，但按部位/设备/阵列建立参数档；至少分开幅度比例、慢态速率上限、确认窗、恢复窗和噪声阈值。",
    "2. 时间处理应以 timestamp 为主，显式标记非正间隔、重复/打包间隔和长间隔；异常间隔不得直接参与普通斜率估计。",
    "3. 参考基线必须来自有证据的稳定空载或平台窗口；当前 raw-only 文件没有同步操作标签，不能把自动候选高段验收为恒载，也不能把名义 10g 文件名当作逐次事件真值。",
    "4. 结构层与传感器层分开验证：先验证时间异常、迟滞、零点受载保护、部分卸载和状态重置，再用本审计的部位差异去标定参数。",
    "",
    "## 证据边界",
    "本结果是原始信号形态画像，不是 8.9N 恒载或 10g 事件验收。要完成需求文档中的联合验收，必须补齐操作时间戳、荷载方向/大小/保持时长、真实 ADC-力映射和设备序列号；否则只能报告相对变化与候选窗口。",
    "",
    "详细逐会话结果见 `raw_only_results/raw_only_audit.csv`。"
]
(HERE / "RAW_ONLY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(df[['id','channels','duration_s','first_total_adc','last_total_adc','end_minus_start_adc','slope_first30_adc_min','slope_last30_adc_min','timestamp_nonpositive','timestamp_max_gap_s']].round(3).to_string(index=False))
print(f"WROTE {OUT / 'raw_only_audit.csv'}")
print(f"WROTE {HERE / 'RAW_ONLY_ANALYSIS.md'}")
