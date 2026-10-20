"""慢漂到底跟载荷有关还是无关：决定它该由 x_slow（蠕变）还是 zero（零漂）吸收。

逐通道取 1100~2850 s 的漂移量与该通道载荷（相对空载首帧），算相关与比值分布。
  - 若漂移 ∝ 载荷（相关高、比值稳定）→ 是蠕变，x_slow 补它是对的，缺的是"饱和"；
  - 若漂移与载荷无关（相关低）→ 是零点漂移，应该由 zero 慢跟踪吸收，x_slow 在替它背锅。

输出：out/drift_correlation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")


def load_cells(name):
    path = ROOT / name
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


ts, pre = load_cells("device_001_pre_seg0.csv")
t = ts - ts[0]
m = (t >= 1100) & (t <= 2850)
tt = t[m]
seg = pre[m]
tc = tt - tt.mean()
slope = (tc[:, None] * (seg - seg.mean(axis=0))).sum(axis=0) / float((tc ** 2).sum()) * 60.0  # ADC/min
load = np.median(seg, axis=0) - pre[0]          # 相对首帧（空载）

ok = load > 50.0
print(f"参与统计的通道: {int(ok.sum())}/{len(load)}（载荷>50 ADC）")
rho = float(np.corrcoef(slope[ok], load[ok])[0, 1])
ratio = slope[ok] / load[ok] * 100.0            # 每 100 ADC 载荷的漂移率
report = dict(
    channels_used=int(ok.sum()),
    corr_slope_vs_load=round(rho, 3),
    slope_per_min_median=round(float(np.median(slope[ok])), 4),
    slope_per_min_std=round(float(slope[ok].std()), 4),
    load_median=round(float(np.median(load[ok])), 1),
    load_std=round(float(load[ok].std()), 1),
    ratio_per_100adc_median=round(float(np.median(ratio)), 4),
    ratio_p25=round(float(np.percentile(ratio, 25)), 4),
    ratio_p75=round(float(np.percentile(ratio, 75)), 4),
    ratio_cv=round(float(np.std(ratio) / abs(np.mean(ratio))), 3),
)
print(json.dumps(report, ensure_ascii=False, indent=2))
print("\n斜率(ADC/min) 分布：", np.round(np.percentile(slope[ok], [5, 25, 50, 75, 95]), 3))
print("载荷(ADC) 分布：   ", np.round(np.percentile(load[ok], [5, 25, 50, 75, 95]), 1))

(OUT / "drift_correlation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
print(f"\n-> {OUT / 'drift_correlation.json'}")
