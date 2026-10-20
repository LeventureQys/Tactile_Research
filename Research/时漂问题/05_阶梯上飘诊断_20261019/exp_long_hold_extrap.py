"""长恒载可行性查验（二）：把 48 min 实录外推到 2 小时，看 gate=1 是否失稳。

方法：取 1100~2850 s 恒载段，逐通道去线性趋势得到"纯噪声+形态"块，块内循环拼接，
再按各通道实测斜率把时间轴线性外推（漂移持续、噪声不重复放大）。这样得到一份
"同一物理状态、漂移持续 2 小时"的合成录制。

关注：输出漂移是否随时间线性恶化（负漂=过补，正漂=欠补），补偿覆盖率是否继续上升。

输出：out/long_hold_extrap.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import core  # noqa: E402

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
HOURS = 2.0


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
m = (t >= 1100) & (t < 2850)
tt = t[m]
seg = pre[m]
dt_nom = float(np.median(np.diff(tt)))

tc = tt - tt.mean()
den = float((tc ** 2).sum())
slope = (tc[:, None] * (seg - seg.mean(axis=0))).sum(axis=0) / den      # ADC/s per channel
detrend = seg - tc[:, None] * slope

block_s = float(tt[-1] - tt[0])
n_rep = int(HOURS * 3600.0 / block_s) + 1
block = np.tile(detrend, (n_rep, 1))
tt0 = tt - tt[0]
ts_rel = np.concatenate([tt0 + k * block_s for k in range(n_rep)])
n_frames = block.shape[0]
hold = block + ts_rel[:, None] * slope

# 前置空载引导段：让 K7 总值旁路在空载电平建立基线，否则首帧即恒载会让旁路恒真（直通）
lead_m = t < 60.0
lead = pre[lead_m]
n_lead = int(lead_m.sum())
values = np.vstack([lead, hold])
ts_long = np.concatenate([ts[:n_lead], ts[0] + 60.0 + ts_rel])

print(f"block={block_s:.1f}s reps={n_rep} frames={len(ts_long)} "
      f"span={ts_long[-1]-ts_long[0]:.0f}s input_drift={values[-1].sum() - lead[-1].sum():.0f} ADC")

report = {"hours": HOURS, "frames": int(len(ts_long)), "block_s": block_s, "reps": n_rep,
          "input_drift_total": round(float(values[-1].sum() - lead[-1].sum()), 1),
          "windows": {}}

for arm, ov in (("GATE r_slow_max=0.6", {"floor_rise_gate": 1}),
                ("GATE r_slow_max=0.35", {"floor_rise_gate": 1, "r_slow_max": 0.35}),
                ("GATE r_slow_max=0.25", {"floor_rise_gate": 1, "r_slow_max": 0.25})):
    v, L, Lb, Ls = core.run(values, ts_long, ov, log=True)
    out = v.sum(axis=1)
    inp = values.sum(axis=1)
    ap = inp - out
    y = values - L["zero"]
    cover = (values - v) / np.maximum(y, 1.0)
    rows = []
    for lo in range(0, int(HOURS * 3600), 1800):
        mm = (ts_long - ts_long[0] >= lo) & (ts_long - ts_long[0] < lo + 1800)
        if mm.sum() == 0:
            continue
        rows.append(dict(
            window=f"{lo//60}-{(lo+1800)//60}min",
            input_med=round(float(np.median(inp[mm])), 1),
            out_med=round(float(np.median(out[mm])), 1),
            applied_med=round(float(np.median(ap[mm])), 1),
            cover_med=round(float(np.median(cover[mm])), 3),
            drift_vs_first=round(float(np.median(out[mm]) - np.median(out[ts_long - ts_long[0] < 1800])), 1),
        ))
    report["windows"][arm] = rows
    print(f"\n=== {arm}（2 h 合成恒载）===")
    print(f"{'窗口':>10s} {'输入':>10s} {'输出':>10s} {'applied':>9s} {'覆盖率':>7s} {'相对首半小时':>12s}")
    for r in rows:
        print(f"{r['window']:>10s} {r['input_med']:10.1f} {r['out_med']:10.1f} "
              f"{r['applied_med']:9.1f} {r['cover_med']:7.3f} {r['drift_vs_first']:+12.1f}")
    del v, L, Lb, Ls

(OUT / "long_hold_extrap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> {OUT / 'long_hold_extrap.json'}")
