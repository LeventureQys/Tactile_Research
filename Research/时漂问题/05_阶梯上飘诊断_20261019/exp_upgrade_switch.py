"""运行中切档：已运行一段后把 floor_rise_gate 打开/关闭，状态不重置。

回答两个部署问题：
  UP   : 设备已带旧基线运行 1500 s（y_floor 已被抬升吸走载荷），此时下发 gate=1 —— 能否止血？
  DOWN : 设备已带 gate=1 运行 1200 s，此时回滚 gate=0 —— 多久回到旧行为？

数据：2026-10-19 拇指指腹 48 min 恒载实录（287374 帧 × 57 通道，r_slow_max=0.6）。
输出：out/upgrade_switch.json
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
LO, HI = 1100.0, 2850.0


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


def summarize(out, pre_total, t):
    m = (t >= LO) & (t <= HI)
    sm = out[m]
    apm = (pre_total - out)[m]
    tt = t[m]
    sec = np.floor(tt).astype(np.int64)
    s0 = sec - sec.min()
    cnt = np.bincount(s0, minlength=s0[-1] + 1)
    aps = np.bincount(s0, weights=apm, minlength=cnt.size) / np.maximum(cnt, 1)
    return dict(drift=round(float(sm[-1] - sm[0]), 1),
                p2p=round(float(sm.max() - sm.min()), 1),
                applied_start=round(float(apm[0]), 1),
                applied_end=round(float(apm[-1]), 1),
                applied_min=round(float(apm.min()), 1),
                steps=int((np.diff(aps) < -30).sum()))


ts, pre = load_cells("device_001_pre_seg0.csv")
t = ts - ts[0]
pre_total = pre.sum(axis=1)
print(f"frames={len(t)} ch={pre.shape[1]} span={t[-1]:.1f}s")

CASES = {
    "K9(全程 gate=0)": dict(p={}, schedule=None),
    "GATE(全程 gate=1)": dict(p={"floor_rise_gate": 1}, schedule=None),
    "UP@1500(gate 0->1)": dict(p={}, schedule=[(ts[0] + 1500.0, {"floor_rise_gate": 1})]),
    "DOWN@1200(1->0)": dict(p={"floor_rise_gate": 1}, schedule=[(ts[0] + 1200.0, {"floor_rise_gate": 0})]),
}

res = {}
for name, cfg in CASES.items():
    v = core.run(pre, ts, cfg["p"], log=False, schedule=cfg["schedule"])
    res[name] = summarize(v.sum(axis=1), pre_total, t)
    print(f"\n{name}")
    print("  " + json.dumps(res[name], ensure_ascii=False))

(OUT / "upgrade_switch.json").write_text(
    json.dumps({"window": [LO, HI], "cases": res}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> {OUT / 'upgrade_switch.json'}")
