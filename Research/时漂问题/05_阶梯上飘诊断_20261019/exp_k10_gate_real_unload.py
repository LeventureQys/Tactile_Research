"""左拇指 0.6 测试1 的**真实卸载尾段**（1060 s 之后录制本身回到空载）四臂对照。

目的：区分"人工部分卸载"与"真实完整卸载"下 gate=1 的恢复能力。
输出：out/k10_gate_real_unload.json
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

FOLDER = Path(
    r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\archived"
    r"\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40"
)
ARMS = {
    "ORIG": {},
    "GATE": {"floor_rise_gate": 1},
    "K10": {"guard_enable": 1},
    "K10+GATE": {"guard_enable": 1, "floor_rise_gate": 1},
}


def load_stream(name):
    path = FOLDER / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
fps = float(session["devices"][0]["observed_fps"])
pre = load_stream("device_001_pre_seg0.csv")
values = np.asarray(pre.iloc[:, 3:], dtype=float)
frame = np.asarray(pre.iloc[:, 2], dtype=float)
time = (frame - frame[0]) / fps
tin = values.sum(axis=1)
print(f"frames={len(time)} span={time[-1]:.1f}s")
for lo, hi in ((1000, 1059), (1060, 1099), (1100, 1149), (1150, 1191)):
    m = (time >= lo) & (time < hi)
    print(f"  input {lo}-{hi}s median={np.median(tin[m]):9.1f} min={tin[m].min():9.1f}")

rows = {}
for arm, ov in ARMS.items():
    v, L, Lb, Ls = core.run(values, time, ov, log=True)
    out = v.sum(axis=1)
    ap = tin - out
    row = {"applied_min": round(float(ap.min()), 2)}
    for lo, hi in ((1000, 1059), (1060, 1099), (1100, 1149), (1150, 1191)):
        m = (time >= lo) & (time < hi)
        row[f"out_{lo}_{hi}"] = round(float(np.median(out[m])), 1)
        row[f"applied_{lo}_{hi}"] = round(float(np.median(ap[m])), 1)
        row[f"err_{lo}_{hi}"] = round(float(np.median(out[m]) - np.median(tin[m])), 1)
        row[f"idle_{lo}_{hi}"] = int(L["idle"][m].sum())
    row["yfloor_end"] = round(float(L["y_floor"].sum(axis=1)[-1]), 1)
    rows[arm] = row
    print(f"\n{arm}: applied min={row['applied_min']:.1f} yfloor_end={row['yfloor_end']:.1f}")
    for lo, hi in ((1060, 1099), (1100, 1149), (1150, 1191)):
        print(f"  {lo}-{hi}s out={row[f'out_{lo}_{hi}']:9.1f} applied={row[f'applied_{lo}_{hi}']:8.1f}"
              f" err={row[f'err_{lo}_{hi}']:+9.1f} idle={row[f'idle_{lo}_{hi}']}")
    del v, L, Lb, Ls

OUT.mkdir(exist_ok=True)
(OUT / "k10_gate_real_unload.json").write_text(
    json.dumps({"fps": fps, "frames": int(len(time)), "arms": rows}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print(f"\n-> {OUT / 'k10_gate_real_unload.json'}")
