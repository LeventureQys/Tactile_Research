"""K10 档（guard=1）叠加 `floor_rise_gate=1` 的后果实测。

三问：
  Q1 K10 的瞬态防护收益在 gate=1 后是否改变（gate 会不会削弱/加强 guard）？
  Q2 gate=1 是否让 y_floor 停在空载电平、把"部分卸载"判成受载（补偿不泄放）？
  Q3 真实完整卸载（回到空载）时 gate=1 能否恢复？

测试床沿用 02_K10 的构造（exp8_bed.py）：左拇指指腹 r_slow_max=0.6 测试1，
重放到 1060 s（y_floor 已接近工作电平），在 1000 s 注入事件；观察窗 1010~1060 s。

四臂：ORIG(gate=0,guard=0) / GATE(1,0) / K10(0,1) / K10+GATE(1,1)
输出：out/k10_gate.json
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
RUN_S = 1060.0
EVENT_S = 1000.0
BEFORE = (EVENT_S - 30.0, EVENT_S - 1.0)
AFTER = (EVENT_S + 10.0, EVENT_S + 59.0)
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


def med(arr, t, span):
    m = (t >= span[0]) & (t < span[1])
    return float(np.median(arr[m]))


session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
fps = float(session["devices"][0]["observed_fps"])
pre = load_stream("device_001_pre_seg0.csv")
values_all = np.asarray(pre.iloc[:, 3:], dtype=float)
frame = np.asarray(pre.iloc[:, 2], dtype=float)
time_all = (frame - frame[0]) / fps
limit = int(np.searchsorted(time_all, RUN_S))
values = values_all[:limit].copy()
time = time_all[:limit]

normal = np.median(values_all[(time_all >= 194.2) & (time_all < 194.45)], axis=0)
band = (time_all >= 194.45) & (time_all < 194.75)
dip_shape = values_all[band] - normal
dip_frames = dip_shape.shape[0]
start = int(EVENT_S * fps)

scenarios = {}
scenarios["dip_inject"] = (values.copy() + 0.0, 0.0)

dip_in = values.copy()
dip_in[start:start + dip_frames] += dip_shape
scenarios["dip_inject"] = (dip_in, 0.0)
for step in (20.0, 50.0, 150.0, 400.0, 800.0):
    arr = values.copy()
    arr[start:] -= step
    scenarios[f"unload_{int(step)}"] = (arr, step)

print(f"fps={fps:.1f} frames={len(time)} ch={values.shape[1]} dip_frames={dip_frames} "
      f"dip_total_min={dip_shape.sum(1).min():.1f}")
print(f"pre total: t={EVENT_S:.0f}s -> {values[start].sum():.1f}; "
      f"卸载档位对应残余载荷 = 上值 − 52×step")

report = {"fps": fps, "frames": int(len(time)), "channels": int(values.shape[1]),
          "dip_frames": int(dip_frames), "dip_total_min": round(float(dip_shape.sum(1).min()), 2),
          "run_s": RUN_S, "event_s": EVENT_S, "before": BEFORE, "after": AFTER, "scenarios": {}}

for name, (data, step) in scenarios.items():
    res = {}
    pre_total = float(np.median(data[(time >= BEFORE[0]) & (time < BEFORE[1])].sum(axis=1)))
    post_total = float(np.median(data[(time >= AFTER[0]) & (time < AFTER[1])].sum(axis=1)))
    for arm, ov in ARMS.items():
        log = name in ("dip_inject", "unload_20")
        res_out = core.run(data, time, ov, log=log)
        if log:
            v, L, Lb, Ls = res_out
        else:
            v = res_out
        out = v.sum(axis=1)
        ap = data.sum(axis=1) - out
        row = dict(
            out_before=round(med(out, time, BEFORE), 2),
            out_after=round(med(out, time, AFTER), 2),
            shift=round(med(out, time, AFTER) - med(out, time, BEFORE), 2),
            applied_before=round(med(ap, time, BEFORE), 2),
            applied_after=round(med(ap, time, AFTER), 2),
            err_vs_input_after=round(med(out, time, AFTER) - post_total, 2),
        )
        if log:
            yf = L["y_floor"].sum(axis=1)
            row.update(
                yfloor_before=round(med(yf, time, BEFORE), 2),
                yfloor_after=round(med(yf, time, AFTER), 2),
                idle_after=int(L["idle"][(time >= AFTER[0]) & (time < AFTER[1])].sum()),
            )
            del L, Lb, Ls
        res[arm] = row
        del v
    res["input"] = dict(pre_total=round(pre_total, 2), post_total=round(post_total, 2))
    report["scenarios"][name] = res
    print(f"\n[{name}]  input {pre_total:.1f} -> {post_total:.1f}")
    for arm in ARMS:
        r = res[arm]
        extra = ""
        if "idle_after" in r:
            extra = (f" | yfloor {r['yfloor_before']:9.1f}->{r['yfloor_after']:9.1f}"
                     f" idle={r['idle_after']}")
        print(f"  {arm:9s} applied {r['applied_before']:8.1f}->{r['applied_after']:8.1f}"
              f" | out {r['out_after']:9.1f} err={r['err_vs_input_after']:+9.1f}{extra}")

OUT.mkdir(exist_ok=True)
(OUT / "k10_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n-> {OUT / 'k10_gate.json'}")
