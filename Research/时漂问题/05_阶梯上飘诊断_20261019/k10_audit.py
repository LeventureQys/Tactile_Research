"""K10 档（guard=1）在抬升门修复之后的独立审计。

重点回答三件事：
  1. gate=1 且 guard=0 时，真实下探会不会把 y_floor 永久钉在谷底（抬升被门控后爬不回来）；
  2. K10 档（guard=1）能否消除该残留；
  3. K10 档的三个 guard 参数在修复后是否仍是最优区间。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"


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
mt = (t >= 1100) & (t <= 2850)


def audit(tag, over):
    v, L, Lb, Ls = core.run(pre, ts, over, log=True)
    s = v.sum(1)
    ap = (pre.sum(1) - s)
    yfl = L["y_floor"]
    ym = L["y_max"]
    # y_floor 相对 y0 的低位程度：末段 (y0 - y_floor) 与 y_max 的比
    row = dict(tag=tag,
               drift=float(s[mt][-1] - s[mt][0]),
               p2p=float(s[mt].max() - s[mt].min()),
               applied_end=float(ap[mt][-1]),
               applied_min=float(ap[mt].min()),
               yfl_end_sum=float(yfl[mt][-1].sum()),
               yfl_min_sum=float(yfl[mt].min(0).sum()),
               span_end=float((ym - yfl)[mt][-1].mean()),
               idle_ratio=float((L["idle"] > 0.5).mean()))
    # y_floor 的"最后一次下跌"之后有没有回抬
    last_drop = []
    for k in range(pre.shape[1]):
        y = yfl[mt, k]
        dy = np.diff(y)
        idx = np.where(dy < -2.0)[0]
        if len(idx) == 0:
            continue
        i = idx[-1]
        last_drop.append(float(y[i + 1] - y[-1]))   # 负=之后继续跌/不回抬
    row["median_post_drop_move"] = float(np.median(last_drop)) if last_drop else 0.0
    return row


print(f"{'组合':38s} {'漂移':>9s} {'applied末':>10s} {'yfl末':>10s} {'yfl最小':>10s} {'span末':>8s} {'idle%':>7s} {'跌后回抬':>9s}")
rows = []
for tag, over in (
    ("gate=1 guard=0（v3.4 档修复态）", {"floor_rise_gate": 1}),
    ("gate=1 guard=1（K10 档修复态）", {"floor_rise_gate": 1, "guard_enable": 1}),
    ("gate=0 guard=0（原始 K9）", {}),
):
    r = audit(tag, over)
    rows.append(r)
    print(f"{tag:38s} {r['drift']:+9.1f} {r['applied_end']:10.1f} {r['yfl_end_sum']:10.1f} "
          f"{r['yfl_min_sum']:10.1f} {r['span_end']:8.1f} {r['idle_ratio']*100:6.2f}% "
          f"{r['median_post_drop_move']:+9.1f}")

print("\n=== K10 档 guard 参数扫描（全部 gate=1）===")
print(f"{'confirm':>8s} {'idle_conf':>10s} {'deep':>7s} {'漂移':>9s} {'applied末':>10s} {'yfl末':>10s} {'span末':>8s}")
best = None
for confirm in (0.0, 0.2, 0.5, 1.0, 2.0):
    for idle_conf in (0.1, 0.3, 0.6):
        for deep in (50.0, 200.0, 1e9):
            over = {"floor_rise_gate": 1, "guard_enable": 1,
                    "guard_floor_confirm_s": confirm, "guard_idle_confirm_s": idle_conf,
                    "guard_floor_deep": deep}
            v = core.run(pre, ts, over)
            s = v.sum(1)
            ap = (pre.sum(1) - s)
            d = float(s[mt][-1] - s[mt][0])
            print(f"{confirm:8.1f} {idle_conf:10.1f} {deep:7.0f} {d:+9.1f} "
                  f"{ap[mt][-1]:10.1f}")
            if best is None or abs(d) < abs(best[0]):
                best = (d, confirm, idle_conf, deep)
print(f"\n最小 |漂移| 组合: 漂移={best[0]:+.1f}  confirm={best[1]} idle_conf={best[2]} deep={best[3]}")
(OUT / "k10_audit_20261019.json").write_text(json.dumps(
    {"rows": rows, "sweep_best": {"drift": best[0], "confirm": best[1],
                                  "idle_confirm": best[2], "deep": best[3]}},
    ensure_ascii=False, indent=2), encoding="utf-8")
