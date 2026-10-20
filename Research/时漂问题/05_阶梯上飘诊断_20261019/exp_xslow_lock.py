"""核对：gate=1 下部分卸载后，x_slow 是否被"e<=0 时不夹上限"锁死。

cpp 里的慢态上限是：
    if (e > 0) { ... hi = r_slow_max * max(e,1); if (x_slow > hi) x_slow = hi; }
e<=0 时既不积分也不夹上限 ⇒ 补偿可以停在超过残余载荷的水平，形成过补偿自锁。
K9 靠 idle 快泄放兜住；gate=1 把 idle 门收紧后，这个自锁显形。
输出：out/xslow_lock.json
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
values_all = np.asarray(pre.iloc[:, 3:], dtype=float)
frame = np.asarray(pre.iloc[:, 2], dtype=float)
time_all = (frame - frame[0]) / fps
limit = int(np.searchsorted(time_all, 1060.0))
values = values_all[:limit].copy()
time = time_all[:limit]
start = int(1000.0 * fps)
rsm = core.BASE["r_slow_max"]

for step in (150.0, 400.0, 800.0):
    data = values.copy()
    data[start:] -= step
    v, L, Lb, Ls = core.run(data, time, {"floor_rise_gate": 1}, log=True)
    xs = L["x_slow"]
    xf = L["x_fast"]
    ap = L["applied"]
    z = L["zero"]
    y = data - z
    e = np.maximum(y - L["x_fast"] - xs, 0.0)
    cap = rsm * np.maximum(e, 1.0)
    m = (time >= 1010.0) & (time < 1059.0)
    locked = xs > (cap + 1.0)
    rows = []
    for lo, hi in ((1010, 1020), (1020, 1040), (1040, 1059)):
        mm = (time >= lo) & (time < hi)
        rows.append(dict(
            window=f"{lo}-{hi}",
            xslow_med=round(float(np.median(xs[mm])), 2),
            xfast_med=round(float(np.median(xf[mm])), 2),
            applied_med=round(float(np.median(ap[mm])), 2),
            y_med=round(float(np.median(y[mm])), 1),
            e_med=round(float(np.median(e[mm])), 1),
            e_zero_ratio=round(float((e[mm] <= 0.0).mean()), 3),
            xslow_gt_cap_ratio=round(float(locked[mm].mean()), 3),
            xslow_over_y_ratio=round(float(np.mean(xs[mm] > rsm * np.maximum(y[mm], 1.0))), 3),
        ))
    print(f"\n=== 每通道卸载 {step:.0f}（gate=1）r_slow_max={rsm} ===")
    for r in rows:
        print(f"  {r['window']:9s} x_slow={r['xslow_med']:7.2f} x_fast={r['xfast_med']:6.2f} "
              f"applied={r['applied_med']:7.2f} y={r['y_med']:7.1f} e={r['e_med']:6.1f} "
              f"e<=0占比={r['e_zero_ratio']:.2f} x_slow>cap={r['xslow_gt_cap_ratio']:.2f} "
              f"x_slow>0.6y={r['xslow_over_y_ratio']:.2f}")
    OUT.mkdir(exist_ok=True)
    (OUT / f"xslow_lock_{int(step)}.json").write_text(
        json.dumps({"step": step, "r_slow_max": rsm, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    del v, L, Lb, Ls
