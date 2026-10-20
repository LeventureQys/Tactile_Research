"""秒级读数连续性指标（从已保存的秒级曲线计算，不重跑算法）。

给甲方讲"读数连不连续"比讲"补偿量塌没塌"更直接，但阈值必须对两条臂公平：
补偿量大的臂更容易触发"补偿跌幅"，所以这里只统计**显示读数本身**的逐秒变化。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ADC_PER_N = 1000.0


def stats(t, y, win):
    m = (t >= win[0]) & (t <= win[1])
    tw, yw = t[m], y[m]
    d = np.abs(np.diff(yw))
    head = yw[: max(1, len(yw) // 60)].mean()
    tail = yw[-max(1, len(yw) // 60):].mean()
    # 单向上爬的最大累积量（"越按越高"的直接量化）
    rel = yw - head
    return dict(
        jump_p50=float(np.percentile(d, 50)),
        jump_p90=float(np.percentile(d, 90)),
        jump_p99=float(np.percentile(d, 99)),
        jump_max=float(d.max()),
        jumps_over_100adc=int((d > 100.0).sum()),
        monotonic_rise_max=float(max(0.0, rel.max())),
        out_head=float(head), out_tail=float(tail),
    )


def main():
    mpath = OUT / "metrics.json"
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    for key in [k for k, v in meta.items() if isinstance(v, dict) and "arms" in v]:
        s = np.load(OUT / f"series_{key}.npz")
        t = s["t"]
        win = meta[key]["arms"]["MEAS"]["window"]
        for arm in ("MEAS", "FIX", "FIXG"):
            y = s[f"out_{arm}"]
            st = stats(t, y, win)
            meta[key]["arms"][arm]["readout"] = st
            print(f"{key:6s} {arm:5s} p99={st['jump_p99']:7.1f} max={st['jump_max']:8.1f} "
                  f">100ADC={st['jumps_over_100adc']:5d} rise_max={st['monotonic_rise_max']:8.1f}")
    mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", mpath)


if __name__ == "__main__":
    main()
