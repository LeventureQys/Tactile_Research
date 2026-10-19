# -*- coding: utf-8 -*-
"""在真实录制上按小台阶逐次核对「慢态是否吞掉真实小变化」。

判据：台阶前 2.5 s 中值 → 台阶后 1~3 s 中值；保留率 = 方法输出台阶幅值 / 未补偿输入台阶幅值。
只用已缓存的 series/*.npz（不重算算法）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ("out_rec", "k9_rec", "k9_0p4", "routeb")
SETS = [
    ("设备1_测试1", 300.0),
    ("设备1_测试2", 300.0),
    ("设备1_10g反复增减", 150.0),
    ("设备2_测试1", 300.0),
    ("设备2_测试2", 300.0),
    ("设备2_10g反复增减", 150.0),
]


def med(t, sig, a, b):
    z = sig[(t >= a) & (t < b)]
    return float(np.median(z)) if z.size else None


def detect(t, sig, min_adc, refractory_s=2.0):
    d = np.diff(sig)
    hits = np.flatnonzero(np.abs(d) >= min_adc)
    out = []
    for i in hits:
        if out and t[i] - out[-1]["t"] < refractory_s:
            out[-1]["t"] = float(t[i])
            out[-1]["i"] = int(i)
            out[-1]["peak"] = max(out[-1]["peak"], abs(float(d[i])))
            continue
        out.append(dict(i=int(i), t=float(t[i]), peak=abs(float(d[i]))))
    return out


def main() -> int:
    report = {}
    for key, threshold in SETS:
        data = np.load(HERE / "series" / ("%s.npz" % key))
        t = data["t"]
        totals = {m: data["total_" + m] for m in ("pre",) + METHODS}
        rows = []
        for step in detect(t, totals["pre"], threshold):
            ts = step["t"]
            before = {m: med(t, totals[m], ts - 2.5, ts - 0.5) for m in totals}
            after = {m: med(t, totals[m], ts + 1.0, ts + 3.0) for m in totals}
            if any(v is None for v in before.values()) or any(v is None for v in after.values()):
                continue
            pre_amp = after["pre"] - before["pre"]
            if abs(pre_amp) < threshold:
                continue
            row = dict(t=round(ts, 2), pre_adc=round(pre_amp, 1))
            for m in METHODS:
                amp = after[m] - before[m]
                row[m] = round(amp, 1)
                row[m + "_ret"] = round(amp / pre_amp, 3) if abs(pre_amp) > 1e-9 else None
            rows.append(row)
        report[key] = rows
        print("\n=== %s （阈值 %.0f ADC，检出 %d 个台阶）" % (key, threshold, len(rows)))
        print("  %-8s %9s | %9s %7s | %9s %7s | %9s %7s | %9s %7s"
              % ("时刻s", "输入ADC", "录制", "保留", "K9 r=.1", "保留", "K9 r=.4", "保留",
                 "路线B", "保留"))
        for row in rows:
            print("  %-8.2f %9.1f | %9.1f %7.3f | %9.1f %7.3f | %9.1f %7.3f | %9.1f %7.3f"
                  % (row["t"], row["pre_adc"],
                     row["out_rec"], row["out_rec_ret"], row["k9_rec"], row["k9_rec_ret"],
                     row["k9_0p4"], row["k9_0p4_ret"], row["routeb"], row["routeb_ret"]))
        if rows:
            for m in METHODS:
                vals = [r[m + "_ret"] for r in rows if r[m + "_ret"] is not None]
                if vals:
                    print("  -> %s 平均保留率 %.3f（最差 %.3f，达标(>=0.7) %d/%d）"
                          % (m, float(np.mean(vals)), float(np.min(vals)),
                             int(np.sum(np.array(vals) >= 0.7)), len(vals)))
    (HERE / "step_retention.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> %s" % (HERE / "step_retention.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
