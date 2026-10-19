# -*- coding: utf-8 -*-
"""路线B 事件通路到底有没有介入？—— 与「K9 单改 r_slow_max=0.4」逐帧做差。

若 (路线B 输出) 与 (K9@0.4 输出) 在整段录制上几乎处处相等，说明事件检测/平台锁存
在真实数据上没有产生任何补偿动作，恒载侧的改善全部来自 r_slow_max 这一个旋钮。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = ["设备1/测试1", "设备1/测试2", "设备1/10g反复增减",
          "设备2/测试1", "设备2/测试2", "设备2/10g反复增减"]


def main() -> int:
    payload = json.loads((HERE / "compare_metrics.json").read_text(encoding="utf-8"))
    by_label = {row.get("label", row["id"]): row for row in payload["sessions"]}
    print("%-22s %10s %10s %10s  %-38s" % ("会话", "max|B-K9.4|", "非零帧占比", "事件数",
                                           "事件时刻(动作,ΔADC,平台ADC)"))
    for label in LABELS:
        data = np.load(HERE / "series" / ("%s.npz" % label.replace("/", "_")))
        diff = data["total_routeb"] - data["total_k9_0p4"]
        t = data["t"]
        row = by_label[label]
        events = row.get("routeb_events", [])
        active = np.abs(diff) > 1.0
        text = "; ".join("%.2fs %s %+.0f→%.0f" % (e["time_s"], e["action"],
                                                 e["delta_total_adc"], e["platform_total_adc"])
                         for e in events[:5]) or "无"
        print("%-22s %10.1f %9.2f%% %10d  %s"
              % (label, float(np.max(np.abs(diff))), 100.0 * float(np.mean(active)),
                 len(events), text))
        if bool(active.any()):
            idx = np.flatnonzero(active)
            print("%-22s   非零区间: %.2f~%.2f s（首 %.2f s 前 %d 帧 / 之后 %d 帧）"
                  % ("", t[idx[0]], t[idx[-1]],
                     float(row["load_time_s"]) + 8.0,
                     int(np.sum(idx < np.searchsorted(t, row["load_time_s"] + 8.0))),
                     int(np.sum(idx >= np.searchsorted(t, row["load_time_s"] + 8.0)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
