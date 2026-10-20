# -*- coding: utf-8 -*-
"""逐通道归因：195s 下探在哪些通道发生，补偿量在哪些通道减少。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\左拇指指腹 r_slow_max=0.6\测试1-2026-10-14_10-18-40")
OUT = Path(__file__).resolve().parent


def load_stream(name: str) -> pd.DataFrame:
    path = FOLDER / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def main() -> None:
    session = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    pre = load_stream("device_001_pre_seg0.csv")
    seg = load_stream("device_001_seg000.csv")
    v = np.asarray(pre.iloc[:, 3:], dtype=float)
    s = np.asarray(seg.iloc[:, 3:], dtype=float)
    frame_index = np.asarray(pre.iloc[:, 2], dtype=float)
    t = (frame_index - frame_index[0]) / fps

    applied = v - s  # 固件实际扣除量（近似，seg 为显示值）

    def band(a, b):
        return (t >= a) & (t < b)

    before = applied[band(193.0, 194.4)].mean(axis=0)
    after = applied[band(195.5, 197.0)].mean(axis=0)
    change = after - before

    dip_before = v[band(194.3, 194.5)].mean(axis=0)
    dip_idx = np.flatnonzero(band(194.5, 194.55))
    dip_min = v[dip_idx].min(axis=0)
    dip_drop = dip_min - dip_before

    rows = []
    for ch in range(v.shape[1]):
        rows.append(dict(ch=ch,
                         dip_drop=round(float(dip_drop[ch]), 1),
                         applied_before=round(float(before[ch]), 1),
                         applied_after=round(float(after[ch]), 1),
                         applied_change=round(float(change[ch]), 1)))

    summary = dict(
        total_applied_before=round(float(before.sum()), 1),
        total_applied_after=round(float(after.sum()), 1),
        total_applied_change=round(float(change.sum()), 1),
        total_dip_drop=round(float(dip_drop.sum()), 1),
        dip_channels_deeper_than_10=int((dip_drop < -10).sum()),
        applied_channels_down_more_than_5=int((change < -5).sum()),
        applied_channels_up_more_than_5=int((change > 5).sum()),
    )
    top_dip = sorted(rows, key=lambda r: r["dip_drop"])[:10]
    top_change = sorted(rows, key=lambda r: r["applied_change"])[:10]
    payload = dict(fps=round(fps, 4), summary=summary, channels=rows,
                   top_dip_channels=top_dip, top_applied_drop_channels=top_change)
    (OUT / "channel_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("下探最深的10个通道:", json.dumps(top_dip, ensure_ascii=False))
    print("补偿减少最多的10个通道:", json.dumps(top_change, ensure_ascii=False))


if __name__ == "__main__":
    main()
