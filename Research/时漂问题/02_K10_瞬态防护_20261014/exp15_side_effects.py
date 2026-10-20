# -*- coding: utf-8 -*-
"""补测：瞬态防护对原有指标的影响（漂移、稳态、小阶跃保留、撤载残差）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右拇指指腹")
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLBOX))
from guarded_k9 import compensate  # noqa: E402
from plugins.k9_creep_observer import _compensate  # noqa: E402

GUARD = dict(enabled=True)
SESSIONS = [
    ("设备1/测试1", "设备1", "测试1-2026-10-13_16-32-15"),
    ("设备1/测试2", "设备1", "测试2-2026-10-13_16-43-59"),
    ("设备1/10g", "设备1", "10g反复增减荷载-2026-10-13_17-07-18"),
    ("设备2/测试1", "设备2", "测试1-2026-10-13_16-53-24"),
    ("设备2/测试2", "设备2", "测试2-2026-10-13_17-01-06"),
    ("设备2/10g", "设备2", "10g反复增减荷载-2026-10-13_17-05-32"),
]
INJECT_HOST = "设备1/测试2"


def load_stream(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip() == "##Data":
                skip = index + 1
                break
        else:
            raise ValueError(f"no ##Data in {path}")
    return pd.read_csv(path, skiprows=skip, header=0, engine="c")


def med(sig, t, a, b):
    mask = (t >= a) & (t < b)
    return float(np.median(sig[mask])) if mask.any() else float("nan")


def main() -> None:
    rows = []
    host = None
    for label, device, name in SESSIONS:
        folder = DATA / device / name
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        params = session["algorithm"]["params"]
        fps = float(session["devices"][0]["observed_fps"])
        pre = load_stream(folder, "device_001_pre_seg0.csv")
        values = np.asarray(pre.iloc[:, 3:], dtype=float)
        frame = np.asarray(pre.iloc[:, 2], dtype=float)
        t = (frame - frame[0]) / fps

        plain = np.asarray(_compensate(values, t, params)[0], dtype=float).sum(axis=1)
        guarded = np.asarray(compensate(values, t, params, GUARD)[0], dtype=float).sum(axis=1)

        jumps = np.flatnonzero(np.abs(np.diff(values.sum(axis=1))) >= 2000.0)
        load = float(t[jumps[0] + 1]) if jumps.size else float(t[0])
        start = load + 15.0
        head = (t >= start) & (t < start + 5.0)
        tail = t >= float(t[-1]) - 5.0

        steady = []
        for lo in np.arange(start, float(t[-1]) - 60.0, 60.0):
            mask = (t >= lo) & (t < lo + 60.0)
            if mask.any():
                steady.append(abs(med(plain, t, lo, lo + 60.0) - med(guarded, t, lo, lo + 60.0)))

        rows.append(dict(
            id=label, frames=int(t.size),
            drift_plain=round(float(np.median(plain[tail]) - np.median(plain[head])), 2),
            drift_guarded=round(float(np.median(guarded[tail]) - np.median(guarded[head])), 2),
            max_60s_diff=round(max(steady), 3) if steady else None,
            mean_60s_diff=round(float(np.mean(steady)), 3) if steady else None,
            max_instant_diff=round(float(np.max(np.abs(plain - guarded))), 2),
        ))
        if label == INJECT_HOST:
            host = dict(label=label, params=params, values=values, t=t, load=load,
                        plain_clean=plain, guarded_clean=guarded)
        print(label, json.dumps(rows[-1], ensure_ascii=False), flush=True)

    result = dict(sessions=rows, injection=None)

    if host is not None:
        t = host["t"]; values = host["values"]; params = host["params"]
        start = host["load"] + 30.0
        rise = 2.5
        hold = 15.0
        level = np.clip((t - start) / rise, 0.0, 1.0)
        level[t >= start + hold] = 0.0
        gate = t >= start
        level = np.where(gate, level, 0.0)
        injected = values.copy()
        injected += level[:, None] * (100.0 / values.shape[1])

        plain_inj = np.asarray(_compensate(injected, t, params)[0], dtype=float).sum(axis=1)
        guarded_inj = np.asarray(compensate(injected, t, params, GUARD)[0], dtype=float).sum(axis=1)

        hold_mask = (t >= start + 10.0) & (t < start + 15.0)
        release_mask = (t >= start + 22.0) & (t < start + 27.0)
        result["injection"] = dict(
            host=host["label"], start_s=round(start, 2), rise_s=rise, hold_s=hold,
            injected_total_adc=100.0, shape="分散57通道",
            plain_hold=round(float(np.median((plain_inj - host["plain_clean"])[hold_mask])), 2),
            guarded_hold=round(float(np.median((guarded_inj - host["guarded_clean"])[hold_mask])), 2),
            plain_retention=round(float(np.median((plain_inj - host["plain_clean"])[hold_mask]) / 100.0), 4),
            guarded_retention=round(float(np.median((guarded_inj - host["guarded_clean"])[hold_mask]) / 100.0), 4),
            plain_release=round(float(np.median((plain_inj - host["plain_clean"])[release_mask])), 2),
            guarded_release=round(float(np.median((guarded_inj - host["guarded_clean"])[release_mask])), 2),
            clean_pair_max_diff=round(float(np.max(np.abs(host["plain_clean"] - host["guarded_clean"]))), 2),
        )
        print("injection", json.dumps(result["injection"], ensure_ascii=False), flush=True)

    (HERE / "side_effect_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
