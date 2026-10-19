"""Re-run K12 v2 against the in-service K9 on the right-thumb-pulp recordings.

In-service K9 = the algorithm stored in the device session
(`algorithm.id = drift_v34_creep_observer`, `device_001_seg000.csv`), i.e. what the
firmware actually applied during the recording. Toolbox replay K9 is kept only as a
fidelity check that our offline copy matches the recorded output.

Writes results/k9_vs_k12v2/{session}.npz plus summary/events/options JSON.
Run: C:\\Python314\\python.exe scripts/k9_vs_k12v2_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(TOOLBOX))

from dptool import algorithms as alg, session_csv as scsv  # noqa: E402

from k12_v2_core import DEFAULTS, compensate  # noqa: E402

ROOT = HERE.parent
DATA = ROOT.parents[1] / "data" / "右拇指指腹"
OUT = ROOT / "results" / "k9_vs_k12v2"
LABELS = (("in_service_k9", "现役K9(设备录制)"), ("replay_k9", "重放K9(工具箱)"),
          ("k12_v2", "K12_v2"), ("k9_tuned", "调参K9(r_slow_max=0.3)"))


def load_cores():
    registry = alg.PluginRegistry()
    registry.scan()
    k9 = registry.get("k9_creep_observer")
    params = alg.default_params(k9.specs)
    params.update(r_fast=.03, tau_c_fast_s=2, tau_r_fast_s=.1, r_slow_max=.1,
                  slope_cap_frac=.02, slow_confirm_s=3, soft_unfreeze_s=2,
                  tau_r_slow_idle_s=.2)
    return k9.module._compensate, params


def window_change(axis, signal, start):
    head = (axis >= start) & (axis < start + 5)
    tail = axis >= axis[-1] - 5
    return float(np.median(signal[tail]) - np.median(signal[head]))


def main() -> int:
    core, params = load_cores()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for manifest in sorted(DATA.glob("*/*/session.json")):
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        folder = manifest.parent
        device = meta["devices"][0]
        pre = scsv.load_table(str(folder / "device_001_pre_seg0.csv"), use_cache=False)
        recorded = scsv.load_table(str(folder / device["data_file"]), use_cache=False)
        assert pre.n_rows == recorded.n_rows, f"{folder.name}: row mismatch"
        assert np.array_equal(pre.data[:, 2], recorded.data[:, 2]), f"{folder.name}: frame mismatch"
        values = pre.channels().astype(float)
        axis = np.asarray(pre.data[:, 2], float) / float(device["observed_fps"])
        axis = axis - axis[0]

        cfg = dict(params)
        cfg.update({key: val for key, val in meta["algorithm"]["params"].items() if key in cfg})
        replay, _ = core(values, axis, cfg)
        tuned, _ = core(values, axis, dict(cfg, r_slow_max=.3))
        new, trace = compensate(values, axis, cfg, core)

        total = values.sum(1)
        jumps = np.flatnonzero(np.abs(np.diff(total)) >= DEFAULTS["large_step_adc"])
        loaded = float(axis[jumps[0] + 1]) if len(jumps) else 0.
        start = loaded + 15.
        name = f"{folder.parent.name}_{folder.name}"
        signals = {"in_service_k9": recorded.channels().sum(1), "replay_k9": replay.sum(1),
                   "k12_v2": new.sum(1), "k9_tuned": tuned.sum(1)}

        np.savez_compressed(
            OUT / f"{name}.npz", time=axis, pre=total,
            in_service_k9=signals["in_service_k9"], replay_k9=signals["replay_k9"],
            k12_v2=signals["k12_v2"], k9_tuned=signals["k9_tuned"],
            offset_total=trace["offset_total"], guard=trace["guard"],
            large_guard=trace["large_guard"])
        (OUT / f"{name}_events.json").write_text(
            json.dumps(trace["events"], ensure_ascii=False, indent=2), encoding="utf-8")

        row = dict(file=f"{name}.npz", device=folder.parent.name, session=folder.name,
                   frames=int(pre.n_rows), fps=round(float(device["observed_fps"]), 3),
                   loaded_at=loaded, analysis_start=start, events=len(trace["events"]),
                   final_ledger_adc=float(trace["offset_total"][-1]),
                   in_service_vs_replay_adc=float(np.max(np.abs(
                       signals["in_service_k9"] - signals["replay_k9"]))))
        for key, _ in LABELS:
            row[key + "_window_change_adc"] = window_change(axis, signals[key], start)
        row["k12_gain_vs_in_service_adc"] = (row["in_service_k9_window_change_adc"]
                                             - row["k12_v2_window_change_adc"])
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (OUT / "options.json").write_text(json.dumps(
        dict(k9=params, k12=DEFAULTS, data_dir=str(DATA)), ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
