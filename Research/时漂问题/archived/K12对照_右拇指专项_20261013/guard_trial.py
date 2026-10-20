"""Independent four-arm K9/K12-v2 comparison on six right-thumb recordings.

Run: python guard_trial.py
Only guard_trial_results.json is written; source recordings and prior reports are read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "右拇指指腹"
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
K12_DIR = HERE.parents[1] / "archived" / "历史版本" / "06_K12_旧实验归档_2026-10-13" / "scripts"
OUT = HERE / "guard_trial_results.json"
sys.path.insert(0, str(TOOLBOX))
sys.path.insert(0, str(K12_DIR))
from dptool.session_csv import load_table  # noqa: E402
from plugins.k9_creep_observer import _compensate  # noqa: E402
from k12_v2_core import DEFAULTS, compensate  # noqa: E402

ARMS = ("k9_original", "k9_rslow_0_3", "k12_v2_original", "k12_v2_rslow_0_3")


def median(signal, mask):
    subset = signal[mask]
    if not subset.size:
        raise ValueError("Empty measurement window")
    return float(np.median(subset))


def metrics(time, signal, load_time):
    start = load_time + 15.0
    end = float(time[-1])
    if end - start < 10.0:
        raise ValueError("Constant-load measurement requires at least 10 seconds")
    baseline = median(signal, (time >= start) & (time < start + 5))
    tail = median(signal, time >= end - 5)
    starts = np.arange(start, end - 5 + 1e-9, 1.0)
    starts = np.append(starts, end - 5)
    offsets = [median(signal, (time >= point) & (time < point + 5)) - baseline
               for point in starts[:-1]]
    offsets.append(tail - baseline)
    return dict(start_s=start, end_s=end, first_5s_median_adc=baseline,
                last_5s_median_adc=tail, net_change_adc=tail - baseline,
                max_abs_offset_adc_5s_median=float(np.max(np.abs(offsets))))


def load_recording(folder):
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    device = session["devices"][0]
    fps = float(device["observed_fps"])
    if fps <= 0:
        raise ValueError(f"Bad observed_fps: {folder}")
    pre = load_table(str(folder / device["algorithm_input_file"]), use_cache=False)
    recorded = load_table(str(folder / device["algorithm_result_file"]), use_cache=False)
    frame_index = np.asarray(pre.data[:, 2], dtype=float)
    if pre.n_rows != recorded.n_rows or not np.array_equal(frame_index, recorded.data[:, 2]):
        raise ValueError(f"Input/output frame mismatch: {folder}")
    time = (frame_index - frame_index[0]) / fps
    values = pre.channels().astype(float)
    if values.shape[1] != 57 or not np.isfinite(values).all():
        raise ValueError(f"Expected 57 finite input channels: {folder}")
    total = values.sum(axis=1)
    jumps = np.flatnonzero(np.abs(np.diff(total)) >= 2000.0)
    if not jumps.size:
        raise ValueError(f"No >=2000 ADC first load jump: {folder}")
    return session, values, time, float(time[jumps[0] + 1])


def replay(values, time, params, arm):
    selected = dict(params)
    if arm.endswith("0_3"):
        selected["r_slow_max"] = 0.3
    if arm.startswith("k12"):
        output, trace = compensate(values, time, selected, _compensate)
        return output.sum(axis=1), trace["events"]
    output, _ = _compensate(values, time, selected)
    return output.sum(axis=1), None


def candidate_summary(events, start, end):
    candidates = [dict(time_s=float(event["time_s"]),
                       delta_total_adc=float(event["delta_total_adc"]))
                  for event in events if start <= event["time_s"] < end]
    return dict(count=len(candidates), events=candidates)


def injected_pair(values, time, params, load_time, clean_signals):
    start = load_time + 30.0
    end = start + 15.0
    if end + 5 > time[-1]:
        raise ValueError("No complete injection withdrawal window")
    altered = values.copy()
    frame_mask = (time >= start) & (time < end)
    altered[frame_mask] += 100.0 / values.shape[1]
    result = dict(start_s=start, end_s=end, amplitude_total_adc=100.0,
                  channel_count=int(values.shape[1]), injected_frames=int(frame_mask.sum()), arms={})
    for arm in ARMS:
        changed, events = replay(altered, time, params, arm)
        difference = changed - clean_signals[arm]
        hold = median(difference, (time >= end - 5) & (time < end))
        withdrawal = median(difference, (time >= end) & (time < end + 5))
        record = dict(hold_last_5s_delta_adc=hold, hold_retention_fraction=hold / 100.0,
                      withdrawal_first_5s_delta_adc=withdrawal,
                      withdrawal_residual_fraction=withdrawal / 100.0)
        if events is not None:
            record["injected_candidate_events"] = candidate_summary(events, start, end + 5)
        result["arms"][arm] = record
    return result


def main():
    folders = sorted(manifest.parent for manifest in DATA.glob("*/*/session.json"))
    if len(folders) != 6:
        raise ValueError(f"Expected six recordings, found {len(folders)}")
    results = dict(method={
        "k9_core": str(TOOLBOX / "plugins" / "k9_creep_observer.py"),
        "k12_core": str(K12_DIR / "k12_v2_core.py"),
        "k12_options": DEFAULTS,
        "time_axis": "(frame_index - first_frame_index) / observed_fps",
        "first_load": "first consecutive-frame absolute total input ADC jump >= 2000",
        "constant_load": "from first load +15s, first 5s median vs recording last 5s median; maximum absolute full 5s window median offset relative to first 5s (1s stride plus final window)",
        "ten_g": "candidate net change only; not actual event detection or detection rate",
        "candidate_events": "K12 ledger events inside +15s to recording end are unverified candidates, not confirmed false positives",
        "injection": "device1 test2: load+30s through +45s, 100 total ADC evenly on 57 channels; injected minus clean output, last 5s loaded and first 5s after withdrawal",
    }, samples=[], injection=None, verdict=None)
    injection_folder = DATA / "设备1" / "测试2-2026-10-13_16-43-59"
    for folder in folders:
        session, values, time, load_time = load_recording(folder)
        params = session["algorithm"]["params"]
        kind = "ten_g" if "10g" in folder.name else "constant_load"
        row = dict(id=f"{folder.parent.name}/{folder.name}", kind=kind,
                   observed_fps=float(session["devices"][0]["observed_fps"]),
                   frame_count=len(time), load_time_s=load_time,
                   original_r_slow_max=float(params["r_slow_max"]), arms={})
        clean_signals = {}
        for arm in ARMS:
            signal, events = replay(values, time, params, arm)
            clean_signals[arm] = signal
            measured = metrics(time, signal, load_time)
            row["arms"][arm] = ({"candidate_net_change_adc": measured["net_change_adc"]}
                                if kind == "ten_g" else {"window": measured})
            if events is not None and kind == "constant_load":
                row["arms"][arm]["hold_candidate_events"] = candidate_summary(
                    events, load_time + 15, float(time[-1]))
        if folder == injection_folder:
            results["injection"] = dict(recording=row["id"],
                                        **injected_pair(values, time, params, load_time, clean_signals))
        results["samples"].append(row)
        print(row["id"], flush=True)
    if results["injection"] is None:
        raise ValueError("Injection recording missing")
    holds = [row for row in results["samples"] if row["kind"] == "constant_load"]
    worse = []
    for row in holds:
        arms = row["arms"]
        combined = arms["k12_v2_rslow_0_3"]
        base = arms["k12_v2_original"]
        if (combined["hold_candidate_events"]["count"] > base["hold_candidate_events"]["count"]
                or abs(combined["window"]["net_change_adc"]) > abs(base["window"]["net_change_adc"])
                or combined["window"]["max_abs_offset_adc_5s_median"] > base["window"]["max_abs_offset_adc_5s_median"]):
            worse.append(row["id"])
    over_limit = [row["id"] for row in holds if
                  row["arms"]["k12_v2_rslow_0_3"]["window"]["max_abs_offset_adc_5s_median"] > 300]
    candidates = [row["id"] for row in holds if
                  row["arms"]["k12_v2_rslow_0_3"]["hold_candidate_events"]["count"] > 0]
    injection = results["injection"]["arms"]
    combo = injection["k12_v2_rslow_0_3"]
    k12_base = injection["k12_v2_original"]
    injection_worse = (combo["hold_retention_fraction"] < k12_base["hold_retention_fraction"]
                       or abs(combo["withdrawal_residual_fraction"]) > abs(k12_base["withdrawal_residual_fraction"]))
    failed = bool(worse or over_limit or candidates or injection_worse)
    results["verdict"] = dict(combination_candidate_failed=failed,
                              worsened_vs_k12_recordings=worse,
                              over_300_adc_max_offset_recordings=over_limit,
                              constant_load_with_unverified_k12_candidates=candidates,
                              injection_retention_or_withdrawal_worse_than_k12=injection_worse,
                              rule="candidate fails on any K12 comparison regression, >300 ADC max offset, unverified K12 constant-load candidate, or paired-injection hold/withdrawal regression; no actual 10g detection claim")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {OUT}; combination_candidate_failed={failed}", flush=True)


if __name__ == "__main__":
    main()
