from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data" / "右拇指指腹"
OUT_ROOT = HERE / "autotune_results"
GUARD = dict(enabled=True, floor_confirm_s=0.5, idle_confirm_s=0.3,
             floor_deep_adc=200.0, floor_guard=True, idle_guard=True, spike_guard=False)
sys.path.insert(0, str(HERE))
from guarded_k9 import DEFAULT_PARAMS, compensate


@dataclass
class Session:
    folder: Path
    sid: str
    device: str
    kind: str
    t: np.ndarray
    x: np.ndarray
    recorded: np.ndarray | None
    events: list[dict[str, float]] | None = None
    event_info: dict[str, Any] | None = None


def table(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if line.strip() == "##Data":
                break
        else:
            raise ValueError(f"缺少 ##Data: {path}")
    return pd.read_csv(path, skiprows=i + 1, engine="c")


def load(root: Path) -> list[Session]:
    result = []
    for manifest in sorted(root.rglob("session.json")):
        folder = manifest.parent
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        dev = (meta.get("devices") or [{}])[0]
        name = dev.get("algorithm_input_file")
        if not name or not (folder / name).exists():
            continue
        frame = table(folder / name)
        if frame.shape[1] < 4:
            continue
        fps = float(dev.get("observed_fps") or 100.0)
        raw_t = frame.iloc[:, 0].to_numpy(float)
        t = raw_t - raw_t[0]
        if np.any(np.diff(t) < 0):
            raise ValueError(f"时间戳回退: {folder}")
        x = frame.iloc[:, 3:].to_numpy(float)
        sid = str(meta.get("session_id") or folder.name)
        result_path = folder / (dev.get("algorithm_result_file") or "")
        recorded = None
        if result_path.exists():
            out = table(result_path)
            if out.shape[0] == x.shape[0] and out.shape[1] >= x.shape[1] + 3:
                recorded = out.iloc[:, 3:3 + x.shape[1]].to_numpy(float)
        result.append(Session(folder, sid, str(dev.get("device_id", "unknown")),
                              "small_load" if "10g" in sid.lower() else "hold",
                              t, x, recorded))
    return result


def med(t, y, a, b):
    m = (t >= a) & (t < b)
    return float(np.median(y[m])) if m.any() else None


def bins(t, y, a, b, width=10.0):
    out = []
    while a + width <= b:
        v = med(t, y, a, a + width)
        if v is not None:
            out.append((a + width / 2, v))
        a += width
    return out


def hold_onset(t, total, minimum=300.0):
    q = bins(t, total, 0.0, float(t[-1]), 0.5)
    if len(q) < 4:
        return 0.0
    d = np.diff([v for _, v in q])
    i = int(np.argmax(d))
    return float(q[i + 1][0]) if d[i] >= minimum else 0.0


def detect_events(t, total, minimum=30.0, maximum=800.0):
    centers = np.arange(0.35, max(0.35, float(t[-1]) - 0.35), 0.1)
    delta = []
    for c in centers:
        a = med(t, total, c - 0.35, c - 0.1)
        b = med(t, total, c + 0.1, c + 0.35)
        delta.append(np.nan if a is None or b is None else b - a)
    delta = np.asarray(delta)
    finite = delta[np.isfinite(delta)]
    sigma = 1.4826 * np.median(np.abs(finite - np.median(finite))) if finite.size else 0.0
    threshold = max(minimum, 3.0 * float(sigma))
    active = np.flatnonzero(np.isfinite(delta) & (np.abs(delta) >= threshold))
    # Require one same-sign neighboring sample to suppress isolated noise spikes.
    persistent = []
    for i in active:
        left = delta[i - 1] if i > 0 else np.nan
        right = delta[i + 1] if i + 1 < delta.size else np.nan
        if (np.isfinite(left) and left * delta[i] > 0.0 and abs(left) >= threshold) or (
            np.isfinite(right) and right * delta[i] > 0.0 and abs(right) >= threshold
        ):
            persistent.append(i)
    active = np.asarray(persistent, dtype=int)
    events = []
    for i in active:
        if abs(delta[i]) > maximum:
            continue
        if events and centers[i] - events[-1]["time_s"] < 1.25:
            if abs(delta[i]) > abs(events[-1]["input_delta_adc"]):
                events[-1]["time_s"] = float(centers[i])
                events[-1]["input_delta_adc"] = float(delta[i])
            continue
        events.append({"time_s": float(centers[i]), "input_delta_adc": float(delta[i])})
    return events, {"threshold_adc": threshold, "noise_sigma_adc": float(sigma), "candidate_count": len(events)}


def hold_metric(s: Session, y, settle=15.0, scale=1000.0):
    total = s.x.sum(1); out = y.sum(1)
    onset = hold_onset(s.t, total)
    a = onset + settle; b = float(s.t[-1]) - 1.0
    if b - a < 5:
        return {"valid": False, "reason": "恒载窗口不足", "onset_s": onset}
    ref = med(s.t, out, a, a + min(10, (b - a) / 3))
    tail = med(s.t, out, max(a, b - 10), b)
    windows = bins(s.t, out, a, b)
    offset = max([abs(v - ref) for _, v in windows] or [abs(tail - ref)])
    late = bins(s.t, out, max(a, b - 120), b)
    slope = float(np.polyfit([q for q, _ in late], [v for _, v in late], 1)[0]) if len(late) > 1 else None
    return {"valid": True, "onset_s": onset, "start_s": a, "end_s": b,
            "endpoint_drift_adc": tail - ref, "max_abs_offset_adc": offset,
            "endpoint_drift_n_rough": (tail - ref) / scale, "tail_slope_adc_s": slope}


def small_metric(s: Session, y, retention=0.7, platform_horizon=4.0):
    total = y.sum(1); rows = []
    for i, e in enumerate(s.events or []):
        c = e["time_s"]; inp = e["input_delta_adc"]
        before = med(s.t, total, c - 0.35, c - 0.1)
        after = med(s.t, total, c + 0.1, c + 0.35)
        if before is None or after is None:
            continue
        delta = after - before
        ratio = math.copysign(1, inp) * delta / max(abs(inp), 1e-9)
        next_c = s.events[i + 1]["time_s"] if i + 1 < len(s.events) else float(s.t[-1])
        platform = None
        if next_c >= c + platform_horizon + 0.5:
            late = med(s.t, total, c + platform_horizon, c + platform_horizon + 0.5)
            if late is not None:
                platform = math.copysign(1, inp) * (late - before) / max(abs(inp), 1e-9)
        rows.append({"time_s": c, "input_delta_adc": inp, "output_delta_adc": delta,
                     "retention_ratio": ratio, "event_pass": ratio >= retention,
                     "platform_retention_ratio": platform})
    ratios = [r["retention_ratio"] for r in rows]
    platforms = [r["platform_retention_ratio"] for r in rows if r["platform_retention_ratio"] is not None]
    return {"event_count": len(rows), "events_passed": sum(r["event_pass"] for r in rows),
            "median_retention": float(np.median(ratios)) if ratios else None,
            "minimum_retention": min(ratios) if ratios else None,
            "platform_count": len(platforms), "minimum_platform_retention": min(platforms) if platforms else None,
            "events": rows}


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def main():
    p = argparse.ArgumentParser(description="K10 右拇指 ADC 离线联合调参")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--r-slow-max", type=lambda x: [float(v) for v in x.split(",")], default=[.1, .2, .3, .4, .6])
    p.add_argument("--slope-cap-frac", type=lambda x: [float(v) for v in x.split(",")], default=[.01, .02, .03, .05])
    p.add_argument("--max-hold-drift-adc", type=float, default=300.0)
    p.add_argument("--min-retention", type=float, default=.70)
    p.add_argument("--platform-horizon-s", type=float, default=4.0)
    p.add_argument("--settling-exclusion-s", type=float, default=15.0)
    p.add_argument("--adc-per-newton", type=float, default=1000.0)
    args = p.parse_args()
    sessions = load(args.data_root)
    if not sessions or not any(s.kind == "hold" for s in sessions) or not any(s.kind == "small_load" for s in sessions):
        raise SystemExit("数据必须同时包含恒载和10g会话")
    for s in sessions:
        if s.kind == "small_load":
            s.events, s.event_info = detect_events(s.t, s.x.sum(1))
    base = dict(DEFAULT_PARAMS)
    meta = json.loads(next(args.data_root.rglob("session.json")).read_text(encoding="utf-8"))
    base.update((meta.get("algorithm") or {}).get("params") or {})
    out_dir = args.out_dir or OUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(itertools.product(args.r_slow_max, args.slope_cap_frac))
    summaries = []; session_rows = []; event_rows = []; baseline_rows = []
    for candidate_id, (slow, cap) in enumerate(candidates, 1):
        params = dict(base, r_slow_max=slow, slope_cap_frac=cap)
        local = []
        for s in sessions:
            y, _ = compensate(s.x, s.t, params, dict(GUARD, enabled=True))
            hold = hold_metric(s, y, args.settling_exclusion_s, args.adc_per_newton) if s.kind == "hold" else None
            small = small_metric(s, y, args.min_retention, args.platform_horizon_s) if s.kind == "small_load" else None
            row = {"candidate": candidate_id, "session_id": s.sid, "device_id": s.device,
                   "kind": s.kind, "r_slow_max": slow, "slope_cap_frac": cap,
                   "hold": hold, "small": {k: v for k, v in small.items() if k != "events"} if small else None}
            local.append(row); session_rows.append(row)
            for i, e in enumerate((small or {}).get("events", []), 1):
                event_rows.append({"candidate": candidate_id, "session_id": s.sid, "event_index": i, **e})
        holds = [r["hold"] for r in local if r["hold"] and r["hold"].get("valid")]
        events = [e for e in event_rows if e["candidate"] == candidate_id]
        platforms = [e["platform_retention_ratio"] for e in events if e["platform_retention_ratio"] is not None]
        max_offset = max([h["max_abs_offset_adc"] for h in holds] or [float("inf")])
        min_ratio = min([e["retention_ratio"] for e in events] or [-float("inf")])
        hold_pass = len(holds) == sum(s.kind == "hold" for s in sessions) and max_offset <= args.max_hold_drift_adc
        small_pass = bool(events) and min_ratio >= args.min_retention
        platform_pass = bool(platforms) and min(platforms) >= args.min_retention
        summaries.append({"candidate": candidate_id, "r_slow_max": slow, "slope_cap_frac": cap,
                          "worst_hold_max_abs_offset_adc": max_offset, "worst_hold_endpoint_abs_drift_adc": max([abs(h["endpoint_drift_adc"]) for h in holds] or [None]),
                          "minimum_small_event_retention": min_ratio, "small_events": len(events),
                          "platform_checks": len(platforms), "minimum_platform_retention": min(platforms) if platforms else None,
                          "hold_pass": hold_pass, "small_load_pass_proxy": small_pass,
                          "platform_pass_proxy": platform_pass, "joint_pass_proxy": hold_pass and small_pass and platform_pass,
                          "decision": "联合候选" if hold_pass and small_pass and platform_pass else ("待补平台证据" if hold_pass and small_pass else "未联合达标")})
        print(f"候选 {candidate_id}/{len(candidates)}: slow={slow:.3f}, cap={cap:.3f}, drift={max_offset:.1f}, retention={min_ratio:.3f}", flush=True)
    summaries.sort(key=lambda r: (not r["joint_pass_proxy"], r["worst_hold_max_abs_offset_adc"], -r["minimum_small_event_retention"]))
    for rank, row in enumerate(summaries, 1): row["rank"] = rank
    recommended = next((r for r in summaries if r["joint_pass_proxy"]), None)
    for s in sessions:
        if s.recorded is None: continue
        replay, _ = compensate(s.x, s.t, base, dict(GUARD, enabled=False))
        baseline_rows.append({"session_id": s.sid, "max_abs_channel_diff_adc": float(np.max(np.abs(replay - s.recorded))),
                             "max_abs_total_diff_adc": float(np.max(np.abs(replay.sum(1) - s.recorded.sum(1))) )})
    report = {"generated_at": datetime.now().astimezone().isoformat(), "tool_version": "0.1.0",
              "source": {"data_root": str(args.data_root), "algorithm": str(HERE / "guarded_k9.py"),
                         "sessions": [{"session_id": s.sid, "device_id": s.device, "kind": s.kind,
                                       "frames": int(s.x.shape[0]), "channels": int(s.x.shape[1]), "duration_s": float(s.t[-1]),
                                       "event_info": s.event_info} for s in sessions]},
              "optimizer": {"strategy": "穷举 r_slow_max × slope_cap_frac；K10 guard 开启；其余参数固定",
                            "candidates": len(candidates), "max_hold_drift_adc": args.max_hold_drift_adc,
                            "minimum_event_retention": args.min_retention, "adc_per_newton_rough": args.adc_per_newton},
              "decision": {"status": "joint_candidate_found" if recommended else "no_joint_candidate_in_grid",
                           "recommended": recommended, "force_mapping_table_available": False,
                           "limits": ["10g事件为总值自动提取，需人工核对操作时间标签。", "平台证据不足的候选不会联合通过。", "未覆盖强扰动、部分卸载、跨荷载量级和真实力值映射表。"]},
              "recorded_k9_replay_check": baseline_rows, "candidate_summaries": summaries}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "candidate_summary.csv", summaries); write_csv(out_dir / "session_metrics.csv", session_rows); write_csv(out_dir / "event_metrics.csv", event_rows)
    print(f"报告目录: {out_dir}"); print(f"联合候选: {recommended or '未找到'}")


if __name__ == "__main__":
    main()
