"""Render the K9-vs-K12_v2 comparison through the data-parsing toolbox.

Figures land directly in `figures/` with the `K9vsK12v2_` prefix so nothing from the
earlier K12 v1 / v2 rounds is overwritten. Reuses the verified helpers from
`k12_v2_plot.py` (SourceTable.from_arrays -> merge_sources -> plot_table).

Run: C:\\Python314\\python.exe scripts/k9_vs_k12v2_plot.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import k12_v2_plot as base  # noqa: E402

ROOT = HERE.parent
DATA_DIR = ROOT / "results" / "k9_vs_k12v2"
FIG_DIR = ROOT / "figures"
PREFIX = "K9vsK12v2_"
SERIES = (("01_现役K9(设备录制)", "in_service_k9"),
          ("02_重放K9(工具箱)", "replay_k9"),
          ("03_K12_v2", "k12_v2"),
          ("04_调参K9(r_slow_max=0.3)", "k9_tuned"))
FIELDS = ("time", "pre", "offset_total", "guard", "large_guard") + tuple(k for _, k in SERIES)


def render_session(item: dict, records: list[dict]) -> None:
    filename = base.filename_for(item)
    arrays = base.arrays_from(DATA_DIR / filename, FIELDS)
    time = arrays["time"]
    analysis_start, loaded_at = float(item["analysis_start"]), float(item["loaded_at"])
    if not (np.isfinite(analysis_start) and np.isfinite(loaded_at)
            and time[0] <= loaded_at <= analysis_start < time[-1]):
        raise ValueError(f"{filename}: loading/analysis bounds must be within time")
    start = int(np.searchsorted(time, analysis_start))
    if len(time) - start < 3:
        raise ValueError(f"{filename}: analysis window has fewer than three rows")
    elapsed = time[start:] - analysis_start
    device, session = str(item["device"]), str(item["session"])
    label = f"{device} / {session}"
    stem = PREFIX + re.sub(r"[^\w\-]+", "_", Path(filename).stem)
    curves = [(name, arrays[key][start:]) for name, key in SERIES]
    ledger = float(item["final_ledger_adc"])
    base.render(FIG_DIR / f"{stem}_绝对值.png", elapsed, curves,
           f"{label} | 加载后绝对总ADC | 现役K9 / 重放K9 / K12_v2 / 调参K9"
           f" | K12账本末值 {ledger:+.0f} ADC, 候选事件 {item['events']} 次",
           records, "absolute")

    baseline_stop = min(len(elapsed), max(3, int(np.searchsorted(elapsed, 1.0))))
    deltas = [(name, values - float(np.nanmedian(values[:baseline_stop])))
              for name, values in curves]
    base.render(FIG_DIR / f"{stem}_漂移增量.png", elapsed, deltas,
           f"{label} | 相对分析起点后首1秒中位的漂移增量 (ADC)"
           f" | 窗口净变化 现役K9 {item['in_service_k9_window_change_adc']:+.0f}"
           f" / K12_v2 {item['k12_v2_window_change_adc']:+.0f} ADC",
           records, "drift")
    if "10g" in session.lower() or "10g" in filename.lower():
        left, right = base.candidate_window(elapsed, arrays["in_service_k9"][start:])
        local = [(name, values[left:right]) for name, values in deltas]
        base.render(FIG_DIR / f"{stem}_候选窗.png", elapsed[left:right], local,
               f"{label} | 现役K9波动最大窗口（仅为候选波动，非已确认10g操作）"
               f" | {elapsed[left]:.1f}-{elapsed[right - 1]:.1f} s",
               records, "unverified_candidate")


def main() -> int:
    entries = base.entries_from(json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8")))
    expected = {base.filename_for(item) for item in entries}
    actual = {path.name for path in DATA_DIR.glob("*.npz")}
    if expected != actual:
        raise ValueError(f"summary mismatch: missing={sorted(actual - expected)}, "
                         f"unknown={sorted(expected - actual)}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for item in entries:
        render_session(item, records)
    log = FIG_DIR / "k9_vs_k12v2_plot_check.json"
    log.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(records)} figures; log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
