"""Render K12 v2 comparisons through dptool.merge, from results/v2 exports.

Run with C:\\Python314\\python.exe scripts/k12_v2_plot.py after summary.json and
session NPZ files are ready. Figures and plot_check.json live in figures/v2.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(TOOLBOX))
from dptool import merge as mg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "v2"
FIG_DIR = ROOT / "figures" / "v2"
SESSION_FIELDS = ("time", "pre", "recorded_k9", "replay_k9", "k12",
                  "k9_tuned", "offset_total", "guard", "large_guard")
SYNTHETIC_FIELDS = ("time", "reference", "step", "k9_response", "k12_response", "guard")
SERIES = (("01_recorded K9", "recorded_k9"),
          ("02_replay K9", "replay_k9"),
          ("03_K12 v2", "k12"),
          ("04_tuned K9", "k9_tuned"))


def arrays_from(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        absent = set(required) - set(archive.files)
        if absent:
            raise ValueError(f"{path.name}: missing fields {sorted(absent)}; found {archive.files}")
        arrays = {field: np.asarray(archive[field], dtype=np.float64).ravel()
                  for field in required}
    length = len(arrays["time"])
    if length < 3 or any(len(values) != length for values in arrays.values()):
        raise ValueError(f"{path.name}: field lengths disagree or fewer than three samples")
    if not np.all(np.isfinite(arrays["time"])) or np.any(np.diff(arrays["time"]) <= 0):
        raise ValueError(f"{path.name}: time must be finite, strictly increasing seconds")
    return arrays


def entries_from(summary: object) -> list[dict]:
    if isinstance(summary, list):
        entries = summary
    elif isinstance(summary, dict) and isinstance(summary.get("sessions"), list):
        entries = summary["sessions"]
    elif isinstance(summary, dict) and all(isinstance(v, dict) for v in summary.values()):
        entries = [dict(value, file=value.get("file", key)) for key, value in summary.items()]
    else:
        raise ValueError("summary.json: expected a sessions list or filename-to-metadata map")
    if not entries or any(not isinstance(item, dict) for item in entries):
        raise ValueError("summary.json: expected nonempty session metadata objects")
    return entries


def filename_for(item: dict) -> str:
    file = item.get("file") or item.get("npz") or item.get("filename")
    if not isinstance(file, str) or not file.strip():
        raise ValueError(f"metadata lacks file/npz/filename: {item}")
    filename = Path(file).name
    if filename != file or filename == "synthetic.npz":
        raise ValueError(f"invalid session file: {file}")
    return filename if filename.endswith(".npz") else filename + ".npz"


def build_table(time: np.ndarray, curves: list[tuple[str, np.ndarray]]):
    sources = [mg.SourceTable.from_arrays(
        label, ("total",), values, prefix=label, stream=mg.STREAM_OTHER,
        path=label + ".csv", time={"elapsed": time})
        for label, values in curves]
    return mg.merge_sources(sources, align="elapsed", grid="primary")


def candidate_window(time: np.ndarray, recorded: np.ndarray) -> tuple[int, int]:
    duration = min(12.0, max(4.0, float(time[-1] - time[0]) / 8.0))
    width = max(3, int(round(duration / float(np.median(np.diff(time))))))
    width = min(width, len(time))
    if width == len(time):
        return 0, len(time)
    start_floor = min(len(time) - width, int(np.searchsorted(time, time[0] + 3.0)))
    best_start, best_range = start_floor, -np.inf
    for start in range(start_floor, len(time) - width + 1, max(1, width // 10)):
        chunk = recorded[start:start + width]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) < 3:
            continue
        spread = float(np.percentile(finite, 95) - np.percentile(finite, 5))
        if spread > best_range:
            best_start, best_range = start, spread
    if not np.isfinite(best_range):
        raise ValueError("recorded K9 has no finite candidate window")
    return best_start, best_start + width


def render(path: Path, time: np.ndarray, curves: list[tuple[str, np.ndarray]],
           title: str, records: list[dict], kind: str) -> None:
    if len(curves) != 4 or any(np.count_nonzero(np.isfinite(y)) < 3 for _, y in curves):
        raise ValueError(f"{path.name}: expected four nonempty curves")
    table = build_table(time, curves)
    info = mg.plot_table(table, str(path), mode="overlay", series_by="file",
                         signal="sum", smooth_s=0.0, width_in=15.0,
                         panel_height_in=6.0, dpi=110, suptitle=title, check=True)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        deviation = ImageStat.Stat(image.convert("RGB")).stddev
        dimensions = list(image.size)
    check = info.get("check") or {}
    if (info["curves"] != 4 or check.get("empty") or info["bytes"] < 10_000
            or max(deviation) < 8.0):
        raise RuntimeError(f"{path.name}: figure failed curve/blankness check: {info}")
    records.append({"file": path.name, "kind": kind, "curves": info["curves"],
                    "bytes": info["bytes"], "dimensions": dimensions,
                    "rgb_stddev": deviation, "check": check})
    print(f"{path.name}: 4 curves, {info['bytes']} bytes, {dimensions[0]}x{dimensions[1]}")


def render_session(item: dict, records: list[dict]) -> None:
    filename = filename_for(item)
    arrays = arrays_from(DATA_DIR / filename, SESSION_FIELDS)
    if not all(key in item for key in ("device", "session", "loaded_at", "analysis_start")):
        raise ValueError(f"{filename}: metadata needs device/session/loaded_at/analysis_start")
    time = arrays["time"]
    analysis_start, loaded_at = float(item["analysis_start"]), float(item["loaded_at"])
    if not (np.isfinite(analysis_start) and np.isfinite(loaded_at)
            and time[0] <= loaded_at <= analysis_start < time[-1]):
        raise ValueError(f"{filename}: loading/analysis bounds must be seconds within time")
    start = int(np.searchsorted(time, analysis_start))
    if len(time) - start < 3:
        raise ValueError(f"{filename}: analysis window contains fewer than three rows")
    elapsed = time[start:] - analysis_start
    device, session = str(item["device"]), str(item["session"])
    label = f"{device} / {session}"
    stem = re.sub(r"[^\w\-]+", "_", Path(filename).stem)
    curves = [(name, arrays[key][start:]) for name, key in SERIES]
    render(FIG_DIR / f"{stem}_absolute.png", elapsed, curves,
           f"{label} | post-loading absolute ADC | recorded K9 / replay K9 / K12 v2 / tuned K9",
           records, "absolute")

    baseline_stop = min(len(elapsed), max(3, int(np.searchsorted(elapsed, 1.0))))
    deltas = [(name, values - float(np.nanmedian(values[:baseline_stop])))
              for name, values in curves]
    render(FIG_DIR / f"{stem}_drift.png", elapsed, deltas,
           f"{label} | drift relative to first 1 s after analysis_start (ADC)",
           records, "drift")
    if "10g" in session.lower() or "10g" in filename.lower():
        left, right = candidate_window(elapsed, arrays["recorded_k9"][start:])
        local = [(name, values[left:right]) for name, values in deltas]
        render(FIG_DIR / f"{stem}_candidate.png", elapsed[left:right], local,
               f"{label} | recorded-signal fluctuation candidate, NOT a confirmed 10g operation"
               f" | {elapsed[left]:.1f}-{elapsed[right - 1]:.1f} s",
               records, "unverified_candidate")


def render_synthetic(records: list[dict]) -> None:
    path = DATA_DIR / "synthetic.npz"
    if not path.exists():
        raise ValueError("synthetic.npz is missing")
    arrays = arrays_from(path, SYNTHETIC_FIELDS)
    time = arrays["time"]
    curves = [("01_reference", arrays["reference"]), ("02_step", arrays["step"]),
              ("03_K9 response", arrays["k9_response"]),
              ("04_K12 v2 response", arrays["k12_response"])]
    render(FIG_DIR / "synthetic_100adc_steps.png", time, curves,
           "Controlled dispersed 100 ADC steps | reference / step / K9 / K12 v2",
           records, "synthetic")


def main() -> int:
    summary_path = DATA_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = entries_from(summary)
    expected = {filename_for(item) for item in entries}
    actual = {path.name for path in DATA_DIR.glob("*.npz")} - {"synthetic.npz"}
    if expected != actual:
        raise ValueError(f"summary.json session mapping differs: missing={sorted(actual - expected)}, "
                         f"unknown={sorted(expected - actual)}")
    if len(expected) != len(entries):
        raise ValueError("summary.json repeats a session NPZ")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for item in entries:
        render_session(item, records)
    render_synthetic(records)
    (FIG_DIR / "plot_check.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(records)} figures; log: {FIG_DIR / 'plot_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
