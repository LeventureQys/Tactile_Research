"""Shared loader for tactile recording CSV exports (##Session header + ##Data block)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "gbk")


def sniff_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc) as fh:
                fh.read(4096)
            return enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"no usable encoding for {path}")


def read_header(path: Path):
    """Return (meta dict, data_line_index, encoding)."""
    enc = sniff_encoding(path)
    meta: dict[str, str] = {}
    header_idx = None
    with path.open("r", encoding=enc) as fh:
        for i, line in enumerate(fh):
            s = line.rstrip("\n").rstrip("\r")
            if s.startswith("##Data"):
                header_idx = i
                break
            if s.startswith("##") or not s.strip():
                continue
            if "," in s:
                k, _, v = s.partition(",")
                meta.setdefault(k.strip(), v.strip())
    if header_idx is None:
        raise RuntimeError(f"no ##Data block in {path}")
    return meta, header_idx, enc


def load_stream(path: Path, usecols=None) -> pd.DataFrame:
    meta, header_idx, enc = read_header(path)
    df = pd.read_csv(
        path,
        skiprows=header_idx + 1,
        encoding=enc,
        usecols=usecols,
        engine="c",
    )
    df.attrs["meta"] = meta
    return df


def load_matrix(path: Path):
    """Return (elapsed_s, frame_index, values[N,F], meta) for cell-indexed streams."""
    meta, header_idx, enc = read_header(path)
    arr = pd.read_csv(path, skiprows=header_idx + 1, encoding=enc).to_numpy(dtype=np.float64)
    elapsed = arr[:, 1]
    frame = arr[:, 2]
    values = arr[:, 3:]
    return elapsed, frame, values, meta


def summarize(elapsed: np.ndarray) -> dict:
    d = np.diff(elapsed)
    return {
        "frames": int(elapsed.size),
        "duration_s": float(elapsed[-1] - elapsed[0]) if elapsed.size else 0.0,
        "mean_dt_ms": float(np.mean(d) * 1000) if d.size else 0.0,
        "fps": float(1.0 / np.mean(d)) if d.size else 0.0,
    }


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


def block_stats(t: np.ndarray, y: np.ndarray, t0: float, t1: float) -> dict:
    m = (t >= t0) & (t <= t1)
    if not m.any():
        return {"n": 0}
    seg = y[m]
    return {
        "n": int(seg.size),
        "t0": float(t0),
        "t1": float(t1),
        "mean": float(np.mean(seg)),
        "median": float(np.median(seg)),
        "std": float(np.std(seg)),
        "min": float(np.min(seg)),
        "max": float(np.max(seg)),
        "first_val": float(seg[0]),
        "last_val": float(seg[-1]),
        "delta": float(seg[-1] - seg[0]),
        "range": float(np.max(seg) - np.min(seg)),
    }
