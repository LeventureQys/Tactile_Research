# -*- coding: utf-8 -*-
"""K10 实现一致性校验（parity）：产品实装 vs 离线 guarded_k9。

素材：回收站中的 `右手掌-slope_cap_frac0.009/设备1-2026-10-15_15-43-38`
      （algorithm.id = drift_k10_creep_observer，guard_enable = 1，137 帧，ADC 显示）
      device_001_pre_seg0.csv = 算法输入；device_001_seg000.csv = 算法输出（设备实算）。

口径：把 pre 作为输入喂给离线 guarded_k9.compensate（参数与 guard 全部取自该会话 session.json），
      与 seg 逐帧逐通道比对。两者等价 ⇒ 离线调参结论可直接映射到设备。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from guarded_k9 import compensate  # noqa: E402

FOLDER = Path(r"D:\$RECYCLE.BIN\S-1-5-21-3129820652-4156655561-3743858824-1001"
              r"\$R3DWW9D.009\设备1-2026-10-15_15-43-38")
OUT = HERE / "palm_tune_20261015"
OUT.mkdir(parents=True, exist_ok=True)


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


def main() -> None:
    meta = json.loads((FOLDER / "session.json").read_text(encoding="utf-8"))
    dev = meta["devices"][0]
    params = dict((meta.get("algorithm") or {}).get("params") or {})
    fps = float(dev["observed_fps"])

    pre = load_stream(FOLDER, dev["algorithm_input_file"])
    seg = load_stream(FOLDER, dev["algorithm_result_file"])
    print(f"pre 列: {list(pre.columns[:5])} ... 共 {pre.shape[1]} 列")
    print(f"pre {pre.shape} | seg {seg.shape} | fps {fps:.4f}")

    n = min(pre.shape[0], seg.shape[0])
    x_in = np.asarray(pre.iloc[:n, 3:], dtype=float)
    x_dev = np.asarray(seg.iloc[:n, 3:], dtype=float)
    t = np.arange(n, dtype=float) / fps

    guard = dict(enabled=bool(int(params.get("guard_enable", 0))),
                 floor_confirm_s=float(params.get("guard_floor_confirm_s", 0.5)),
                 idle_confirm_s=float(params.get("guard_idle_confirm_s", 0.3)),
                 floor_deep_adc=float(params.get("guard_floor_deep", 200.0)),
                 floor_guard=True, idle_guard=True, spike_guard=False)
    out, info = compensate(x_in.copy(), t, params, guard)

    diff = np.abs(out - x_dev)
    total_diff = np.abs(out.sum(1) - x_dev.sum(1))
    report = dict(source=str(FOLDER), frames=int(n), channels=int(x_in.shape[1]),
                  fps=fps, algorithm_id=meta["algorithm"]["id"],
                  param_set=params.get("param_set"), guard=guard,
                  max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                  max_abs_total_diff=float(total_diff.max()),
                  equivalent=bool(diff.max() <= 1e-6), guard_info=info)
    (OUT / "parity_check_k10_live.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "guard"}, ensure_ascii=False, indent=1))
    print("前 12 帧总值  pre → 离线 → 设备:")
    for i in range(min(12, n)):
        print(f"   t={t[i]:.3f}s  {x_in[i].sum():9.1f} → {out[i].sum():9.1f} → {x_dev[i].sum():9.1f}"
              f"  |Δ|={total_diff[i]:.3e}")
    worst = int(np.argmax(diff))
    print(f"最大逐通道差位置: 帧 {worst} 通道 {worst % x_in.shape[1]}  差 {diff.max():.6f}")


if __name__ == "__main__":
    main()
