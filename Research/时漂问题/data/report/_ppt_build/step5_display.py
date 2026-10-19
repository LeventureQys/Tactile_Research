"""Cross-check the 61 s transient across display streams (seg000 / forcetotal / algoadc)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import load_matrix, load_stream, save_json  # noqa: E402

REPORT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\report\右手掌-new_param")
S1 = REPORT / "设备1-2026-10-15_15-58-21-61s左右数据剧烈上抬"
S2 = REPORT / "设备2-2026-10-15_16-31-43"
HERE = Path(__file__).parent

for tag, folder in (("dev1", S1), ("dev2", S2)):
    seg_t, _f, seg_v, _m = load_matrix(folder / "device_001_seg000.csv")
    seg_sum = seg_v.sum(axis=1)
    raw_t, _f2, raw_v, _m2 = load_matrix(folder / "device_001_raw_seg000.csv")
    raw_sum = raw_v.sum(axis=1)
    ft = load_stream(folder / "device_001_forcetotal_seg0.csv")
    t_ft = ft.iloc[:, 1].to_numpy(dtype=np.float64)
    f_ft = ft.iloc[:, 4].to_numpy(dtype=np.float64)
    n = min(seg_t.size, raw_t.size, t_ft.size)
    print(f"\n===== {tag} =====")
    print("streams:", seg_t.size, raw_t.size, t_ft.size)
    print(f"{'t':>7} {'raw_sum':>9} {'seg_sum':>9} {'forcetotal':>10}")
    for x in list(np.arange(0, 100, 1.0)) + list(np.arange(100, 705, 40.0)):
        i = int(np.searchsorted(t_ft, x))
        i = min(i, n - 1)
        print(f"{t_ft[i]:7.2f} {raw_sum[i]:9.0f} {seg_sum[i]:9.0f} {f_ft[i]:10.0f}")
    np.savez_compressed(HERE / f"{tag}_display.npz", t=t_ft, raw_adc_sum=raw_sum,
                        seg_sum=seg_sum, force_total=f_ft)
