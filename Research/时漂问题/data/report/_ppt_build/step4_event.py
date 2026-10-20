"""Fine-grained look at the ~61 s baseline lift in session 1 and event scan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import load_matrix, save_json  # noqa: E402

REPORT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\report\右手掌-new_param")
HERE = Path(__file__).parent

d = np.load(HERE / "dev1_curves.npz")
t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]

print("=== dev1 raw / algo at 0.5 s resolution, 0-100 s ===")
for x in np.arange(0, 100, 0.5):
    i = int(np.searchsorted(t, x))
    if i >= t.size:
        break
    print(f"{x:6.1f}s raw={raw[i]:9.1f} algo={algo[i]:9.1f} diff={raw[i]-algo[i]:8.1f}")

print("\n=== dev1 per-frame 58-72 s ===")
m = (t >= 58) & (t <= 72)
tt = t[m]
rr = raw[m]
aa = algo[m]
for i in range(0, tt.size, 20):
    print(f"{tt[i]:7.2f}s raw={rr[i]:9.1f} algo={aa[i]:9.1f}")
print(f"\nraw  58.0s {np.interp(58, t, raw):.0f} -> 62.0s {np.interp(62, t, raw):.0f} "
      f"-> 72s {np.interp(72, t, raw):.0f}")
print(f"algo 58.0s {np.interp(58, t, algo):.0f} -> 62.0s {np.interp(62, t, algo):.0f} "
      f"-> 72s {np.interp(72, t, algo):.0f}")

# event scan: rolling 1 s rise, both signals
def rises(t, y, win=1.0):
    out = []
    for start in np.arange(t[0], t[-1] - win, 0.25):
        a = np.interp(start, t, y)
        b = np.interp(start + win, t, y)
        out.append((start, b - a))
    return out

for name, y in (("raw", raw), ("algo", algo)):
    ev = sorted(rises(t, y), key=lambda x: -x[1])[:8]
    print(f"\ntop rises in 1 s ({name}):")
    for s, dv in ev:
        print(f"  t={s:7.2f}s  +{dv:7.1f} mN/s")

# also check the pre/algo ADC streams around 61 s to see raw device behaviour
pre_t, _f, pre_v, _m = load_matrix(REPORT / "设备1-2026-10-15_15-58-21-61s左右数据剧烈上抬" / "device_001_pre_seg0.csv")
alg_t, _f2, alg_v, _m2 = load_matrix(REPORT / "设备1-2026-10-15_15-58-21-61s左右数据剧烈上抬" / "device_001_algoadc_seg0.csv")
pre_s = pre_v.sum(axis=1)
alg_s = alg_v.sum(axis=1)
print("\n=== ADC domain sums (algorithm input vs output) ===")
for x in (10, 30, 58, 60, 61, 62, 65, 70, 100, 300, 700):
    i = int(np.searchsorted(pre_t, x))
    i = min(i, pre_t.size - 1)
    print(f"{x:6.0f}s pre_sum={pre_s[i]:9.0f} algoadc_sum={alg_s[i]:9.0f} diff={pre_s[i]-alg_s[i]:8.0f}")

np.savez_compressed(HERE / "dev1_adc.npz", t=pre_t, pre=pre_s, algoadc=alg_s)
save_json({"note": "see stdout"}, HERE / "event_scan.json")
