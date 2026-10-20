"""Deep numeric read-out of both sessions: phases, drift rates, transient events."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_lib import save_json  # noqa: E402

HERE = Path(__file__).parent
curves = {k: np.load(HERE / f"{k}_curves.npz") for k in ("dev1", "dev2")}

report: dict = {}
for key, d in curves.items():
    t, raw, algo = d["t"], d["raw_sum"], d["algo_sum"]
    print(f"\n========== {key} ==========")
    line = []
    # 1 s downsample
    secs = np.arange(0, int(t[-1]) + 1, 1.0)
    raw_s = np.interp(secs, t, raw)
    algo_s = np.interp(secs, t, algo)
    print(f"duration {t[-1]:.1f}s  frames {t.size}")
    print("t(s)  raw    algo   diff")
    for i in range(0, len(secs), 5):
        print(f"{secs[i]:6.0f} {raw_s[i]:8.0f} {algo_s[i]:8.0f} {raw_s[i] - algo_s[i]:8.0f}")
    # event scan: biggest 3 s rise in raw over the whole session
    win = 30  # frames ~0.3 s
    print("\ntop 3 s rises (raw):")
    order = np.argsort(np.diff(raw_s, 3))[::-1][:6]
    for i in sorted(order):
        print(f"  t={secs[i]:6.0f}s  raw {raw_s[i]:8.0f} -> {raw_s[i+3]:8.0f} (+{raw_s[i+3]-raw_s[i]:.0f}) "
              f"algo {algo_s[i]:8.0f} -> {algo_s[i+3]:8.0f} (+{algo_s[i+3]-algo_s[i]:.0f})")
    # load phase: first time raw exceeds 50% of its median
    med = np.median(raw_s[len(raw_s) // 4:])
    idx = np.argmax(raw_s > 0.5 * med)
    print(f"\nfirst crossing 50% of median ({0.5*med:.0f} mN) at t={secs[idx]:.1f}s")
    # plateau after load
    t_load = secs[idx] + 10
    m = (t >= t_load) & (t <= t[-1])
    A = np.vstack([t[m] - t_load, np.ones(m.sum())]).T
    kr, br = np.linalg.lstsq(A, raw[m], rcond=None)[0]
    ka, ba = np.linalg.lstsq(A, algo[m], rcond=None)[0]
    print(f"plateau {t_load:.0f}-{t[-1]:.0f}s  raw slope {kr*60:+.2f} mN/min (total {kr*(t[-1]-t_load):+.0f} mN)")
    print(f"                              algo slope {ka*60:+.2f} mN/min (total {ka*(t[-1]-t_load):+.0f} mN)")
    print(f"raw std {np.std(raw[m]):.1f}  algo std {np.std(algo[m]):.1f}  "
          f"raw mean {np.mean(raw[m]):.0f}  algo mean {np.mean(algo[m]):.0f}")
    report[key] = {
        "duration_s": float(t[-1]),
        "frames": int(t.size),
        "load_t_s": float(secs[idx]),
        "plateau": {
            "t0": float(t_load), "t1": float(t[-1]),
            "raw_slope_mN_per_min": float(kr * 60),
            "algo_slope_mN_per_min": float(ka * 60),
            "raw_total_mN": float(kr * (t[-1] - t_load)),
            "algo_total_mN": float(ka * (t[-1] - t_load)),
            "raw_std": float(np.std(raw[m])),
            "algo_std": float(np.std(algo[m])),
            "raw_mean": float(np.mean(raw[m])),
            "algo_mean": float(np.mean(algo[m])),
        },
    }

save_json(report, HERE / "deep_read.json")
print("\nsaved deep_read.json")
