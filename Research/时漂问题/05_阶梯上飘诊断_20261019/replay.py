"""K9/K10 逐帧重放器（严格对着 src/domain/drift_v6/creep_observer.cpp）。

用法:
    python replay.py            # 重放并校验
输入: D:\\Temp\\拇指指腹-2026-10-19_16-33-49\\device_001_pre_seg0.csv
输出: out/replay.npz（重放结果与逐帧状态轨迹）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# session.json 记录的实际下发参数（plan-v4 observer-k9 + 现场 r_slow_max=0.6）
PARAMS = dict(
    r_fast=0.03, tau_c_fast_s=2.0, tau_r_fast_s=0.1,
    r_slow_max=0.6, tau_r_slow_s=150.0, slope_cap_frac=0.02, slope_gate_frac=0.05,
    tau_zero_s=8.0, idle_frac=0.05, y_max_tau_s=600.0, tau_slope_s=1.0,
    edge_slope_thres=60.0, edge_refract_s=2.0, edge_boost_s=2.0, tau_c_fast_boost_s=2.0,
    soft_unfreeze_s=2.0, hold_eps=2.0, hold_tau_s=0.5,
    slow_confirm_s=3.0, y_floor_tau_s=300.0, tau_r_slow_idle_s=0.2,
    bypass_release_frac=1.15, bypass_engage_frac=1.25, bypass_noise_sigma=3.0,
    bypass_base_tau_s=10.0, ramp_slope_min=0.5, ramp_full_s=4.0,
    guard_enable=0, guard_floor_confirm_s=0.5, guard_idle_confirm_s=0.3, guard_floor_deep=200.0,
)


def load_cells(name):
    path = ROOT / name
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.startswith("##Data"):
                break
        skip += 1
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c")
    ts = df.iloc[:, 0].to_numpy(dtype=np.float64)
    cells = df.iloc[:, 3:].to_numpy(dtype=np.float64)
    return ts, cells


def run(v_in, ts_abs, p, log_states=True):
    n_frames, n = v_in.shape
    v = np.array(v_in, dtype=np.float64, copy=True)

    x_fast = np.zeros(n); x_slow = np.zeros(n)
    v_lp = np.zeros(n); zero = np.zeros(n); y_max = np.zeros(n)
    t_edge = np.full(n, 1.0e6); applied = np.zeros(n); v_fast_lp = np.zeros(n)
    y_floor = np.zeros(n); load_dwell = np.zeros(n); ramp_dwell = np.zeros(n)
    below_el = np.zeros(n); idle_el = np.zeros(n)
    total_baseline = 0.0; total_noise = 0.0
    last_ts = 0.0
    first = True

    if log_states:
        L = {k: np.zeros((n_frames, n), dtype=np.float32) for k in
             ("applied", "x_slow", "x_fast", "y_floor", "y_max", "zero", "dwell", "idle", "t_edge", "slope")}
        Lb = np.zeros(n_frames, dtype=bool)
        Ls = {k: np.zeros(n_frames, dtype=np.float64) for k in
              ("total_baseline", "total_noise", "total", "applied_sum")}

    guard_on = p["guard_enable"] != 0
    for i in range(n_frames):
        if first:
            first = False
            last_ts = ts_abs[i]
            zero = v[i].copy(); y_max = np.zeros(n); v_lp = v[i].copy()
            v_fast_lp = v[i].copy(); y_floor = v[i].copy()
            total_baseline = float(v[i].sum())
            if log_states:
                Ls["total_baseline"][i] = total_baseline
                Ls["total"][i] = float(v[i].sum())
            continue

        dt = ts_abs[i] - last_ts
        last_ts = ts_abs[i]
        dt = min(dt, 0.1) if dt > 0.0 else 0.0
        vi = v[i]
        bypass = False
        if dt > 0.0:
            total = float(vi.sum())
            if np.isfinite(total):
                alpha = min(dt / p["bypass_base_tau_s"], 1.0)
                release = max(p["bypass_release_frac"] * total_baseline,
                              total_baseline + p["bypass_noise_sigma"] * total_noise)
                engage = max(p["bypass_engage_frac"] * total_baseline,
                             total_baseline + (p["bypass_noise_sigma"] + 1.0) * total_noise)
                if total < engage:
                    total_baseline += alpha * (total - total_baseline)
                    total_noise += alpha * (abs(total - total_baseline) - total_noise)
                    if total_noise < 0.0:
                        total_noise = 0.0
                if total < release:
                    bypass = True

        if dt > 0.0:
            y0 = vi - zero
            down = y0 < y_floor
            if guard_on:
                below_el = np.where(down, below_el + dt, below_el)
                deep = (y_floor - y0) > p["guard_floor_deep"]
                y_floor = np.where(down & (deep | (below_el >= p["guard_floor_confirm_s"])), y0, y_floor)
                below_el = np.where(down, below_el, 0.0)
            else:
                y_floor = np.where(down, y0, y_floor)
            up = ~down
            y_floor = np.where(up, y_floor + (dt / p["y_floor_tau_s"]) * (y0 - y_floor), y_floor)
            y_max = np.maximum(y_max * np.exp(-dt / p["y_max_tau_s"]), np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            idle = (y0 - y_floor) < p["idle_frac"] * span
            load_dwell = np.where(idle, 0.0, load_dwell + dt)
            zero = np.where(idle, zero + (dt / p["tau_zero_s"]) * (vi - zero), zero)

            y = vi - zero
            slope = (vi - v_lp) / p["tau_slope_s"]
            e_now = np.maximum(y - x_fast - x_slow, 0.0)
            refract = max(p["edge_refract_s"], p["edge_boost_s"])

            t_edge = t_edge + dt
            fire = (slope > p["edge_slope_thres"]) & (t_edge > refract)
            t_edge = np.where(fire, 0.0, t_edge)
            ramp_on = (slope > p["ramp_slope_min"]) & (slope <= p["edge_slope_thres"])
            ramp_dwell = np.where(ramp_on, ramp_dwell + dt, 0.0)
            load_dwell = np.where(np.abs(slope) > p["edge_slope_thres"], 0.0, load_dwell)

            tc1 = np.where((p["edge_boost_s"] > 0.0) & (t_edge < p["edge_boost_s"]),
                           p["tau_c_fast_boost_s"], p["tau_c_fast_s"])
            if p["ramp_full_s"] > 0.0:
                w = np.minimum(ramp_dwell / p["ramp_full_s"], 1.0)
                tc1 = tc1 * (1.0 - w) + p["tau_c_fast_boost_s"] * w

            dx1_rate = np.where(e_now > 0.0,
                                (p["r_fast"] * e_now - x_fast) / tc1,
                                -x_fast / p["tau_r_fast_s"])
            x_fast = np.maximum(x_fast + dx1_rate * dt, 0.0)
            v_lp = v_lp + (dt / p["tau_slope_s"]) * (vi - v_lp)

            e = np.maximum(y - x_fast - x_slow, 0.0)
            span_k = np.maximum(y_max - y_floor, 1.0)
            idle_k = (y - y_floor) < p["idle_frac"] * span_k
            pos = e > 0.0
            base = np.maximum(e, 1.0)
            cap = p["slope_cap_frac"] * base
            soft_ok = (1.0 - np.exp(-t_edge / p["soft_unfreeze_s"])) > 0.5
            dwell_ok = load_dwell >= p["slow_confirm_s"]
            integ = pos & dwell_ok & soft_ok & (np.abs(slope) < p["slope_gate_frac"] * base)
            dx2 = np.clip(slope - dx1_rate, -cap, cap)
            x_slow = np.where(integ, x_slow + dt * dx2, x_slow)
            hi = p["r_slow_max"] * base
            x_slow = np.where(pos, np.minimum(np.maximum(x_slow, 0.0), hi), x_slow)

            rel = (e <= 0.0) | idle_k
            if guard_on:
                idle_el = np.where(idle_k, idle_el + dt, 0.0)
                fast_ok = idle_k & (idle_el >= p["guard_idle_confirm_s"]) & (p["tau_r_slow_idle_s"] > 0.0)
                tau_r2 = np.where(fast_ok, p["tau_r_slow_idle_s"], p["tau_r_slow_s"])
            else:
                tau_r2 = np.where(idle_k & (p["tau_r_slow_idle_s"] > 0.0),
                                  p["tau_r_slow_idle_s"], p["tau_r_slow_s"])
            x_slow = np.where(rel, np.maximum(x_slow - (dt / tau_r2) * x_slow, 0.0), x_slow)

            if p["hold_eps"] > 0.0:
                slope_a = (vi - v_fast_lp) / p["hold_tau_s"]
                v_fast_lp = v_fast_lp + (dt / p["hold_tau_s"]) * (vi - v_fast_lp)
                d_des = x_fast + x_slow
                a_up = applied + dt * np.maximum(slope_a + p["hold_eps"], 0.0)
                a_dn = applied + dt * (slope_a - p["hold_eps"])
                a = np.where(d_des > applied, np.minimum(d_des, a_up), np.maximum(d_des, a_dn))
                a = np.where(pos, np.clip(a, 0.0, np.maximum(d_des, 0.0)), d_des)
                applied = a

            if log_states:
                L["applied"][i] = applied; L["x_slow"][i] = x_slow; L["x_fast"][i] = x_fast
                L["y_floor"][i] = y_floor; L["y_max"][i] = y_max; L["zero"][i] = zero
                L["dwell"][i] = load_dwell; L["idle"][i] = idle_k.astype(np.float32)
                L["t_edge"][i] = t_edge; L["slope"][i] = slope
                Ls["total_baseline"][i] = total_baseline
                Ls["total_noise"][i] = total_noise
                Ls["total"][i] = float(vi.sum())

        if bypass:
            applied = x_fast + x_slow
            if log_states:
                Lb[i] = True
                Ls["applied_sum"][i] = float(applied.sum())
            continue
        out = vi - applied if p["hold_eps"] > 0.0 else vi - x_fast - x_slow
        v[i] = out
        if log_states:
            Ls["applied_sum"][i] = float(applied.sum())
            if not (dt > 0.0):
                L["applied"][i] = applied; L["x_slow"][i] = x_slow; L["x_fast"][i] = x_fast
                L["y_floor"][i] = y_floor; L["y_max"][i] = y_max; L["zero"][i] = zero
                L["dwell"][i] = load_dwell

    if log_states:
        return v, L, Lb, Ls
    return v


def main():
    ts_abs, pre = load_cells("device_001_pre_seg0.csv")
    _, algoadc = load_cells("device_001_algoadc_seg0.csv")
    t = ts_abs - ts_abs[0]

    v_k9, L, Lb, Ls = run(pre, ts_abs, PARAMS)
    err = v_k9 - algoadc
    print(f"K9 重放 vs algoadc: max|err|={np.abs(err).max():.4f}  "
          f"mean|err|={np.abs(err).mean():.6f}  frames={len(t)}")
    bad = np.argmax(np.abs(err).sum(1))
    print(f"  最差帧 #{bad} t={t[bad]:.3f}s  err_sum={err[bad].sum():.3f}")
    mid = t < 2850.0  # 排除末尾算法切档导致的重置
    err_mid = err[mid]
    print(f"  排除末尾切档（t<2850s）: max|err|={np.abs(err_mid).max():.5f}  "
          f"mean|err|={np.abs(err_mid).mean():.6f}")

    # 逐通道 applied 在 18~33min 的塌陷
    m = (t >= 1080) & (t < 1980)
    ap = L["applied"]
    d = ap[m][-1] - ap[m][0]
    order = np.argsort(d)[:10]
    print("\n18~33min applied 塌陷最多的通道:")
    for k in order:
        print(f"  cell{k:2d}  applied {ap[m][0][k]:8.2f} -> {ap[m][-1][k]:8.2f} (delta {d[k]:+8.2f})  "
              f"x_slow {L['x_slow'][m][0][k]:8.2f} -> {L['x_slow'][m][-1][k]:8.2f}  "
              f"y_floor {L['y_floor'][m][0][k]:8.2f} -> {L['y_floor'][m][-1][k]:8.2f}")

    print(f"\nbypass 帧数={int(Lb.sum())} 占比={Lb.mean()*100:.2f}%")

    # 同一数据跑 K10（guard_enable=1）
    p10 = dict(PARAMS); p10["guard_enable"] = 1
    v_k10, L10, Lb10, Ls10 = run(pre, ts_abs, p10)
    e10 = v_k10 - algoadc
    print(f"\nK10(guard=1) 重放 vs algoadc: mean|err|={np.abs(e10).mean():.6f}  "
          f"max|err|(t<2850)={np.abs(e10[mid]).max():.4f}")
    print(f"K10 vs K9 输出差: mean={np.mean(v_k10 - v_k9):.4f}  "
          f"max|diff|={np.abs(v_k10 - v_k9).max():.2f}")

    keep = [6, 44, 17, 2, 7, 12, 14, 1, 21, 11]
    np.savez_compressed(
        OUT / "replay.npz", t=t,
        pre_sum=pre.sum(1), algoadc_sum=algoadc.sum(1),
        k9_sum=v_k9.sum(1), k10_sum=v_k10.sum(1), bypass=Lb, bypass10=Lb10,
        k9_applied_sum=L["applied"].sum(1), k10_applied_sum=L10["applied"].sum(1),
        k9_xslow_sum=L["x_slow"].sum(1), k10_xslow_sum=L10["x_slow"].sum(1),
        k9_xfast_sum=L["x_fast"].sum(1),
        k9_yfloor_sum=L["y_floor"].sum(1), k10_yfloor_sum=L10["y_floor"].sum(1),
        k9_idle_count=L["idle"].sum(1), k10_idle_count=L10["idle"].sum(1),
        k9_dwell_max=L["dwell"].max(1),
        k9_total_baseline=Ls["total_baseline"], k9_total=Ls["total"],
        **{f"k9_{k}_c{c}": L[k][:, c] for k in ("applied", "x_slow", "y_floor", "y_max", "zero", "slope", "dwell") for c in keep},
        **{f"k10_{k}_c{c}": L10[k][:, c] for k in ("applied", "x_slow", "y_floor", "zero") for c in keep},
        **{f"pre_c{c}": pre[:, c] for c in keep},
    )
    print(f"轨迹已存 {OUT / 'replay.npz'}")


if __name__ == "__main__":
    main()
