"""K10 档专项：验证"下探回抬锚"能否修掉 y_floor 单向钉底。

现状（修复后）：
    audit 显示 gate=1 两档的 y_floor 末值 = 最小值（-758.9 / -654.2），即一旦被真实下探
    拉低就**永不回抬**（"跌后回抬" = +0.0）。抬升门把"吸载荷"堵死的同时，也堵死了自愈，
    于是 span 长期偏大、idle 判据变钝、空载期 zero 跟踪随之失效。

候选：
    floor_recover_tau_s —— 记录每次下探前的 y_floor 作为锚；回到锚以上时以该 τ 回抬到锚，
    不回到锚以上（真实卸载）则不动。τ=0 关闭（与现行为逐帧一致）。
    同时把首帧 y_floor 由 v 修正为 y0（= v − zero = 0），消除首帧伪下探。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Temp\拇指指腹-2026-10-19_16-33-49")
OUT = Path(__file__).resolve().parent / "out"


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
    return df.iloc[:, 0].to_numpy(dtype=np.float64), df.iloc[:, 3:].to_numpy(dtype=np.float64)


def run(v_in, ts_abs, p, recover_tau=0.0, fix_first_frame=True, log=False):
    q = dict(p)
    v = np.array(v_in, dtype=np.float64, copy=True)
    n_frames, n = v.shape
    z = lambda: np.zeros(n)
    x_fast, x_slow, v_lp, zero, y_max = z(), z(), z(), z(), z()
    t_edge, applied, v_fast_lp, y_floor = np.full(n, 1e6), z(), z(), z()
    load_dwell, ramp_dwell, below_el, idle_el = z(), z(), z(), z()
    anchor, in_drop = z(), np.zeros(n, dtype=bool)
    total_baseline = total_noise = 0.0
    last_ts, first = 0.0, True
    guard_on = q["guard_enable"] != 0
    L = {k: np.zeros((n_frames, n), dtype=np.float32) for k in ("applied", "x_slow", "y_floor", "idle")} if log else None

    for i in range(n_frames):
        if first:
            first = False
            last_ts = ts_abs[i]
            zero = v[i].copy(); v_lp = v[i].copy(); v_fast_lp = v[i].copy()
            y_floor = (v[i] - zero) if fix_first_frame else v[i].copy()
            total_baseline = float(v[i].sum())
            continue
        dt = ts_abs[i] - last_ts
        last_ts = ts_abs[i]
        dt = min(dt, 0.1) if dt > 0.0 else 0.0
        vi = v[i]
        bypass = False
        if dt > 0.0:
            total = float(vi.sum())
            if np.isfinite(total):
                alpha = min(dt / q["bypass_base_tau_s"], 1.0)
                release = max(q["bypass_release_frac"] * total_baseline,
                              total_baseline + q["bypass_noise_sigma"] * total_noise)
                engage = max(q["bypass_engage_frac"] * total_baseline,
                             total_baseline + (q["bypass_noise_sigma"] + 1.0) * total_noise)
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
            anchor = np.where(down & ~in_drop, y_floor, anchor)   # 下探起点 = 锚
            in_drop = down
            if guard_on:
                below_el = np.where(down, below_el + dt, below_el)
                deep = (y_floor - y0) > q["guard_floor_deep"]
                y_floor = np.where(down & (deep | (below_el >= q["guard_floor_confirm_s"])), y0, y_floor)
                below_el = np.where(down, below_el, 0.0)
            else:
                y_floor = np.where(down, y0, y_floor)
            y_max = np.maximum(y_max * np.exp(-dt / q["y_max_tau_s"]), np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            idle = (y0 - y_floor) < q["idle_frac"] * span
            rise = np.where(~down, (dt / q["y_floor_tau_s"]) * (y0 - y_floor), 0.0)
            if q["floor_rise_gate"]:
                rise = np.where(idle, rise, 0.0)
                if recover_tau > 0.0:
                    back_ok = (~down) & (~idle) & (y0 > anchor) & (anchor > y_floor)
                    rise = np.where(back_ok, (dt / recover_tau) * (anchor - y_floor), rise)
            y_floor = y_floor + rise
            load_dwell = np.where(idle, 0.0, load_dwell + dt)
            zero = np.where(idle, zero + (dt / q["tau_zero_s"]) * (vi - zero), zero)

            y = vi - zero
            slope = (vi - v_lp) / q["tau_slope_s"]
            e_now = np.maximum(y - x_fast - x_slow, 0.0)
            refract = max(q["edge_refract_s"], q["edge_boost_s"])
            t_edge = t_edge + dt
            fire = (slope > q["edge_slope_thres"]) & (t_edge > refract)
            t_edge = np.where(fire, 0.0, t_edge)
            ramp_on = (slope > q["ramp_slope_min"]) & (slope <= q["edge_slope_thres"])
            ramp_dwell = np.where(ramp_on, ramp_dwell + dt, 0.0)
            load_dwell = np.where(np.abs(slope) > q["edge_slope_thres"], 0.0, load_dwell)
            tc1 = np.where((q["edge_boost_s"] > 0.0) & (t_edge < q["edge_boost_s"]),
                           q["tau_c_fast_boost_s"], q["tau_c_fast_s"])
            if q["ramp_full_s"] > 0.0:
                w = np.minimum(ramp_dwell / q["ramp_full_s"], 1.0)
                tc1 = tc1 * (1.0 - w) + q["tau_c_fast_boost_s"] * w
            dx1 = np.where(e_now > 0.0, (q["r_fast"] * e_now - x_fast) / tc1, -x_fast / q["tau_r_fast_s"])
            x_fast = np.maximum(x_fast + dx1 * dt, 0.0)
            v_lp = v_lp + (dt / q["tau_slope_s"]) * (vi - v_lp)
            e = np.maximum(y - x_fast - x_slow, 0.0)
            span_k = np.maximum(y_max - y_floor, 1.0)
            idle_k = (y - y_floor) < q["idle_frac"] * span_k
            pos = e > 0.0
            base = np.maximum(e, 1.0)
            cap = q["slope_cap_frac"] * base
            soft_ok = (1.0 - np.exp(-t_edge / q["soft_unfreeze_s"])) > 0.5
            dwell_ok = load_dwell >= q["slow_confirm_s"]
            integ = pos & dwell_ok & soft_ok & (np.abs(slope) < q["slope_gate_frac"] * base)
            dx2 = np.clip(slope - dx1, -cap, cap)
            x_slow = np.where(integ, x_slow + dt * dx2, x_slow)
            hi = q["r_slow_max"] * base
            x_slow = np.where(pos, np.minimum(np.maximum(x_slow, 0.0), hi), x_slow)
            rel = (e <= 0.0) | idle_k
            if guard_on:
                idle_el = np.where(idle_k, idle_el + dt, 0.0)
                fast_ok = idle_k & (idle_el >= q["guard_idle_confirm_s"]) & (q["tau_r_slow_idle_s"] > 0.0)
                tau_r2 = np.where(fast_ok, q["tau_r_slow_idle_s"], q["tau_r_slow_s"])
            else:
                tau_r2 = np.where(idle_k & (q["tau_r_slow_idle_s"] > 0.0),
                                  q["tau_r_slow_idle_s"], q["tau_r_slow_s"])
            x_slow = np.where(rel, np.maximum(x_slow - (dt / tau_r2) * x_slow, 0.0), x_slow)
            if q["hold_eps"] > 0.0:
                slope_a = (vi - v_fast_lp) / q["hold_tau_s"]
                v_fast_lp = v_fast_lp + (dt / q["hold_tau_s"]) * (vi - v_fast_lp)
                d_des = x_fast + x_slow
                a_up = applied + dt * np.maximum(slope_a + q["hold_eps"], 0.0)
                a_dn = applied + dt * (slope_a - q["hold_eps"])
                a = np.where(d_des > applied, np.minimum(d_des, a_up), np.maximum(d_des, a_dn))
                applied = np.where(pos, np.clip(a, 0.0, np.maximum(d_des, 0.0)), d_des)
            if log:
                L["applied"][i] = applied; L["x_slow"][i] = x_slow
                L["y_floor"][i] = y_floor; L["idle"][i] = idle_k.astype(np.float32)
        if bypass:
            applied = x_fast + x_slow
            continue
        v[i] = vi - applied
    return (v, L) if log else v


import core  # noqa: E402

ts, pre = load_cells("device_001_pre_seg0.csv")
t = ts - ts[0]
mt = (t >= 1100) & (t <= 2850)
P = dict(core.BASE)

print(f"{'配置':52s} {'漂移':>9s} {'p2p':>8s} {'ap末':>8s} {'yfl末':>9s} {'yfl最小':>9s} {'idle%':>7s}")
for tag, gate, guard, tau, fixff in (
    ("基线 gate=0 guard=0（prod 现状）", 0, 0, 0.0, False),
    ("gate=1 guard=0（v3.4 档）", 1, 0, 0.0, False),
    ("gate=1 guard=1（K10 档）", 1, 1, 0.0, False),
    ("gate=1 guard=1 + 首帧修正", 1, 1, 0.0, True),
    ("gate=1 guard=1 + 回抬 τ=30", 1, 1, 30.0, True),
    ("gate=1 guard=1 + 回抬 τ=60", 1, 1, 60.0, True),
    ("gate=1 guard=0 + 回抬 τ=30", 1, 0, 30.0, True),
):
    o = dict(P); o["floor_rise_gate"] = gate; o["guard_enable"] = guard
    v, L = run(pre, ts, o, recover_tau=tau, fix_first_frame=fixff, log=True)
    s = v.sum(1); ap = (pre.sum(1) - s)
    yfl = L["y_floor"]
    print(f"{tag:52s} {s[mt][-1]-s[mt][0]:+9.1f} {s[mt].max()-s[mt].min():8.1f} {ap[mt][-1]:8.1f} "
          f"{yfl[mt][-1].sum():9.1f} {yfl[mt].min(0).sum():9.1f} {(L['idle']>0.5).mean()*100:6.2f}%")
