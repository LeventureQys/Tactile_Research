"""K9/K10 观测器重放内核（严格对照 src/domain/drift_v6/creep_observer.cpp）。

在原实现之上提供三个**可选**硬化参数，默认值全部等于原行为（需求文档 §3.6：
新增参数必须默认不改变现有行为）：

    idle_band_floor_adc : idle 判据的绝对地板（ADC）。原判据为
                          (y0 - y_floor) < idle_frac * (y_max - y_floor)，
                          恒载稳态下 span -> 0，判据退化为"单帧噪声下沿"。
                          设为 >0 时改用 (y0 - y_floor) < idle_frac * max(span, floor)。
    zero_track_gate     : 零点跟踪的受载门。原实现只要 idle 为真就让 zero 追 v；
                          设为 >0 时，若 (x_fast+x_slow) > gate * max(y_max,1) 则冻结
                          zero（即"补偿还在扛载荷"时不许零点吞载）。
    zero_rate_cap_adc_s : 零点更新速率上限（ADC/s），0 = 不限。
"""
import numpy as np

BASE = dict(
    r_fast=0.03, tau_c_fast_s=2.0, tau_r_fast_s=0.1,
    r_slow_max=0.6, tau_r_slow_s=150.0, slope_cap_frac=0.02, slope_gate_frac=0.05,
    tau_zero_s=8.0, idle_frac=0.05, y_max_tau_s=600.0, tau_slope_s=1.0,
    edge_slope_thres=60.0, edge_refract_s=2.0, edge_boost_s=2.0, tau_c_fast_boost_s=2.0,
    soft_unfreeze_s=2.0, hold_eps=2.0, hold_tau_s=0.5,
    slow_confirm_s=3.0, y_floor_tau_s=300.0, tau_r_slow_idle_s=0.2,
    bypass_release_frac=1.15, bypass_engage_frac=1.25, bypass_noise_sigma=3.0,
    bypass_base_tau_s=10.0, ramp_slope_min=0.5, ramp_full_s=4.0,
    guard_enable=0, guard_floor_confirm_s=0.5, guard_idle_confirm_s=0.3, guard_floor_deep=200.0,
    idle_band_floor_adc=0.0, zero_track_gate=0.0, zero_rate_cap_adc_s=0.0,
    floor_rise_gate=0, xslow_leak_tau_s=0.0,
)


def run(v_in, ts_abs, p=None, log=False, schedule=None):
    """逐帧跑观测器。p 覆盖 BASE 的键。返回补偿后数组（log=True 时附带状态轨迹）。

    schedule: [(绝对时间戳, {参数覆盖}), ...] —— 用于模拟"运行中下发参数"：
              时间戳到达后把覆盖合入参数集，状态不重置。
    """
    q = dict(BASE)
    if p:
        q.update(p)
    p = q
    sched = list(schedule) if schedule else []
    si = 0
    v = np.array(v_in, dtype=np.float64, copy=True)
    n_frames, n = v.shape

    x_fast = np.zeros(n); x_slow = np.zeros(n)
    v_lp = np.zeros(n); zero = np.zeros(n); y_max = np.zeros(n)
    t_edge = np.full(n, 1.0e6); applied = np.zeros(n); v_fast_lp = np.zeros(n)
    y_floor = np.zeros(n); load_dwell = np.zeros(n); ramp_dwell = np.zeros(n)
    below_el = np.zeros(n); idle_el = np.zeros(n)
    total_baseline = 0.0; total_noise = 0.0
    last_ts = 0.0
    first = True
    guard_on = p["guard_enable"] != 0
    band_floor = p["idle_band_floor_adc"]
    zgate = p["zero_track_gate"]
    zcap = p["zero_rate_cap_adc_s"]

    if log:
        K = ("applied", "x_slow", "x_fast", "y_floor", "y_max", "zero", "dwell", "idle", "t_edge", "slope")
        L = {k: np.zeros((n_frames, n), dtype=np.float32) for k in K}
        Lb = np.zeros(n_frames, dtype=bool)
        Ls = {k: np.zeros(n_frames) for k in ("total_baseline", "total_noise", "total")}

    for i in range(n_frames):
        if si < len(sched) and ts_abs[i] >= sched[si][0]:
            p.update(sched[si][1])
            guard_on = p["guard_enable"] != 0
            band_floor = p["idle_band_floor_adc"]
            zgate = p["zero_track_gate"]
            zcap = p["zero_rate_cap_adc_s"]
            si += 1
        if first:
            first = False
            last_ts = ts_abs[i]
            zero = v[i].copy(); y_max = np.zeros(n); v_lp = v[i].copy()
            v_fast_lp = v[i].copy(); y_floor = v[i].copy()
            total_baseline = float(v[i].sum())
            if log:
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
            # ① 零点跟踪 + 去趋势空载门
            y0 = vi - zero
            down = y0 < y_floor
            if guard_on:
                below_el = np.where(down, below_el + dt, below_el)
                deep = (y_floor - y0) > p["guard_floor_deep"]
                y_floor = np.where(down & (deep | (below_el >= p["guard_floor_confirm_s"])), y0, y_floor)
                below_el = np.where(down, below_el, 0.0)
            else:
                y_floor = np.where(down, y0, y_floor)
            y_max = np.maximum(y_max * np.exp(-dt / p["y_max_tau_s"]), np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            band = np.maximum(span, band_floor) if band_floor > 0.0 else span
            idle = (y0 - y_floor) < p["idle_frac"] * band
            # 上升分支：原实现无条件以 y_floor_tau_s 向 y0 爬；floor_rise_gate=1 时
            # 仅在被判为空载（近零带）时才允许上升，否则冻结（需求 §5.1：零点只在
            # 高置信度空载/近零状态下更新，受载时冻结）。
            rise = np.where(~down, (dt / p["y_floor_tau_s"]) * (y0 - y_floor), 0.0)
            if p["floor_rise_gate"]:
                rise = np.where(idle, rise, 0.0)
            y_floor = y_floor + rise
            load_dwell = np.where(idle, 0.0, load_dwell + dt)
            if zgate > 0.0:
                allow = (x_fast + x_slow) <= zgate * np.maximum(y_max, 1.0)
            else:
                allow = np.ones(n, dtype=bool)
            step = (dt / p["tau_zero_s"]) * (vi - zero)
            if zcap > 0.0:
                step = np.clip(step, -zcap * dt, zcap * dt)
            zero = np.where(idle & allow, zero + step, zero)

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
            band_k = np.maximum(span_k, band_floor) if band_floor > 0.0 else span_k
            idle_k = (y - y_floor) < p["idle_frac"] * band_k
            pos = e > 0.0
            base = np.maximum(e, 1.0)
            cap = p["slope_cap_frac"] * base
            soft_ok = (1.0 - np.exp(-t_edge / p["soft_unfreeze_s"])) > 0.5
            dwell_ok = load_dwell >= p["slow_confirm_s"]
            integ = pos & dwell_ok & soft_ok & (np.abs(slope) < p["slope_gate_frac"] * base)
            dx2 = np.clip(slope - dx1_rate, -cap, cap)
            # 可选：泄漏积分（物理蠕变会饱和，纯积分会无限增长；0=关闭，等于原实现）
            if p["xslow_leak_tau_s"] > 0.0:
                dx2 = dx2 - x_slow / p["xslow_leak_tau_s"]
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

            if log:
                L["applied"][i] = applied; L["x_slow"][i] = x_slow; L["x_fast"][i] = x_fast
                L["y_floor"][i] = y_floor; L["y_max"][i] = y_max; L["zero"][i] = zero
                L["dwell"][i] = load_dwell; L["idle"][i] = idle_k.astype(np.float32)
                L["t_edge"][i] = t_edge; L["slope"][i] = slope
                Ls["total_baseline"][i] = total_baseline
                Ls["total_noise"][i] = total_noise
                Ls["total"][i] = float(vi.sum())

        if bypass:
            applied = x_fast + x_slow
            if log:
                Lb[i] = True
            continue
        v[i] = vi - applied if p["hold_eps"] > 0.0 else vi - x_fast - x_slow
        if log and not (dt > 0.0):
            L["applied"][i] = applied; L["x_slow"][i] = x_slow; L["x_fast"][i] = x_fast
            L["y_floor"][i] = y_floor; L["y_max"][i] = y_max; L["zero"][i] = zero
            L["dwell"][i] = load_dwell

    if log:
        return v, L, Lb, Ls
    return v
