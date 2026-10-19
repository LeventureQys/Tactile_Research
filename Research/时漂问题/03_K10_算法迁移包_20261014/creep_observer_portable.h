/* =============================================================================
 * creep_observer_portable.h
 *
 * 在线双态蠕变观测器（K9 / drift_v34 creep_observer）—— 单文件可移植实现
 *
 * 本文件是**唯一权威算法文件**。上位机（C/C++/C#/Python）、嵌入式（裸机/RTOS）
 * 都从这一份文件派生，不再各自维护算法副本。
 *
 * 用法（stb 风格单文件）：
 *     在恰好一个 .c/.cpp 里：
 *         #define DV6_IMPLEMENTATION
 *         #include "creep_observer_portable.h"
 *     其它文件只 #include 头部分（不要定义 DV6_IMPLEMENTATION）。
 *
 * 特性：
 *     - 纯 C99，无外部依赖（仅 <math.h> <string.h> <stddef.h>）
 *     - 无动态内存：状态缓冲区由调用方提供，可在栈/静态区分配
 *     - 无 STL / Eigen / 异常 / RTTI
 *     - 就地补偿：dv6_process() 直接把输入数组改写为补偿后显示值
 *
 * 语义权威：与 src/domain/drift_v6/creep_observer.cpp 逐帧等价。
 *           等价性由 golden_vectors/ 的逐帧向量核验（见迁移文档第 8 节）。
 *
 * 版本：portable-1.0（对应固定版 K9 + 可选瞬态防护研究层）
 * ========================================================================== */

#ifndef DV6_CREEP_OBSERVER_PORTABLE_H
#define DV6_CREEP_OBSERVER_PORTABLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 参数（27 项）───────────────────────────────────────────────────────────
 * 单位：时间 s，斜率 ADC/s，比值无量纲。ADC 域标定值不可直接用于力值域。
 * ────────────────────────────────────────────────────────────────────────── */
typedef struct dv6_params {
    /* 快态 x1 */
    double r_fast;                /* 快态幅度比：终值 = r_fast * e              */
    double tau_c_fast_s;          /* 快态受载收敛 τ                             */
    double tau_r_fast_s;          /* 快态空载恢复 τ                             */
    /* 慢态 x2 */
    double r_slow_max;            /* 慢态幅度上限：x2 ≤ r_slow_max * max(e,1)   */
    double tau_r_slow_s;          /* 慢态常规泄放 τ                             */
    double slope_cap_frac;        /* 慢态积分速率上限：|dx2| ≤ 该值 * max(e,1)  */
    double slope_gate_frac;       /* 慢态积分斜率门：|slope| < 该值 * max(e,1)  */
    /* 零点与包络 */
    double tau_zero_s;            /* 零点跟踪 τ（仅近零带内生效）               */
    double idle_frac;             /* 近零带比例（去趋势口径）                   */
    double y_max_tau_s;           /* 量程包络衰减 τ                             */
    double tau_slope_s;           /* 低通导数 τ（slope 估计）                   */
    /* 沿检测与前馈 */
    double edge_slope_thres;      /* 上升沿判定阈值（逐通道 slope）             */
    double edge_refract_s;        /* 沿触发最小间隔                             */
    double edge_boost_s;          /* 沿后前馈窗                                 */
    double tau_c_fast_boost_s;    /* 前馈窗内快态 τ                             */
    double soft_unfreeze_s;       /* 沿后慢态软冻结窗                           */
    /* 预留池 K6 */
    double hold_eps;              /* 限额余量（ADC/s）；≤0 关闭预留池           */
    double hold_tau_s;            /* 限额用快速低通 τ                           */
    /* 受载确认 K7 */
    double slow_confirm_s;        /* 慢态积分需连续受载时长                      */
    double y_floor_tau_s;         /* 去趋势基线包络爬升 τ                        */
    double tau_r_slow_idle_s;     /* 空载期快泄放 τ；≤0 回落 tau_r_slow_s        */
    /* 全局空载旁路 K7 */
    double bypass_release_frac;   /* 直通释放比                                  */
    double bypass_engage_frac;    /* 直通恢复比（迟滞）                          */
    double bypass_noise_sigma;    /* 门限噪声带宽（σ）                           */
    double bypass_base_tau_s;     /* 空载总值基线跟踪 τ                          */
    /* 缓坡前馈 K9 */
    double ramp_slope_min;        /* 缓坡斜率下限                                */
    double ramp_full_s;           /* 缓坡满窗时长；≤0 关闭缓坡层                 */
    /* ── 可选瞬态防护层（默认关闭；开启后不影响稳态指标）── */
    int    guard_enable;          /* 0=关闭（与原实现逐帧一致）；1=开启          */
    double guard_floor_confirm_s; /* y_floor 下行跟随需持续确认的时长            */
    double guard_idle_confirm_s;  /* 快速泄放需 idle 持续确认的时长              */
    double guard_floor_deep;      /* 深跌破该值立即跟随（真实卸载不延迟）        */
} dv6_params;

/* 观测器实例。所有数组指针指向调用方提供的内存（见 dv6_buffer_bytes）。 */
typedef struct dv6_observer {
    int    n;               /* 通道数 */
    int    first_frame;     /* 1=尚未初始化首帧 */
    double last_ts;         /* 上一帧时间戳（秒） */

    /* 全局空载旁路状态（标量） */
    double total_baseline;
    double total_noise;

    dv6_params p;

    /* 逐通道状态（各 n 个 double） */
    double *x_fast;         /* 快态补偿量 */
    double *x_slow;         /* 慢态补偿量 */
    double *v_lp;           /* slope 估计用低通 */
    double *zero;           /* 零点 */
    double *y_max;          /* 量程包络 */
    double *t_edge;         /* 距上次上升沿的时长 */
    double *applied;        /* 实际施加补偿（预留池） */
    double *v_fast_lp;      /* 预留池快速低通 */
    double *y_floor;        /* 去趋势基线包络 */
    double *load_dwell;     /* 连续受载计时 */
    double *ramp_dwell;     /* 缓坡计时 */
    double *below_elapsed;  /* 防护：y_floor 跌破计时 */
    double *idle_elapsed;   /* 防护：idle 持续计时 */

    /* 工作区（各 n 个 double，调用方无需关心内容） */
    double *scratch_y;
    double *scratch_slope;
    double *scratch_e;
} dv6_observer;

/* ── API ─────────────────────────────────────────────────────────────────── */

/* 填入「工作树默认」参数（代码默认值）。 */
void dv6_params_default(dv6_params *p);

/* 填入 plan-v4 observer-k9 现场参数集（设备录制实际下发值）。 */
void dv6_params_plan_v4_observer_k9(dv6_params *p);

/* 所需缓冲区字节数（调用方用 static/栈/堆分配这么多字节即可）。 */
size_t dv6_buffer_bytes(int n_channels);

/* 初始化实例。buffer 需至少 dv6_buffer_bytes(n) 字节且已对齐到 double。
 * 返回 0 成功，非 0 失败（n<=0 或 buffer 为空）。 */
int dv6_init(dv6_observer *o, int n_channels, const dv6_params *p,
             void *buffer, size_t bytes);

/* 重置为「未初始化首帧」状态（缓冲区内容无需清理）。 */
void dv6_reset(dv6_observer *o);

/* 运行一帧。v 指向长度 n 的数组，**就地**改写为补偿后显示值。
 * 时间戳单调递增（秒）；内部按 dt = min(dt, 0.1) 截断，dt<=0 视为 0。 */
void dv6_process(dv6_observer *o, double timestamp_s, double *v);

/* 诊断读取：当前实际施加的补偿总量（= Σ applied）。 */
double dv6_applied_total(const dv6_observer *o);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* DV6_CREEP_OBSERVER_PORTABLE_H */

/* =============================================================================
 *                              实 现 部 分
 * ========================================================================== */

#ifdef DV6_IMPLEMENTATION

#include <math.h>
#include <string.h>

#ifndef DV6_MIN
#define DV6_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef DV6_MAX
#define DV6_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

void dv6_params_default(dv6_params *p)
{
    if (!p) return;
    p->r_fast = 0.01;
    p->tau_c_fast_s = 40.0;
    p->tau_r_fast_s = 6.0;
    p->r_slow_max = 0.2;
    p->tau_r_slow_s = 150.0;
    p->slope_cap_frac = 0.05;
    p->slope_gate_frac = 0.05;
    p->tau_zero_s = 8.0;
    p->idle_frac = 0.05;
    p->y_max_tau_s = 600.0;
    p->tau_slope_s = 1.0;
    p->edge_slope_thres = 60.0;
    p->edge_refract_s = 2.0;
    p->edge_boost_s = 2.0;
    p->tau_c_fast_boost_s = 2.0;
    p->soft_unfreeze_s = 8.0;
    p->hold_eps = 2.0;
    p->hold_tau_s = 0.5;
    p->slow_confirm_s = 5.0;
    p->y_floor_tau_s = 300.0;
    p->tau_r_slow_idle_s = 0.5;
    p->bypass_release_frac = 1.15;
    p->bypass_engage_frac = 1.25;
    p->bypass_noise_sigma = 3.0;
    p->bypass_base_tau_s = 10.0;
    p->ramp_slope_min = 0.5;
    p->ramp_full_s = 4.0;
    p->guard_enable = 0;
    p->guard_floor_confirm_s = 0.5;
    p->guard_idle_confirm_s = 0.3;
    p->guard_floor_deep = 200.0;
}

void dv6_params_plan_v4_observer_k9(dv6_params *p)
{
    if (!p) return;
    dv6_params_default(p);
    p->r_fast = 0.03;
    p->tau_c_fast_s = 2.0;
    p->tau_r_fast_s = 0.1;
    p->r_slow_max = 0.1;
    p->slope_cap_frac = 0.02;
    p->slow_confirm_s = 3.0;
    p->soft_unfreeze_s = 2.0;
    p->tau_r_slow_idle_s = 0.2;
}

size_t dv6_buffer_bytes(int n_channels)
{
    if (n_channels <= 0) return 0;
    /* 13 个状态数组 + 3 个工作数组 = 16n doubles */
    return (size_t)n_channels * 16u * sizeof(double);
}

void dv6_reset(dv6_observer *o)
{
    if (!o) return;
    o->first_frame = 1;
    o->last_ts = 0.0;
    o->total_baseline = 0.0;
    o->total_noise = 0.0;
    /* 逐通道状态在首帧分支里初始化，这里无需清零 */
}

int dv6_init(dv6_observer *o, int n_channels, const dv6_params *p,
             void *buffer, size_t bytes)
{
    if (!o || !p || !buffer || n_channels <= 0) return 1;
    if (bytes < dv6_buffer_bytes(n_channels)) return 2;

    memset(o, 0, sizeof(*o));
    o->n = n_channels;
    o->p = *p;

    double *cursor = (double *)buffer;
    o->x_fast        = cursor; cursor += n_channels;
    o->x_slow        = cursor; cursor += n_channels;
    o->v_lp          = cursor; cursor += n_channels;
    o->zero          = cursor; cursor += n_channels;
    o->y_max         = cursor; cursor += n_channels;
    o->t_edge        = cursor; cursor += n_channels;
    o->applied       = cursor; cursor += n_channels;
    o->v_fast_lp     = cursor; cursor += n_channels;
    o->y_floor       = cursor; cursor += n_channels;
    o->load_dwell    = cursor; cursor += n_channels;
    o->ramp_dwell    = cursor; cursor += n_channels;
    o->below_elapsed = cursor; cursor += n_channels;
    o->idle_elapsed  = cursor; cursor += n_channels;
    o->scratch_y     = cursor; cursor += n_channels;
    o->scratch_slope = cursor; cursor += n_channels;
    o->scratch_e     = cursor; cursor += n_channels;

    dv6_reset(o);
    return 0;
}

double dv6_applied_total(const dv6_observer *o)
{
    if (!o || !o->applied || o->n <= 0) return 0.0;
    double sum = 0.0;
    for (int k = 0; k < o->n; ++k) sum += o->applied[k];
    return sum;
}

void dv6_process(dv6_observer *o, double timestamp_s, double *v)
{
    if (!o || !v) return;
    const int n = o->n;
    if (n <= 0) return;
    const dv6_params *p = &o->p;

    /* ── 首帧初始化 ── */
    if (o->first_frame) {
        o->first_frame = 0;
        o->last_ts = timestamp_s;
        double total = 0.0;
        for (int k = 0; k < n; ++k) {
            o->zero[k] = v[k];
            o->y_max[k] = 0.0;
            o->v_lp[k] = v[k];
            o->v_fast_lp[k] = v[k];
            o->y_floor[k] = v[k];
            o->load_dwell[k] = 0.0;
            o->ramp_dwell[k] = 0.0;
            o->t_edge[k] = 1.0e6;
            o->applied[k] = 0.0;
            o->x_fast[k] = 0.0;
            o->x_slow[k] = 0.0;
            o->below_elapsed[k] = 0.0;
            o->idle_elapsed[k] = 0.0;
            total += v[k];
        }
        o->total_baseline = total;
        o->total_noise = 0.0;
        /* 首帧不补偿：applied 为 0，values_io -= 0 */
        return;
    }

    /* ── 时间步：dt 截断到 0.1s，非正与异常视为 0 ── */
    double dt = timestamp_s - o->last_ts;
    o->last_ts = timestamp_s;
    dt = (dt > 0.0) ? DV6_MIN(dt, 0.1) : 0.0;

    /* ── K7 全局总值旁路（迟滞）── */
    int bypass_frame = 0;
    if (dt > 0.0) {
        double total = 0.0;
        for (int k = 0; k < n; ++k) total += v[k];
        if (total == total) {  /* 非 NaN 判定（不引入 isfinite 依赖） */
            const double alpha = DV6_MIN(dt / p->bypass_base_tau_s, 1.0);
            const double release = DV6_MAX(p->bypass_release_frac * o->total_baseline,
                                           o->total_baseline + p->bypass_noise_sigma * o->total_noise);
            const double engage = DV6_MAX(p->bypass_engage_frac * o->total_baseline,
                                          o->total_baseline + (p->bypass_noise_sigma + 1.0) * o->total_noise);
            if (total < engage) {
                o->total_baseline += alpha * (total - o->total_baseline);
                o->total_noise += alpha * (fabs(total - o->total_baseline) - o->total_noise);
                if (o->total_noise < 0.0) o->total_noise = 0.0;
            }
            if (total < release) bypass_frame = 1;
        }
    }

    if (dt > 0.0) {
        const double y_decay = exp(-dt / p->y_max_tau_s);
        const double refract = DV6_MAX(p->edge_refract_s, p->edge_boost_s);
        const int guard_on = (p->guard_enable != 0);

        /* ── ① 零点跟踪 + 去趋势空载门 ── */
        for (int k = 0; k < n; ++k) {
            const double y0 = v[k] - o->zero[k];

            if (y0 < o->y_floor[k]) {
                /* 下行：原实现瞬时跟随；防护开启时需持续确认（深度跌破除外） */
                if (guard_on) {
                    o->below_elapsed[k] += dt;
                    const double deep = (o->y_floor[k] - y0) > p->guard_floor_deep;
                    if (deep || o->below_elapsed[k] >= p->guard_floor_confirm_s) {
                        o->y_floor[k] = y0;
                    }
                } else {
                    o->y_floor[k] = y0;
                }
            } else {
                if (guard_on) o->below_elapsed[k] = 0.0;
                o->y_floor[k] += (dt / p->y_floor_tau_s) * (y0 - o->y_floor[k]);
            }

            o->y_max[k] = DV6_MAX(o->y_max[k] * y_decay, DV6_MAX(y0, 0.0));

            const double span = DV6_MAX(o->y_max[k] - o->y_floor[k], 1.0);
            const int idle = (y0 - o->y_floor[k]) < p->idle_frac * span;
            o->load_dwell[k] = idle ? 0.0 : (o->load_dwell[k] + dt);
            if (idle) o->zero[k] += (dt / p->tau_zero_s) * (v[k] - o->zero[k]);
        }

        /* ── ② 低通导数 ── */
        for (int k = 0; k < n; ++k) {
            o->scratch_y[k] = v[k] - o->zero[k];
            o->scratch_slope[k] = (v[k] - o->v_lp[k]) / p->tau_slope_s;
        }

        /* ── ③~⑨ 逐通道状态推进 ── */
        for (int k = 0; k < n; ++k) {
            const double y = o->scratch_y[k];
            const double slope = o->scratch_slope[k];

            /* 沿检测（先累加时间再判触发） */
            o->t_edge[k] += dt;
            if (slope > p->edge_slope_thres && o->t_edge[k] > refract) {
                o->t_edge[k] = 0.0;
            }

            /* 缓坡计时 */
            const int ramp_on = (slope > p->ramp_slope_min) && (slope <= p->edge_slope_thres);
            o->ramp_dwell[k] = ramp_on ? (o->ramp_dwell[k] + dt) : 0.0;

            /* 任一快变（升或降）重计受载确认窗 */
            if (fabs(slope) > p->edge_slope_thres) o->load_dwell[k] = 0.0;

            /* 快态收敛 τ：沿后前馈 → 缓坡线性混合 */
            double tc1 = (p->edge_boost_s > 0.0 && o->t_edge[k] < p->edge_boost_s)
                             ? p->tau_c_fast_boost_s
                             : p->tau_c_fast_s;
            if (p->ramp_full_s > 0.0) {
                const double w = DV6_MIN(o->ramp_dwell[k] / p->ramp_full_s, 1.0);
                tc1 = tc1 * (1.0 - w) + p->tau_c_fast_boost_s * w;
            }

            /* 快态 x_fast（e 用更新前的 x_fast 计算） */
            double e_now = y - o->x_fast[k] - o->x_slow[k];
            if (e_now < 0.0) e_now = 0.0;
            const double dx1_rate = (e_now > 0.0)
                                        ? (p->r_fast * e_now - o->x_fast[k]) / tc1
                                        : -o->x_fast[k] / p->tau_r_fast_s;
            o->x_fast[k] = DV6_MAX(o->x_fast[k] + dx1_rate * dt, 0.0);
            o->v_lp[k] += (dt / p->tau_slope_s) * (v[k] - o->v_lp[k]);

            /* 慢态 x_slow（e 用更新后的 x_fast 重算） */
            double e = y - o->x_fast[k] - o->x_slow[k];
            if (e < 0.0) e = 0.0;
            o->scratch_e[k] = e;
            const double span_k = DV6_MAX(o->y_max[k] - o->y_floor[k], 1.0);
            const int idle_k = (y - o->y_floor[k]) < p->idle_frac * span_k;

            if (e > 0.0) {
                const double base = DV6_MAX(e, 1.0);
                const double cap = p->slope_cap_frac * base;
                const int soft_ok = (1.0 - exp(-o->t_edge[k] / p->soft_unfreeze_s)) > 0.5;
                const int dwell_ok = (o->load_dwell[k] >= p->slow_confirm_s);
                if (dwell_ok && soft_ok && fabs(slope) < p->slope_gate_frac * base) {
                    double dx2 = slope - dx1_rate;
                    if (dx2 > cap) dx2 = cap;
                    if (dx2 < -cap) dx2 = -cap;
                    o->x_slow[k] += dt * dx2;
                }
                const double hi = p->r_slow_max * base;
                if (o->x_slow[k] > hi) o->x_slow[k] = hi;
                if (o->x_slow[k] < 0.0) o->x_slow[k] = 0.0;
            }

            /* 空载/残差泄放 */
            if (e <= 0.0 || idle_k) {
                double tau_r2;
                if (guard_on) {
                    o->idle_elapsed[k] = idle_k ? (o->idle_elapsed[k] + dt) : 0.0;
                    const int fast_ok = idle_k &&
                        (o->idle_elapsed[k] >= p->guard_idle_confirm_s) &&
                        (p->tau_r_slow_idle_s > 0.0);
                    tau_r2 = fast_ok ? p->tau_r_slow_idle_s : p->tau_r_slow_s;
                } else {
                    tau_r2 = (idle_k && p->tau_r_slow_idle_s > 0.0)
                                 ? p->tau_r_slow_idle_s
                                 : p->tau_r_slow_s;
                }
                o->x_slow[k] -= (dt / tau_r2) * o->x_slow[k];
                if (o->x_slow[k] < 0.0) o->x_slow[k] = 0.0;
            }

            /* K6 预留池 */
            if (p->hold_eps > 0.0) {
                const double slope_a = (v[k] - o->v_fast_lp[k]) / p->hold_tau_s;
                o->v_fast_lp[k] += (dt / p->hold_tau_s) * (v[k] - o->v_fast_lp[k]);
                const double d_des = o->x_fast[k] + o->x_slow[k];
                const double a_up = o->applied[k] + dt * DV6_MAX(slope_a + p->hold_eps, 0.0);
                const double a_dn = o->applied[k] + dt * (slope_a - p->hold_eps);
                double a = (d_des > o->applied[k]) ? DV6_MIN(d_des, a_up)
                                                   : DV6_MAX(d_des, a_dn);
                if (e > 0.0) {
                    const double lo = 0.0;
                    const double hi2 = DV6_MAX(d_des, 0.0);
                    if (a < lo) a = lo;
                    if (a > hi2) a = hi2;
                } else {
                    a = d_des;
                }
                o->applied[k] = a;
            }
        }
    }

    /* ── 输出 ── */
    if (bypass_frame) {
        for (int k = 0; k < n; ++k) o->applied[k] = o->x_fast[k] + o->x_slow[k];
        return;  /* 空载直通：显示不扣 */
    }
    if (p->hold_eps > 0.0) {
        for (int k = 0; k < n; ++k) v[k] -= o->applied[k];
    } else {
        for (int k = 0; k < n; ++k) v[k] -= (o->x_fast[k] + o->x_slow[k]);
    }
}

#endif /* DV6_IMPLEMENTATION */
