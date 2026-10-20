// K11 v1 内核行为探针（一次性验证，不入产品构建）
// 目的：验证三件事，全部用合成数据，不读真实录制：
//   ① k11_enable=0 时新逻辑不被执行（状态量保持初值，输出与 K10 逐帧一致）；
//   ② 加载跳变期间 x_slow 增长被冻结（k11 档补偿滞后、输出更高），
//      平台确认后恢复跟踪；
//   ③ 断流（dt 超过 k11_max_gap_s）把状态推进到 UNCERTAIN 并冻结慢态。
#include "domain/drift_v6/creep_observer.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>

using drift_v6::CreepObserverCompensator;

namespace {

struct Metrics {
    double max_abs_out_diff = 0.0;   // 两档输出的最大绝对差（补偿后读数之差）
    double sum_out_diff = 0.0;       // 输出差之和（正 = k11 扣得更少）
    int samples = 0;
};

// 合成一帧：n 通道基线 + 载荷。
Eigen::VectorXd MakeFrame(int n, double load) {
    Eigen::VectorXd v(n);
    for (int k = 0; k < n; ++k) v(k) = 8.0 + static_cast<double>(k % 7) + load;
    return v;
}

}  // namespace

int main() {
    const int n = 57;
    const double dt = 0.01;

    CreepObserverCompensator::Params p10;
    p10.guard_enable = 1;
    p10.k11_enable = 0;
    CreepObserverCompensator::Params p11 = p10;
    p11.k11_enable = 1;

    // ── 测试 1：关闭态不被执行 ────────────────────────────────────────────
    {
        CreepObserverCompensator off;
        off.SetParams(p10);
        Eigen::VectorXd v = MakeFrame(n, 0.0);
        double t = 0.0;
        off.Process(t, v);
        for (int i = 0; i < 5000; ++i) {
            t += dt;
            Eigen::VectorXd f = MakeFrame(n, (t > 2.0) ? 500.0 : 0.0);
            off.Process(t, f);
        }
        const bool untouched = off.k11_state() == 0 && off.k11_time_quality() == 0 &&
                               off.k11_allow_slow() && off.k11_short_slope() == 0.0 &&
                               off.k11_long_slope() == 0.0;
        std::printf("[T1] k11 disabled -> state=%d quality=%d allow=%d short=%.3f long=%.3f : %s\n",
                    off.k11_state(), off.k11_time_quality(), static_cast<int>(off.k11_allow_slow()),
                    off.k11_short_slope(), off.k11_long_slope(),
                    untouched ? "PASS (new path not executed)" : "FAIL (new path ran while disabled)");
    }

    // ── 测试 2：加载段冻结 + 平台后恢复 ──────────────────────────────────
    {
        CreepObserverCompensator k10, k11;
        k10.SetParams(p10);
        k11.SetParams(p11);
        Metrics m;
        double t = 0.0;
        int stable_first = -1;
        int rising_samples = 0;
        double xs_growth_during_rising = 0.0;
        double prev_xs11 = 0.0;
        int state_hist[7] = {0, 0, 0, 0, 0, 0, 0};

        Eigen::VectorXd f = MakeFrame(n, 0.0);
        k10.Process(t, f);
        f = MakeFrame(n, 0.0);
        k11.Process(t, f);
        prev_xs11 = 0.0;

        const double t_total = 160.0;
        while (t < t_total) {
            t += dt;
            double load = 0.0;
            if (t >= 5.0 && t < 125.0) {
                const double since = t - 5.0;
                const double ramp = std::min(since / 0.20, 1.0);         // 0.2s 加载完成
                const double creep = 90.0 * (1.0 - std::exp(-since / 60.0));  // 每通道最多 ~90 ADC
                load = (500.0 + creep) * ramp;
            }
            Eigen::VectorXd a = MakeFrame(n, load);
            Eigen::VectorXd b = MakeFrame(n, load);
            k10.Process(t, a);
            k11.Process(t, b);

            const double diff = (a - b).cwiseAbs().maxCoeff();
            m.max_abs_out_diff = std::max(m.max_abs_out_diff, diff);
            m.sum_out_diff += (b.sum() - a.sum());
            ++m.samples;

            const int s = k11.k11_state();
            if (s >= 0 && s < 7) ++state_hist[s];
            if (s == 1) {
                ++rising_samples;
                // 用输出差反推 x_slow 的相对增长（x_slow 不可直接读）
                const double xs11 = (MakeFrame(n, load).sum() - b.sum());
                xs_growth_during_rising += (xs11 - prev_xs11);
                prev_xs11 = xs11;
            }
            if (s == 4 && stable_first < 0 && t >= 5.0) stable_first = static_cast<int>(t * 1000);
            if (t > 30.0 && t < 31.0) {
                std::printf("[T2] t=%.2f state=%d allow=%d short=%8.2f long=%8.2f "
                            "common=%.2f diff_out=%8.1f\n",
                            t, s, static_cast<int>(k11.k11_allow_slow()),
                            k11.k11_short_slope(), k11.k11_long_slope(),
                            k11.k11_common_mode_ratio(), diff);
            }
        }
        std::printf("[T2] states hist init=%d rising=%d hold=%d cand=%d stable=%d unload=%d unc=%d\n",
                    state_hist[0], state_hist[1], state_hist[2], state_hist[3], state_hist[4],
                    state_hist[5], state_hist[6]);
        std::printf("[T2] first STABLE_PLATEAU at t=%.2fs, rising_samples=%d, "
                    "max|out diff|=%0.1f ADC, mean signed diff=%+.2f ADC\n",
                    stable_first < 0 ? -1.0 : stable_first / 1000.0, rising_samples,
                    m.max_abs_out_diff, m.sum_out_diff / std::max(m.samples, 1));
    }

    // ── 测试 3：断流进入 UNCERTAIN ───────────────────────────────────────
    {
        CreepObserverCompensator k11;
        k11.SetParams(p11);
        double t = 0.0;
        Eigen::VectorXd f = MakeFrame(n, 0.0);
        k11.Process(t, f);
        for (int i = 0; i < 4000; ++i) {   // 40s 空载，让状态稳定
            t += dt;
            f = MakeFrame(n, 0.0);
            k11.Process(t, f);
        }
        const int before = k11.k11_state();
        t += 1.50;                          // 注入 1.5s 断流
        f = MakeFrame(n, 0.0);
        k11.Process(t, f);
        const int after = k11.k11_state();
        const int q = k11.k11_time_quality();
        const bool allow = k11.k11_allow_slow();
        std::printf("[T3] gap injection: state %d -> %d, quality=%d, allow_slow=%d : %s\n",
                    before, after, q, static_cast<int>(allow),
                    (after == 6 && q == 2 && !allow) ? "PASS (UNCERTAIN + frozen)"
                                                     : "FAIL (gap not handled)");
        // 恢复：连续正常帧后应离开 UNCERTAIN
        for (int i = 0; i < 500; ++i) {
            t += dt;
            f = MakeFrame(n, 0.0);
            k11.Process(t, f);
        }
        std::printf("[T3] after recovery: state=%d quality=%d allow=%d : %s\n",
                    k11.k11_state(), k11.k11_time_quality(),
                    static_cast<int>(k11.k11_allow_slow()),
                    k11.k11_state() != 6 ? "PASS (left UNCERTAIN)" : "WARN (still UNCERTAIN)");
    }

    // ── 测试 4：重复时间戳不推进慢态 ─────────────────────────────────────
    {
        CreepObserverCompensator k11;
        k11.SetParams(p11);
        double t = 0.0;
        Eigen::VectorXd f = MakeFrame(n, 0.0);
        k11.Process(t, f);
        for (int i = 0; i < 3000; ++i) {
            t += dt;
            f = MakeFrame(n, 300.0);
            k11.Process(t, f);
        }
        const int state_before = k11.k11_state();
        f = MakeFrame(n, 300.0);
        k11.Process(t, f);   // 同一时间戳再来一帧
        std::printf("[T4] repeated ts: state %d -> %d, quality=%d, allow=%d : %s\n",
                    state_before, k11.k11_state(), k11.k11_time_quality(),
                    static_cast<int>(k11.k11_allow_slow()),
                    (k11.k11_time_quality() == 1 && !k11.k11_allow_slow())
                        ? "PASS (frozen on repeated timestamp)"
                        : "FAIL (repeated timestamp not gated)");
    }

    return 0;
}
