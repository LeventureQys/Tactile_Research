// 同编译器对拍：把 portable 权威实现与产品 creep_observer.cpp 放进同一进程、同一优化级别，
// 逐帧比较，定位两份实现的分叉点（排除编译/ctypes 因素）。
#define DV6_IMPLEMENTATION
#include "creep_observer_portable.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "domain/drift_v6/creep_observer.h"

namespace {

void ApplyPlanV4(drift_v6::CreepObserverCompensator::Params& p) {
    p.r_fast = 0.03;
    p.tau_c_fast_s = 2.0;
    p.tau_r_fast_s = 0.1;
    p.r_slow_max = 0.6;
    p.slope_cap_frac = 0.02;
    p.slow_confirm_s = 3.0;
    p.soft_unfreeze_s = 2.0;
    p.tau_r_slow_idle_s = 0.2;
}

void ApplyPlanV4Portable(dv6_params& p) {
    p.r_fast = 0.03;
    p.tau_c_fast_s = 2.0;
    p.tau_r_fast_s = 0.1;
    p.r_slow_max = 0.6;
    p.slope_cap_frac = 0.02;
    p.slow_confirm_s = 3.0;
    p.soft_unfreeze_s = 2.0;
    p.tau_r_slow_idle_s = 0.2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: harness2 <in.bin> <n_ch> [gate] [guard]\n");
        return 1;
    }
    const char* in_path = argv[1];
    const int n_ch = std::atoi(argv[2]);
    const int gate = (argc > 3) ? std::atoi(argv[3]) : 0;
    const int guard = (argc > 4) ? std::atoi(argv[4]) : 0;

    std::FILE* fi = std::fopen(in_path, "rb");
    if (!fi) { std::fprintf(stderr, "open failed\n"); return 2; }

    drift_v6::CreepObserverCompensator prod;
    drift_v6::CreepObserverCompensator::Params pp;
    ApplyPlanV4(pp);
    pp.floor_rise_idle_only = gate;
    pp.guard_enable = guard;
    prod.SetParams(pp);

    dv6_params dp;
    dv6_params_default(&dp);
    ApplyPlanV4Portable(dp);
    dp.floor_rise_idle_only = gate;
    dp.guard_enable = guard;
    std::vector<double> buf_mem(static_cast<std::size_t>(n_ch) * 16, 0.0);
    dv6_observer port;
    dv6_init(&port, n_ch, &dp, buf_mem.data(), buf_mem.size() * sizeof(double));

    std::vector<double> buf(static_cast<std::size_t>(n_ch) + 1);
    std::vector<double> cpp_v(static_cast<std::size_t>(n_ch));
    std::vector<double> c_v(static_cast<std::size_t>(n_ch));
    Eigen::VectorXd ev(n_ch);

    long frames = 0;
    long first_bad = -1; long nan_count = 0;
    double worst = 0.0;
    int worst_ch = -1;
    while (std::fread(buf.data(), sizeof(double), buf.size(), fi) == buf.size()) {
        for (int k = 0; k < n_ch; ++k) c_v[static_cast<std::size_t>(k)] = buf[static_cast<std::size_t>(k) + 1];

        for (int k = 0; k < n_ch; ++k) ev(k) = c_v[static_cast<std::size_t>(k)];
        prod.Process(buf[0], ev);
        for (int k = 0; k < n_ch; ++k) cpp_v[static_cast<std::size_t>(k)] = ev(k);

        dv6_process(&port, buf[0], c_v.data());

        for (int k = 0; k < n_ch; ++k) {
            const double d = std::fabs(cpp_v[static_cast<std::size_t>(k)] - c_v[static_cast<std::size_t>(k)]);
            if (!std::isfinite(d)) { ++nan_count; continue; }
            if (d > worst) { worst = d; worst_ch = k; }
            if (d > 1e-9 && first_bad < 0) {
                first_bad = frames;
                std::fprintf(stderr,
                             "first divergence frame=%ld t=%.6f ch=%d cpp=%.12f portable=%.12f d=%.3e\n",
                             frames, buf[0], k, cpp_v[static_cast<std::size_t>(k)],
                             c_v[static_cast<std::size_t>(k)], d);
            }
        }
        if (frames >= 66 && frames <= 69) {
            std::fprintf(stderr, "frame %ld t=%.6f  cpp=[%.9f %.9f %.9f]  port=[%.9f %.9f %.9f]\n",
                         frames, buf[0], cpp_v[0], cpp_v[1], cpp_v[2], c_v[0], c_v[1], c_v[2]);
        }
        ++frames;
    }
    std::fclose(fi);
    std::fprintf(stderr,
                 "frames=%ld gate=%d guard=%d worst|Δ|=%.6e (ch %d) first_bad=%ld nan=%ld\n",
                 frames, gate, guard, worst, worst_ch, first_bad, nan_count);
    return 0;
}

