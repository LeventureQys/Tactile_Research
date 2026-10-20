// 对拍脚手架（临时验证用，不属于产品代码）：
// 从二进制流读 (ts, v[n]) 逐帧喂给产品 CreepObserverCompensator，输出补偿后 v[n]。
// 用途：把 src/domain/drift_v6/creep_observer.cpp 与 portable 权威实现逐帧对齐。
//
// usage: harness <in.bin> <out.bin> <n_ch> <gate> [guard] [preset] [recover_s]
//   preset = 1 时把 Params 覆盖成现场下发的 plan-v4 observer-k9 值
//   （含 2026-10-19 会话的 r_slow_max=0.6），其余保持产品头文件默认。
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "domain/drift_v6/creep_observer.h"

namespace {

void ApplyPlanV4(drift_v6::CreepObserverCompensator::Params& p) {
    p.r_fast = 0.03;
    p.tau_c_fast_s = 2.0;
    p.tau_r_fast_s = 0.1;
    p.r_slow_max = 0.6;          // 2026-10-19 现场值（默认 0.2 / plan-v4 文档值 0.1）
    p.slope_cap_frac = 0.02;
    p.slow_confirm_s = 3.0;
    p.soft_unfreeze_s = 2.0;
    p.tau_r_slow_idle_s = 0.2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: harness <in.bin> <out.bin> <n_ch> <gate> [guard] [preset] [recover_s]\n");
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    const int n_ch = std::atoi(argv[3]);
    const int gate = std::atoi(argv[4]);
    const int guard = (argc > 5) ? std::atoi(argv[5]) : -1;
    const int preset = (argc > 6) ? std::atoi(argv[6]) : 0;
    const double recover = (argc > 7) ? std::atof(argv[7]) : 0.0;

    std::FILE* fi = std::fopen(in_path, "rb");
    if (!fi) { std::fprintf(stderr, "open in failed\n"); return 2; }
    std::FILE* fo = std::fopen(out_path, "wb");
    if (!fo) { std::fprintf(stderr, "open out failed\n"); std::fclose(fi); return 2; }

    drift_v6::CreepObserverCompensator comp;
    drift_v6::CreepObserverCompensator::Params p;
    if (preset == 1) ApplyPlanV4(p);
    p.floor_rise_idle_only = gate;
    p.guard_floor_recover_s = recover;
    if (guard >= 0) p.guard_enable = guard;
    comp.SetParams(p);

    std::vector<double> buf(static_cast<std::size_t>(n_ch) + 1);
    Eigen::VectorXd v(n_ch);
    long frames = 0;
    while (std::fread(buf.data(), sizeof(double), buf.size(), fi) == buf.size()) {
        for (int k = 0; k < n_ch; ++k) v(k) = buf[static_cast<std::size_t>(k) + 1];
        comp.Process(buf[0], v);
        std::fwrite(v.data(), sizeof(double), static_cast<std::size_t>(n_ch), fo);
        ++frames;
    }
    std::fclose(fi);
    std::fclose(fo);
    std::fprintf(stderr, "frames=%ld gate=%d guard=%d preset=%d recover=%.1f\n", frames, gate, guard, preset, recover);
    return 0;
}
