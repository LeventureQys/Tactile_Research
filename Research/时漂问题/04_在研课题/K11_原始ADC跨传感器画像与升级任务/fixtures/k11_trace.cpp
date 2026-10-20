// K11 v1 逐帧锚点探针（一次性验证，不入产品构建）
// 目的：给「K11 v1 算法移植文档」提供可对拍的逐帧锚点。
// 合成序列与 k11_probe.cpp 的 T2 完全一致（n=57、dt=0.01、t>=5 加载、0.2 s 斜坡、
// creep 90*(1-exp(-(t-5)/60))），同一份参照实现在 K10 档与 K11 v1 档并行跑。
// 输出：CSV，列见表头。移植方用同样序列跑自己的实现，逐列 diff 即可判定等价性。
#include "domain/drift_v6/creep_observer.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>

using drift_v6::CreepObserverCompensator;

namespace {

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

    CreepObserverCompensator k10, k11;
    k10.SetParams(p10);
    k11.SetParams(p11);

    std::printf("# K11 v1 trace, n=%d dt=%.4f\n", n, dt);
    std::printf("# frame,t,state,allow,short,long,common,energy,quality,out11,out10,diffmax\n");

    double t = 0.0;
    Eigen::VectorXd a = MakeFrame(n, 0.0);
    Eigen::VectorXd b = MakeFrame(n, 0.0);
    k10.Process(t, a);
    k11.Process(t, b);
    std::printf("%d,%.4f,%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%.6f,%.6f,%.6f\n", 0, t, k11.k11_state(),
                static_cast<int>(k11.k11_allow_slow()), k11.k11_short_slope(),
                k11.k11_long_slope(), k11.k11_common_mode_ratio(), k11.k11_spatial_energy(),
                k11.k11_time_quality(), b.sum(), a.sum(), 0.0);

    int frame = 0;
    while (t < 160.0) {
        t += dt;
        ++frame;
        double load = 0.0;
        if (t >= 5.0 && t < 125.0) {
            const double since = t - 5.0;
            const double ramp = std::min(since / 0.20, 1.0);
            const double creep = 90.0 * (1.0 - std::exp(-since / 60.0));
            load = (500.0 + creep) * ramp;
        }
        a = MakeFrame(n, load);
        b = MakeFrame(n, load);
        k10.Process(t, a);
        k11.Process(t, b);

        const double diff = (a - b).cwiseAbs().maxCoeff();
        // 采样：加载前每 100 帧（1 s）一行；加载沿附近每 10 帧（0.1 s）一行。
        const bool dense = (frame >= 480 && frame <= 1700 && frame % 10 == 0);
        if (frame % 100 == 0 || dense) {
            std::printf("%d,%.4f,%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%.6f,%.6f,%.6f\n", frame, t,
                        k11.k11_state(), static_cast<int>(k11.k11_allow_slow()),
                        k11.k11_short_slope(), k11.k11_long_slope(),
                        k11.k11_common_mode_ratio(), k11.k11_spatial_energy(),
                        k11.k11_time_quality(), b.sum(), a.sum(), diff);
        }
    }
    return 0;
}
