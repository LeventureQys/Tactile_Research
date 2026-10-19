/* =============================================================================
 * dv6_portable_abi.c
 *
 * 把 creep_observer_portable.h 包装成 C ABI 动态库，供 C# / Python / 其它语言
 * 通过 FFI 调用（同一份算法文件，不产生第二份算法副本）。
 *
 * 编译（MSVC）:
 *     cl /O2 /LD /Fe:dv6_portable.dll dv6_portable_abi.c
 * 编译（GCC）:
 *     gcc -O2 -shared -fPIC -o libdv6_portable.so dv6_portable_abi.c
 *
 * 本文件只做内存管理与参数搬运，**不含任何算法逻辑**。
 * 嵌入式直接包含 creep_observer_portable.h，不需要本文件。
 * ========================================================================== */

#define DV6_IMPLEMENTATION
#include "creep_observer_portable.h"

#include <stdlib.h>

#if defined(_WIN32)
#define DV6_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define DV6_EXPORT __attribute__((visibility("default")))
#else
#define DV6_EXPORT
#endif

typedef struct dv6_handle {
    dv6_observer obs;
    void *buffer;   /* 状态缓冲，free 时需要 */
} dv6_handle;

/* preset: 0 = 工作树默认, 1 = plan-v4 observer-k9；params 非空时以 params 为准 */
DV6_EXPORT void *dv6_portable_new(int n_channels, const dv6_params *params, int preset)
{
    if (n_channels <= 0) return NULL;
    const size_t bytes = dv6_buffer_bytes(n_channels);

    dv6_handle *h = (dv6_handle *)malloc(sizeof(dv6_handle));
    if (!h) return NULL;
    void *buffer = malloc(bytes);
    if (!buffer) { free(h); return NULL; }

    dv6_params p;
    if (params) {
        p = *params;
    } else if (preset == 1) {
        dv6_params_plan_v4_observer_k9(&p);
    } else {
        dv6_params_default(&p);
    }

    if (dv6_init(&h->obs, n_channels, &p, buffer, bytes) != 0) {
        free(buffer);
        free(h);
        return NULL;
    }
    h->buffer = buffer;
    return h;
}

DV6_EXPORT void dv6_portable_free(void *handle)
{
    if (!handle) return;
    dv6_handle *h = (dv6_handle *)handle;
    free(h->buffer);
    free(h);
}

DV6_EXPORT void dv6_portable_set_params(void *handle, const dv6_params *params)
{
    if (!handle || !params) return;
    dv6_handle *h = (dv6_handle *)handle;
    h->obs.p = *params;
    dv6_reset(&h->obs);
}

DV6_EXPORT void dv6_portable_reset(void *handle)
{
    if (!handle) return;
    dv6_reset(&((dv6_handle *)handle)->obs);
}

/* 就地补偿一帧；n 必须等于创建时的通道数 */
DV6_EXPORT void dv6_portable_process(void *handle, double ts, double *values, int n)
{
    if (!handle || !values) return;
    dv6_handle *h = (dv6_handle *)handle;
    if (n != h->obs.n) return;
    dv6_process(&h->obs, ts, values);
}

DV6_EXPORT double dv6_portable_applied_total(void *handle)
{
    if (!handle) return 0.0;
    return dv6_applied_total(&((dv6_handle *)handle)->obs);
}

DV6_EXPORT int dv6_portable_channels(void *handle)
{
    if (!handle) return 0;
    return ((dv6_handle *)handle)->obs.n;
}

DV6_EXPORT int dv6_portable_params_bytes(void) { return (int)sizeof(dv6_params); }
