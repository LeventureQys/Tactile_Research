# -*- coding: utf-8 -*-
"""K12 插件接入验证：
  1. 工具箱能扫描到 k12_motion_guard；
  2. motion_guard_enable=False 时与 K9 输出**逐位等价**（A/B 对照有效的前提）；
  3. 开启保护层后在 10g 会话上的漂移 / 事件保留差异；
  4. 参数 schema 合法（validate_param_specs）。

用法：python k12_verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLBOX = r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具"
if TOOLBOX not in sys.path:
    sys.path.insert(0, TOOLBOX)
from dptool import algorithms as alg  # noqa: E402
from dptool import session_csv as scsv  # noqa: E402

SESSION = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右拇指指腹"
               r"\设备2\10g反复增减荷载-2026-10-13_17-05-32")
FPS_FALLBACK = 100.45


def load_session(session_dir: Path):
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0].get("observed_fps") or FPS_FALLBACK)
    pre = scsv.load_table(str(session_dir / "device_001_pre_seg0.csv"), use_cache=False)
    values = pre.channels().astype(np.float64)
    time_axis = np.asarray(pre.data[:, 2], dtype=np.float64) / fps
    return values, time_axis, tuple(pre.channel_ids), meta


def main() -> int:
    registry = alg.PluginRegistry()
    registry.scan()
    ids = [info.id for info in registry.plugins()]
    print("扫描到的插件：%s" % ids)
    broken = registry.broken()
    if broken:
        for info in broken:
            print("  [损坏] %s: %s" % (info.source_path, info.error))
        return 1
    if "k12_motion_guard" not in ids or "k9_creep_observer" not in ids:
        print("缺少所需插件")
        return 1

    k9 = registry.get("k9_creep_observer")
    k12 = registry.get("k12_motion_guard")
    errors = alg.validate_param_specs(k12.specs)
    print("K12 schema 校验：%s（%d 项参数）"
          % ("通过" if not errors else errors, len(k12.specs)))

    values, time_axis, ids_ch, meta = load_session(SESSION)
    print("会话 %s：%d 帧 × %d 通道，%.1fs"
          % (SESSION.name, values.shape[0], values.shape[1], time_axis[-1]))

    recorded = meta["algorithm"]["params"]
    base = {k: v for k, v in alg.default_params(k9.specs).items() if k != "param_set"}
    base.update({k: v for k, v in recorded.items() if k in base})
    print("使用录制参数：r_fast=%g r_slow_max=%g slope_cap_frac=%g slow_confirm_s=%g"
          % (base["r_fast"], base["r_slow_max"], base["slope_cap_frac"], base["slow_confirm_s"]))

    ctx = alg.AlgorithmContext(mode="dataset", source_path=str(SESSION), time_axis=time_axis,
                               values=values, channel_ids=ids_ch, meta=meta)
    res_k9 = k9.factory().run(ctx, dict(base))
    total_k9 = res_k9.outputs[0].y

    # 1) 保护层关闭 → 必须与 K9 逐位等价
    params_off = dict(base)
    params_off["motion_guard_enable"] = False
    res_off = k12.factory().run(ctx, params_off)
    total_off = res_off.outputs[0].y
    max_diff = float(np.max(np.abs(total_off - total_k9)))
    print("\n[等价性] guard=False vs K9：max|Δ| = %.3e ADC  →  %s"
          % (max_diff, "通过（逐位等价）" if max_diff == 0.0 else "不一致，需排查"))

    # 2) 保护层开启
    for thresh in (25.0, 60.0, 120.0):
        params_on = dict(base)
        params_on["motion_guard_enable"] = True
        params_on["motion_thresh_adc"] = thresh
        res_on = k12.factory().run(ctx, params_on)
        total_on = res_on.outputs[0].y

        def stat(x):
            """加载完成后的净漂移：从 t=3s 起（本会话加载在 1.6s 完成）。"""
            on = int(np.searchsorted(time_axis, 3.0))
            w = max(20, (x.size - on) // 10)
            head = float(np.median(x[on:on + w]))
            tail = float(np.median(x[-w:]))
            return tail - head

        print("\n[motion_thresh=%5.1f ADC] %s" % (thresh, res_on.note))
        print("  K9   加载后净漂移 %+8.0f ADC" % stat(total_k9))
        print("  K12  加载后净漂移 %+8.0f ADC" % stat(total_on))
        print("  K12  与 K9 输出轨线 RMS 差 %.1f ADC，max %.1f ADC"
              % (float(np.sqrt(np.mean((total_on - total_k9) ** 2))),
                 float(np.max(np.abs(total_on - total_k9)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
