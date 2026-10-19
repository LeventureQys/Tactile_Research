# -*- coding: utf-8 -*-
"""K9 / K12 对比出图（调用数据解析工具的绘图链路，不自己写 matplotlib）。

数据流：
    pre 流 → 送进 K9 插件、K12 插件 → 三条补偿后总值
    四个来源（pre / K9-base / K9-big / K12-big）经 SourceTable.from_arrays 投喂
    → merge.merge_sources 对齐 → merge.plot_table 出 PNG

对照臂：
    K9-base   录制参数（r_slow_max=0.1）—— 现役产品行为
    K9-big    仅 r_slow_max→0.3         —— 只调参
    K12-big   r_slow_max=0.3 + 保护层    —— 结构 + 参数

用法：python k12_figure.py            # 全部会话
      python k12_figure.py --zoom     # 额外输出 10g 会话的局部放大图
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
from dptool import merge as mg  # noqa: E402
from dptool import session_csv as scsv  # noqa: E402

DATA_ROOT = Path(r"D:\workshop\Github\Tactile_Research\Research\时漂问题\data\右拇指指腹")
OUT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = OUT_ROOT / "figures"
RES_DIR = OUT_ROOT / "results"


def load_pre(session_dir: Path):
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    fps = float(meta["devices"][0].get("observed_fps") or 100.45)
    table = scsv.load_table(str(session_dir / "device_001_pre_seg0.csv"), use_cache=False)
    values = table.channels().astype(np.float64)
    time_axis = np.asarray(table.data[:, 2], dtype=np.float64) / fps
    return values, time_axis, tuple(table.channel_ids), meta


def build_table(series, time_axis):
    """series: [(label, y)] → MergedTable（用工具的对齐 + 绘图数据结构）。"""
    sources = []
    for label, values in series:
        source = mg.SourceTable.from_arrays(
            label=label, names=("total",), data=np.asarray(values, dtype=np.float64),
            prefix=label, stream=mg.STREAM_OTHER, path=label + ".csv",
            time={"elapsed": time_axis},
        )
        sources.append(source)
    return mg.merge_sources(sources, align="elapsed")


def med_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(x, pad, mode="edge")
    stride = padded.strides[0]
    view = np.lib.stride_tricks.as_strided(padded, shape=(x.size, win), strides=(stride, stride))
    return np.median(view, axis=1)


def on_index(total: np.ndarray, time: np.ndarray) -> int:
    """加载完成帧：1s 中值平滑后首次超过「空载中位 + 50% 量程」处。"""
    smooth = med_smooth(total, 101)
    lo = float(np.median(smooth[:200]))
    hi = float(np.percentile(smooth, 90))
    idx = int(np.argmax(smooth > lo + 0.5 * (hi - lo)))
    return idx if smooth[idx] > lo + 0.5 * (hi - lo) else 0


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    registry = alg.PluginRegistry()
    registry.scan()
    k9 = registry.get("k9_creep_observer")
    k12 = registry.get("k12_motion_guard")
    defaults = {k: v for k, v in alg.default_params(k9.specs).items() if k != "param_set"}

    sessions = [p for p in sorted(DATA_ROOT.glob("设备*/*")) if (p / "session.json").exists()]
    indices = []
    for session_dir in sessions:
        values, time_axis, ids, meta = load_pre(session_dir)
        base = dict(defaults)
        base.update({k: v for k, v in meta["algorithm"]["params"].items() if k in base})
        big = dict(base, r_slow_max=0.3)

        ctx = alg.AlgorithmContext(mode="dataset", source_path=str(session_dir),
                                   time_axis=time_axis, values=values, channel_ids=ids, meta=meta)
        pre_total = values.sum(axis=1)
        out_k9 = k9.factory().run(ctx, dict(base)).outputs[0].y
        out_big = k9.factory().run(ctx, dict(big)).outputs[0].y
        out_k12 = k12.factory().run(
            ctx, dict(big, motion_guard_enable=True, motion_thresh_adc=60.0)).outputs[0].y

        device = session_dir.parent.name
        tag = "10g" if "10g" in session_dir.name else "恒载"
        title = "%s / %s / %s" % (device, session_dir.name, tag)
        safe = "%s_%s" % (device, "10g" if tag == "10g" else session_dir.name.split("-")[0])

        on = on_index(pre_total, time_axis)
        w = max(20, (pre_total.size - on) // 10)
        series_raw = [("01_pre原始", pre_total), ("02_K9_录制参数", out_k9),
                      ("03_K9_r_slow_max0.3", out_big), ("04_K12_保护层", out_k12)]
        # 漂移增量：各自减去「加载完成后的基线」→ 四条从 0 起，直接可比漂移量。
        # 绝对值图里 2.5 万 ADC 的加载台阶会把几百 ADC 的漂移差异压扁，看不出对比。
        series_delta = [(label, y - float(np.median(y[on:on + w]))) for label, y in series_raw]

        table = build_table(series_raw, time_axis)
        png = FIG_DIR / ("%s_绝对值.png" % safe)
        info = mg.plot_table(table, str(png), mode="overlay", series_by="file",
                             signal="sum", smooth_s=1.0, width_in=16.0,
                             panel_height_in=7.0, dpi=110,
                             suptitle=title + "｜绝对值 · 1s 中值平滑")
        print("出图 %s（%d 条曲线，%d 行）" % (png.name, info["curves"], table.n_rows))

        table_d = build_table(series_delta, time_axis)
        png_d = FIG_DIR / ("%s_漂移增量.png" % safe)
        info_d = mg.plot_table(table_d, str(png_d), mode="overlay", series_by="file",
                               signal="sum", smooth_s=1.0, width_in=16.0,
                               panel_height_in=7.0, dpi=110, check=True,
                               suptitle=title + "｜已减加载后基线（漂移增量 / ADC）")
        print("出图 %s（%d 条曲线）" % (png_d.name, info_d["curves"]))

        # 局部放大：10g 会话取前 35s，看小荷载变化的保留情况
        if "--zoom" in sys.argv and tag == "10g":
            dt = float(np.median(np.diff(time_axis)))
            stop = int(min(time_axis.size, 35.0 / dt))
            zoom = build_table(series_delta, time_axis).slice(0, stop)
            png_zoom = FIG_DIR / ("%s_漂移增量_前35s.png" % safe)
            info2 = mg.plot_table(zoom, str(png_zoom), mode="overlay", series_by="file",
                                  signal="sum", smooth_s=0.5, width_in=16.0,
                                  panel_height_in=7.0, dpi=110,
                                  suptitle=title + "｜前 35s 放大 · 0.5s 平滑")
            print("出图 %s（%d 条曲线，%d 行）" % (png_zoom.name, info2["curves"], zoom.n_rows))

        indices.append({
            "session": session_dir.name, "device": device, "tag": tag, "on_s": float(time_axis[on]),
            "drift_adc": {"pre": float(np.median(pre_total[-w:]) - np.median(pre_total[on:on + w])),
                          "K9_base": float(np.median(out_k9[-w:]) - np.median(out_k9[on:on + w])),
                          "K9_big": float(np.median(out_big[-w:]) - np.median(out_big[on:on + w])),
                          "K12_big": float(np.median(out_k12[-w:]) - np.median(out_k12[on:on + w]))},
            "figure_delta": png_d.name,
        })

    (RES_DIR / "k12_figures.json").write_text(
        json.dumps(indices, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n共 %d 张对比图，索引写入 results/k12_figures.json" % len(indices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
