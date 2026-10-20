# -*- coding: utf-8 -*-
"""K9 现役基线 vs 路线B —— 右拇指指腹 6 份录制的补偿效果对比。

对比口径（同一份 pre 输入，四种算法输出）：
  pre       未补偿原始（算法输入，dc 基线）
  out_rec   录制算法结果（设备端实跑的 K9，param_set = plan-v4 observer-k9）
  k9_rec    Python K9 复算，参数 = session.json 原值（r_slow_max=0.1）→ 用于与 out_rec 对拍
  k9_0p4    Python K9 复算，只把 r_slow_max 提到 0.4（路线B 的背景通路）
  routeb    路线B（K9@0.4 + 因果创新检测 + 可回滚平台保护）

用法：
  python compare_k9_vs_routeb.py metrics      # 只算指标，落 metrics.json
  python compare_k9_vs_routeb.py figures      # 只出图（复用 metrics 阶段缓存）
  python compare_k9_vs_routeb.py all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROUTE_B_DIR = HERE.parent
K9_WORK = ROUTE_B_DIR.parents[1]                     # 时漂问题
K9_IMPL = K9_WORK / "01_当前工作_K9" / "02_现场调参" / "09-21_09-26_工程A_v3.4"
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
DATA = K9_WORK / "data" / "右拇指指腹"

for path in (str(ROUTE_B_DIR), str(K9_IMPL), str(TOOLBOX)):
    if path not in sys.path:
        sys.path.insert(0, path)

from creep_observer_k9 import CreepObserverK9, Params, fields  # noqa: E402
from dptool.session_csv import ROLE_FRAME_INDEX, load_table    # noqa: E402
import route_b as RB                                           # noqa: E402

SESSIONS = [
    ("设备1/测试1", "设备1", "测试1-2026-10-13_16-32-15"),
    ("设备1/测试2", "设备1", "测试2-2026-10-13_16-43-59"),
    ("设备1/10g反复增减", "设备1", "10g反复增减荷载-2026-10-13_17-07-18"),
    ("设备2/测试1", "设备2", "测试1-2026-10-13_16-53-24"),
    ("设备2/测试2", "设备2", "测试2-2026-10-13_17-01-06"),
    ("设备2/10g反复增减", "设备2", "10g反复增减荷载-2026-10-13_17-05-32"),
]

JUMP_ADC = 2000.0        # 与 route_b.load_folder 一致的加载跳变判据
METHODS = ("pre", "out_rec", "k9_rec", "k9_0p4", "routeb")


def _frame_axis(table) -> np.ndarray:
    """按角色取帧序列（B 代中文表头的列名是「帧序号」，不能用名字取）。"""
    found = table.header.by_role(ROLE_FRAME_INDEX)
    if not found:
        raise KeyError("表里没有帧序号列：%s" % table.header.path)
    return np.asarray(table.data[:, found[0].index], dtype=np.float64)


# ── 数据 ──────────────────────────────────────────────────────
def load_session(device: str, name: str) -> dict:
    folder = DATA / device / name
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    fps = float(session["devices"][0]["observed_fps"])
    pre = load_table(str(folder / "device_001_pre_seg0.csv"), use_cache=True)
    out = load_table(str(folder / "device_001_seg000.csv"), use_cache=True)
    frame = _frame_axis(pre)
    t = (frame - frame[0]) / fps
    return dict(
        id="%s/%s" % (device, name), device=device, name=name, folder=folder,
        session=session, fps=fps, frame=frame, t=t,
        pre=np.asarray(pre.channels(), dtype=np.float64),
        out=np.asarray(out.channels(), dtype=np.float64),
        out_frame=_frame_axis(out),
    )


def params_of(session: dict, r_slow_max: float) -> Params:
    raw = dict(session["algorithm"]["params"])
    names = {f.name for f in fields(Params)}
    p = {k: float(v) for k, v in raw.items() if k in names}
    p["r_slow_max"] = float(r_slow_max)
    return Params(**p)


def run_k9(values: np.ndarray, t: np.ndarray, params: Params) -> np.ndarray:
    obs = CreepObserverK9(params)
    obs._trace_frame = lambda *args: None
    out = np.empty_like(values)
    for i in range(values.shape[0]):
        out[i] = obs.process(float(t[i]), values[i])
    return out


def run_routeb(values: np.ndarray, t: np.ndarray, raw_params: dict) -> tuple[np.ndarray, dict]:
    params = dict(raw_params)
    params["r_slow_max"] = 0.4
    out, trace = RB.route_b(values, t, params)
    return out, trace


# ── 指标 ──────────────────────────────────────────────────────
def detect_steps(t: np.ndarray, total: np.ndarray, min_adc: float,
                 refractory_s: float = 1.0) -> list[dict]:
    """总力曲线上检出台阶：相邻帧 |Δ| ≥ min_adc，合并不应期内的连续帧。"""
    d = np.diff(total)
    idx = np.flatnonzero(np.abs(d) >= min_adc)
    steps: list[dict] = []
    for i in idx:
        if steps and t[i] - steps[-1]["t"] < refractory_s:
            step = steps[-1]
            step["adc"] = float(total[min(i + 1, total.size - 1)] - total[step["i"]])
            step["i"] = int(i)
            step["t"] = float(t[i])
            step["peak_delta"] = max(step["peak_delta"], abs(float(d[i])))
            continue
        steps.append(dict(i=int(i), t=float(t[i]),
                          adc=float(total[min(i + 1, total.size - 1)] - total[i]),
                          peak_delta=abs(float(d[i]))))
    return steps


def med(values: np.ndarray, t: np.ndarray, a: float, b: float) -> float | None:
    z = values[(t >= a) & (t < b)]
    return float(np.median(z)) if z.size else None


def step_span(t: np.ndarray, sig: np.ndarray, load: float) -> float | None:
    """台阶幅值：加载前 12~7 s 中值 → 加载后 1~6 s 中值；窗口被截断时返回 None。"""
    before = med(sig, t, max(float(t[0]), load - 12.), load - 7.)
    after = med(sig, t, load + 1., load + 6.)
    if before is None or after is None:
        return None
    return float(after - before)


def window_offsets(t: np.ndarray, sig: np.ndarray, start: float, width: float = 5.,
                   step: float = 1.) -> np.ndarray:
    base = med(sig, t, start, start + width)
    if base is None:
        return np.zeros(0)
    out = []
    for s in np.arange(start, t[-1] - width + 1e-9, step):
        value = med(sig, t, s, s + width)
        if value is not None:
            out.append(value - base)
    return np.asarray(out, dtype=np.float64)


def step_retention(t: np.ndarray, sig: np.ndarray, ref: np.ndarray, steps: list[dict],
                   n_each: int = 6) -> list[dict]:
    """每个台阶的保留率：台阶前后各取 3 秒中值，比较输出与未补偿参考的差值。

    保留率 = (sig 的台阶幅值) / (ref 的台阶幅值)；仅统计 |ref 幅值| ≥ 50 ADC 的台阶。"""
    rows = []
    for step in steps[:n_each]:
        ts = step["t"]
        before = med(sig, t, ts - 3.0, ts - 0.3)
        after = med(sig, t, ts + 1.0, ts + 4.0)
        rb_before = med(ref, t, ts - 3.0, ts - 0.3)
        rb_after = med(ref, t, ts + 1.0, ts + 4.0)
        if None in (before, after, rb_before, rb_after):
            continue
        ref_step = rb_after - rb_before
        if abs(ref_step) < 50.0:
            continue
        rows.append(dict(t=round(ts, 2), ref_step_adc=float(ref_step),
                         out_step_adc=float(after - before),
                         retention=float((after - before) / ref_step)))
    return rows


def metrics(t: np.ndarray, sig: np.ndarray, load: float, tail_width: float = 5.) -> dict:
    start = load + 15.
    first = med(sig, t, start, start + 5.)
    last = med(sig, t, max(t[-1] - tail_width, start), t[-1])
    offs = window_offsets(t, sig, start)
    if first is None or last is None or offs.size == 0:
        return dict(net_change_adc=None, max_abs_offset_adc=None, tail_window_adc=None)
    return dict(net_change_adc=float(last - first),
                max_abs_offset_adc=float(np.max(np.abs(offs))),
                tail_window_adc=float(offs[-1]))


def analyse(sess: dict) -> tuple[dict, dict]:
    t, pre, out_rec = sess["t"], sess["pre"], sess["out"]
    raw_total = pre.sum(1)
    steps = detect_steps(t, raw_total, JUMP_ADC)
    load = steps[0]["t"] if steps else float(t[0])

    outputs: dict[str, np.ndarray] = {"pre": pre, "out_rec": out_rec}
    raw_params = dict(sess["session"]["algorithm"]["params"])
    outputs["k9_rec"] = run_k9(pre, t, params_of(sess["session"], raw_params["r_slow_max"]))
    outputs["k9_0p4"] = run_k9(pre, t, params_of(sess["session"], 0.4))
    routeb_out, trace = run_routeb(pre, t, raw_params)
    outputs["routeb"] = routeb_out

    totals = {k: v.sum(1) for k, v in outputs.items()}
    report: dict = {
        "id": sess["id"], "fps": sess["fps"], "frames": int(t.size),
        "duration_s": float(t[-1] - t[0]), "load_time_s": load,
        "steps": steps,
        "frame_match": bool(np.array_equal(sess["frame"], sess["out_frame"])),
        "k9_python_vs_recording": dict(
            max_abs_adc=float(np.max(np.abs(outputs["k9_rec"] - out_rec))),
            mean_abs_adc=float(np.mean(np.abs(outputs["k9_rec"] - out_rec)))),
        "routeb_events": trace["events"],
        "methods": {},
    }
    for method in METHODS:
        entry = dict(metrics(t, totals[method], load))
        entry["post_load_first5_adc"] = med(totals[method], t, load + 15., load + 20.)
        entry["load_step_adc"] = step_span(t, totals[method], load)
        entry["step_retention"] = step_retention(t, totals[method], raw_total, steps, n_each=10)
        report["methods"][method] = entry
    report["load_step_ref_adc"] = step_span(t, raw_total, load)
    if len(steps) > 1:
        unload = steps[-1]["t"] if steps[-1]["adc"] < 0 else None
        if unload is not None:
            report["withdrawal"] = {
                "t": unload,
                "residual_1_5s": {m: med(totals[m], t, unload + 1., unload + 5.)
                                  for m in METHODS},
                "residual_7_12s": {m: med(totals[m], t, unload + 7., unload + 12.)
                                   for m in METHODS},
            }
    series = {"t": t, "frame": sess["frame"], "load_time_s": np.float64(load),
              **{("total_" + m): totals[m] for m in METHODS}}
    return report, series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="all",
                        choices=("metrics", "figures", "all"))
    parser.add_argument("--only", default=None, help="只跑匹配 id 的会话（子串）")
    args = parser.parse_args()

    rows = []
    (HERE / "series").mkdir(exist_ok=True)
    for label, device, name in SESSIONS:
        if args.only and args.only not in label:
            continue
        started = time.perf_counter()
        sess = load_session(device, name)
        report, series = analyse(sess)
        report["label"] = label
        np.savez_compressed(HERE / "series" / (label.replace("/", "_") + ".npz"), **series)
        report["elapsed_s"] = round(time.perf_counter() - started, 2)
        rows.append(report)
        m = report["methods"]
        print("%-20s load=%7.2fs steps=%2d  "
              "net[pre=%9.1f k9rec=%8.1f k9.4=%8.1f B=%8.1f]  "
              "maxoff[pre=%9.1f k9rec=%8.1f k9.4=%8.1f B=%8.1f]  pyK9vsRec=%.3f"
              % (label, report["load_time_s"], len(report["steps"]),
                 m["pre"]["net_change_adc"], m["k9_rec"]["net_change_adc"],
                 m["k9_0p4"]["net_change_adc"], m["routeb"]["net_change_adc"],
                 m["pre"]["max_abs_offset_adc"], m["k9_rec"]["max_abs_offset_adc"],
                 m["k9_0p4"]["max_abs_offset_adc"], m["routeb"]["max_abs_offset_adc"],
                 report["k9_python_vs_recording"]["max_abs_adc"]), flush=True)

    payload = {"sessions": rows, "jump_adc": JUMP_ADC,
               "config": {"route_b": RB.CONFIG}}
    (HERE / "compare_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> %s" % (HERE / "compare_metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
