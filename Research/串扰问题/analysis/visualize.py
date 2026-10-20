# -*- coding: utf-8 -*-
"""可视化：单点事件压制前后对比 + 串扰随距离的衰减规律。

使用 Pillow 直接绘制热力格阵，输出 PNG 到 analysis/out/。
"""

import io
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosstalk_core import response, response_with_baseline, to_frames  # noqa: E402
from crosstalk_lib import iter_all_recordings  # noqa: E402
from crosstalk_suppress import BiasTable, CrosstalkSuppressor, SuppressConfig  # noqa: E402
from run_suppress import match_table_key  # noqa: E402

RAW = r"D:\workshop\Github\Tactile_Research\Research\串扰问题\raw_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BIAS_JSON = os.path.join(OUT, "bias_table.json")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def get_font(size: int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def color_of(v: float, vmax: float) -> tuple[int, int, int]:
    """黑 → 蓝 → 红 → 黄 → 白的非线性映射。"""
    if vmax <= 0:
        return (0, 0, 0)
    x = min(1.0, max(0.0, (v / vmax) ** 0.5))
    stops = [(0.0, (8, 8, 16)), (0.15, (20, 40, 120)), (0.4, (150, 30, 60)),
             (0.7, (230, 140, 30)), (1.0, (255, 250, 210))]
    for i in range(len(stops) - 1):
        a, ca = stops[i]
        b, cb = stops[i + 1]
        if a <= x <= b:
            t = (x - a) / (b - a)
            return tuple(int(ca[k] + (cb[k] - ca[k]) * t) for k in range(3))
    return stops[-1][1]


def draw_grid(draw, y, rec, x0, y0, cell, vmax, label_font, title=None, mask_na=True):
    idx = rec.index_map
    for r in range(rec.rows):
        for c in range(rec.cols):
            x = x0 + c * cell
            yy = y0 + r * cell
            j = idx[r, c]
            if j < 0:
                if mask_na:
                    draw.rectangle([x, yy, x + cell - 2, yy + cell - 2], fill=(22, 22, 26))
                continue
            v = y[r, c]
            if not np.isfinite(v):
                fill = (22, 22, 26)
                txt = ""
            else:
                fill = color_of(v, vmax)
                txt = f"{int(round(v))}" if v > 0.5 else ""
            draw.rectangle([x, yy, x + cell - 2, yy + cell - 2], fill=fill)
            if txt:
                lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
                draw.text((x + cell / 2, yy + cell / 2), txt,
                          fill=(0, 0, 0) if lum > 130 else (235, 235, 235),
                          font=label_font, anchor="mm")
    if title:
        draw.text((x0, y0 - 8), title, fill=(230, 230, 235), font=get_font(15), anchor="lb")


def figure_case(rec, t, frames, y_out, info, path, cell=42):
    """一行：原始 | 压制后，附带文字说明。"""
    pad = 40
    rows, cols = rec.rows, rec.cols
    w = pad + cols * cell + 40 + cols * cell + pad
    h = pad + rows * cell + pad + 34
    img = Image.new("RGB", (w, h), (12, 12, 14))
    d = ImageDraw.Draw(img)
    vmax = float(np.nanmax(frames[t]))
    f_small = get_font(12)
    f_norm = get_font(14)
    axis_name = "横向(同行列间)" if info["direction"] == "row" else "纵向(同列行间)"
    d.text((pad, 14), f"{rec.label}   帧 {rec.frame_index[t]}   t={rec.elapsed[t]:.2f}s",
           fill=(235, 235, 240), font=f_norm)
    d.text((pad, 32), f"主点=({info['r0']},{info['c0']})  峰值={vmax:.0f}  "
                      f"串扰方向={axis_name}  压制点={info['suppressed']}",
           fill=(180, 190, 205), font=f_small)
    draw_grid(d, frames[t], rec, pad, pad + 26, cell, vmax, f_small, title="原始读数")
    x2 = pad + cols * cell + 40
    draw_grid(d, y_out[t], rec, x2, pad + 26, cell, vmax, f_small, title="压制后（恢复静息底值）")
    img.save(path)


def figure_decay(records, path, cell_w=560, cell_h=250, ncols=4):
    """串扰幅值/比例 随主点位置距离的分布（2 行网格）。"""
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))   # label -> dist -> ratios
    for rec in records:
        d = response(rec.data)
        frames = to_frames(rec, d)
        cfg = SuppressConfig(direction="auto", mode="zero")
        sup = CrosstalkSuppressor(rec, cfg).calibrate(d)
        P_all = np.nanmax(frames, axis=(1, 2))
        thr = 0.3 * np.nanpercentile(P_all, 99.5)
        for t in range(frames.shape[0]):
            if not np.isfinite(P_all[t]) or P_all[t] < thr:
                continue
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            P = f[r0, c0]
            rs = np.nansum(np.delete(f[r0, :], c0)) / P
            cs = np.nansum(np.delete(f[:, c0], r0)) / P
            if rs > cs:
                for c in range(rec.cols):
                    if c != c0 and np.isfinite(f[r0, c]):
                        data[rec.label][abs(c - c0)].append(float(f[r0, c] / P))
            elif cs > rs:
                for r in range(rec.rows):
                    if r != r0 and np.isfinite(f[r, c0]):
                        data[rec.label][abs(r - r0)].append(float(f[r, c0] / P))

    labels = sorted(data.keys())
    nrows = (len(labels) + ncols - 1) // ncols
    w, h = cell_w * ncols, cell_h * nrows + 60
    img = Image.new("RGB", (w, h), (16, 16, 20))
    d = ImageDraw.Draw(img)
    f = get_font(15)
    f2 = get_font(12)
    f3 = get_font(11)
    for i, lab in enumerate(labels):
        x0 = (i % ncols) * cell_w
        y_base = (i // ncols) * cell_h + 40
        is_multi = "大拇指" in lab
        d.text((x0 + 14, y_base - 26), f"{lab}   ── 串扰沿{'横向(同行列间)' if is_multi else '纵向(同列行间)'}",
               fill=(235, 235, 240), font=f)
        ax0, ay0 = x0 + 52, y_base
        ax1, ay1 = x0 + cell_w - 24, y_base + cell_h - 46
        d.rectangle([ax0, ay0, ax1, ay1], outline=(70, 70, 80))
        maxd = max(data[lab].keys())
        ymax = max(0.05, min(1.0, max(np.percentile(v, 90) for v in data[lab].values())))
        for k in range(5):
            yv = ymax * k / 4
            yy = ay1 - (ay1 - ay0) * (yv / ymax)
            d.line([ax0 - 4, yy, ax1, yy], fill=(40, 40, 46))
            d.text((ax0 - 8, yy), f"{yv*100:.0f}%", fill=(170, 170, 180), font=f3, anchor="rm")
        for dist in sorted(data[lab].keys()):
            v = np.asarray(data[lab][dist])
            xx = ax0 + (ax1 - ax0) * (dist / max(maxd, 1))
            p50 = np.median(v)
            p25, p75 = np.percentile(v, 25), np.percentile(v, 75)
            yy = ay1 - (ay1 - ay0) * (min(p50, ymax) / ymax)
            y25 = ay1 - (ay1 - ay0) * (min(p75, ymax) / ymax)
            y75 = ay1 - (ay1 - ay0) * (min(p25, ymax) / ymax)
            d.line([xx, y25, xx, y75], fill=(90, 160, 220), width=3)
            d.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=(240, 200, 60))
            d.text((xx, ay1 + 5), str(dist), fill=(170, 170, 180), font=f3, anchor="ma")
        d.text((ax0 - 44, ay0 - 16), "串扰/主点", fill=(170, 170, 180), font=f3)
        d.text((ax1, ay1 + 22), "距主点格数", fill=(170, 170, 180), font=f3, anchor="ra")
    img.save(path)


def figure_direction_evidence(records, path, cell_w=620, cell_h=150):
    """方向判据证据：主点同行其余格之和 / 同列其余格之和（归一化），10 分位~90 分位箱线。"""
    from collections import defaultdict
    stats = defaultdict(lambda: dict(row=[], col=[]))
    for rec in records:
        d = response(rec.data)
        frames = to_frames(rec, d)
        P_all = np.nanmax(frames, axis=(1, 2))
        thr = 0.3 * np.nanpercentile(P_all, 99.5)
        for t in range(frames.shape[0]):
            if not np.isfinite(P_all[t]) or P_all[t] < thr:
                continue
            f = frames[t]
            r0, c0 = np.unravel_index(np.nanargmax(f), (rec.rows, rec.cols))
            P = f[r0, c0]
            stats[rec.label]["row"].append(float(np.nansum(np.delete(f[r0, :], c0)) / P))
            stats[rec.label]["col"].append(float(np.nansum(np.delete(f[:, c0], r0)) / P))

    labels = sorted(stats.keys())
    n = len(labels)
    w, h = cell_w, cell_h * ((n + 1) // 2) + 60
    img = Image.new("RGB", (w, h), (16, 16, 20))
    d = ImageDraw.Draw(img)
    f = get_font(16)
    f2 = get_font(12)
    d.text((16, 16), "方向判据：主点所在行的其余格之和（横向） vs 所在列的其余格之和（纵向）",
           fill=(235, 235, 240), font=f)
    for i, lab in enumerate(labels):
        y0 = 52 + (i // 2) * cell_h
        x0 = (i % 2) * (cell_w // 2) + 16
        r = np.asarray(stats[lab]["row"])
        c = np.asarray(stats[lab]["col"])
        rm, cm = np.median(r), np.median(c)
        bar_w = 150
        # 对数刻度长度
        def ln(v):
            return math.log10(max(v, 1e-4) + 1.0)
        scale = 130.0 / max(ln(max(rm, cm)), 1e-6)
        d.text((x0, y0), lab, fill=(230, 230, 240), font=get_font(14))
        d.rectangle([x0, y0 + 22, x0 + ln(rm) * scale, y0 + 40], fill=(90, 160, 220))
        d.text((x0 + ln(rm) * scale + 6, y0 + 24), f"横向 {rm*100:.1f}%", fill=(190, 210, 235), font=f2)
        d.rectangle([x0, y0 + 48, x0 + ln(cm) * scale, y0 + 66], fill=(230, 140, 30))
        d.text((x0 + ln(cm) * scale + 6, y0 + 50), f"纵向 {cm*100:.1f}%", fill=(240, 200, 150), font=f2)
    img.save(path)


def main():
    tables = {}
    if os.path.exists(BIAS_JSON):
        with io.open(BIAS_JSON, "r", encoding="utf-8") as f:
            import json
            tables = json.load(f)

    recs = list(iter_all_recordings(RAW))
    for rec in recs:
        d, base = response_with_baseline(rec.data)
        # 展示统一在**数据域**（原始读数），与写回的 CSV 一致：
        # 未压制位置与输入一致，压制位置回到该通道的静息底值。
        base_grid = to_frames(rec, base)
        frames = to_frames(rec, d) + base_grid
        key = match_table_key(tables, rec.label)
        bt = BiasTable.from_json(tables[key]) if key else None
        cfg = SuppressConfig(direction=(bt.direction if bt else "auto"), mode="auto")
        sup = CrosstalkSuppressor(rec, cfg, bias=bt).calibrate(d)
        y_out_resp, infos = sup.process_batch(d)
        y_out = y_out_resp + base_grid

        # 选 4 个代表帧：峰值最高的若干单点帧，去掉时间相邻的
        sel = [t for t, i in enumerate(infos) if i["mode"] == "single"]
        sel.sort(key=lambda t: -infos[t]["peak"])
        chosen = []
        for t in sel:
            if all(abs(t - u) > 60 for u in chosen):
                chosen.append(t)
            if len(chosen) == 4:
                break
        if not chosen:
            continue
        # 拼成 2x2 的大图
        tiles = []
        for t in chosen:
            p = os.path.join(OUT, f"_tmp_{rec.label.replace('/', '_')}_{t}.png")
            figure_case(rec, t, frames, y_out, infos[t], p)
            tiles.append(Image.open(p))
        tw = max(i.width for i in tiles)
        th = sum(i.height for i in tiles[:2])
        canvas = Image.new("RGB", (tw * 2 + 20, th * 2 + 20), (10, 10, 12))
        for k, im in enumerate(tiles):
            canvas.paste(im, ((k % 2) * (tw + 20), (k // 2) * (im.height + 20)))
        out = os.path.join(OUT, f"fig_单点压制对比_{rec.label.replace('/', '_')}.png")
        canvas.save(out)
        for t in chosen:
            p = os.path.join(OUT, f"_tmp_{rec.label.replace('/', '_')}_{t}.png")
            if os.path.exists(p):
                os.remove(p)
        print(f"已生成 {out}", file=sys.stderr)

    figure_decay(recs, os.path.join(OUT, "fig_串扰随距离衰减.png"))
    figure_direction_evidence(recs, os.path.join(OUT, "fig_方向判据证据.png"))
    print("已生成 串扰随距离衰减图 / 方向判据证据图", file=sys.stderr)


if __name__ == "__main__":
    main()
