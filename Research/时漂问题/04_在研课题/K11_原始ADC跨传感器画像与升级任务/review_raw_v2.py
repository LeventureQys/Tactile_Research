from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data"
PREVIOUS = ROOT / "07_原始ADC跨传感器画像_20261020"
OUT = HERE / "raw_review_v2"
WINDOWS = ((30, 120), (300, 420), (480, 600))


def slope(t, y):
    x = t - np.mean(t)
    return np.tensordot(x, y, axes=(0, 0)) / np.sum(x * x) * 60


def sen_slope(t, y):
    i, j = np.triu_indices(len(t), 1)
    return float(np.median((y[j] - y[i]) / (t[j] - t[i])) * 60)


def read_raw(path):
    header = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            header.append(line.strip())
            if line.strip() == "##Data":
                break
        else:
            raise ValueError(f"Missing data marker: {path}")
    assert any(line.split(",")[-1].strip() == "raw_adc" for line in header)
    df = pd.read_csv(path, skiprows=len(header), encoding="utf-8-sig")
    assert all(str(c).startswith("原始_通道") for c in df.columns[3:])
    a = df.to_numpy(dtype=float)
    assert np.isfinite(a).all() and a.shape[1] > 3
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3:]


def bin_data(t, raw):
    sec = np.floor(t - t[0]).astype(int)
    assert np.all(sec >= 0)
    g = pd.DataFrame(raw).groupby(sec)
    channels = g.median()
    total = pd.Series(raw.sum(axis=1)).groupby(sec).median()
    counts = g.size()
    valid = counts.to_numpy() >= max(5, .5 * np.median(counts))
    return channels.index.to_numpy(float) + .5, total.to_numpy(), channels.to_numpy(), valid, counts.to_numpy()


def window_row(sid, axis, t, total, channel, valid, start, end, clock):
    m = (t >= start) & (t < end)
    coverage = int(np.sum(m & valid)) / (end - start)
    present = t[m]
    if len(present) < 5:
        return None
    local_dt = np.diff(clock)
    local_t = clock[:-1] - clock[0]
    gap = float(local_dt[(local_t >= start) & (local_t < end)].max(initial=0))
    usable = bool(coverage >= .95 and np.all(valid[m]) and np.all(np.diff(present) <= 1.01) and gap <= .25)
    base = dict(id=sid, axis=axis, start_s=start, end_s=end, valid_coverage=coverage,
                max_gap_s=gap, usable=usable)
    if not usable:
        return base
    x, y, c = t[m], total[m], channel[m]
    cs = slope(x, c)
    absolute = np.abs(cs)
    delta = float(np.median(y[-5:]) - np.median(y[:5]))
    base.update(total_median_adc=float(np.median(y)), total_delta_adc=delta,
                per_channel_delta_adc=delta / c.shape[1], ols_adc_min=float(slope(x, y)),
                sen_adc_min=sen_slope(x, y), first_half_adc_min=float(slope(x[:len(x)//2], y[:len(x)//2])),
                second_half_adc_min=float(slope(x[len(x)//2:], y[len(x)//2:])),
                max_adjacent_1s_change_adc=float(np.max(np.abs(np.diff(y)))),
                positive_slope_channels=int(np.sum(cs > 1)), negative_slope_channels=int(np.sum(cs < -1)),
                channel_slope_l1_adc_min=float(absolute.sum()),
                cancellation_fraction=float(1 - abs(cs.sum()) / absolute.sum()) if absolute.sum() else None,
                slope_effective_channels=float(absolute.sum() ** 2 / np.sum(cs ** 2)) if np.sum(cs ** 2) else None,
                slope_top5_share=float(np.sort(absolute)[-5:].sum() / absolute.sum()) if absolute.sum() else None)
    return base


def draw(curves):
    image = Image.new("RGB", (1560, 1815), "#f8fafc")
    d = ImageDraw.Draw(image)
    f = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    title = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 23)
    d.text((25, 12), "RAW ONLY | 1 s median of frame totals / channel count", font=title, fill="#17273b")
    d.text((25, 43), "Timestamp axis; independent panel scales; gaps > 0.25 s break lines; no load labels inferred", font=f, fill="#526071")
    for k, (sid, t, y, valid, gaps) in enumerate(curves):
        row, col = divmod(k, 3)
        left, top = 15 + 515 * col, 80 + 245 * row
        x0, x1, y0, y1 = left + 62, left + 495, top + 30, top + 208
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, 1)
        d.text((left + 8, top), sid + " | ADC / channel", font=f, fill="#17273b")
        for q in (0, .5, 1):
            yy = y1 - q * (y1 - y0)
            d.line((x0, yy, x1, yy), fill="#dce3eb")
            d.text((left, yy - 7), f"{lo+q*span:.0f}", font=f, fill="#526071")
        prev = None
        for i, (tt, yy) in enumerate(zip(t, y)):
            point = (x0 + tt / max(t[-1], 1) * (x1-x0), y1 - (yy-lo)/span * (y1-y0))
            crossed_gap = i > 0 and any(t[i-1] <= a <= tt or t[i-1] <= b <= tt or (a <= t[i-1] and b >= tt) for a,b in gaps)
            if valid[i] and prev is not None and t[i]-t[i-1] <= 1.01 and not crossed_gap:
                d.line((prev, point), fill="#167e94", width=2)
            prev = point if valid[i] else None
        d.text((x0, y1+5), "0 s", font=f, fill="#526071")
        d.text((x1-65, y1+5), f"{t[-1]:.0f} s", font=f, fill="#526071")
    image.save(OUT / "raw_per_channel_overview.png")


def main():
    manifest = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))
    expected = {r["id"]: r["sha256"] for r in json.loads((PREVIOUS / "results/provenance.json").read_text(encoding="utf-8"))}
    paths = {DATA / r[2] / "device_001_raw_seg000.csv" for r in manifest["sessions"]}
    assert paths == set(DATA.rglob("device_001_raw_seg*.csv")), "Raw inventory changed"
    OUT.mkdir(exist_ok=True)
    sessions, rows, gaps, provenance, curves = [], [], [], [], []
    for sid, group, rel in manifest["sessions"]:
        path = DATA / rel / "device_001_raw_seg000.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected[sid], f"Raw content changed: {sid}"
        ts, elapsed, frame, raw = read_raw(path)
        dt, de = np.diff(ts), np.diff(elapsed)
        assert np.all(dt >= 0) and np.all(de >= 0)
        assert np.all(np.diff(frame) == 1)
        t = ts - ts[0]
        e = elapsed - elapsed[0]
        gap_intervals = [(float(t[i]),float(t[i+1])) for i in np.flatnonzero(dt > .25)]
        for i in np.flatnonzero(dt > .1):
            gaps.append(dict(id=sid,start_s=float(t[i]),end_s=float(t[i+1]),timestamp_gap_s=float(dt[i]),
                             elapsed_gap_s=float(de[i]),frame_before=int(frame[i]),frame_after=int(frame[i+1])))
        session = dict(id=sid,group=group,frames=len(ts),channels=raw.shape[1],duration_timestamp_s=float(t[-1]),
                       duration_elapsed_s=float(e[-1]),timestamp_repeat=int(np.sum(dt==0)),timestamp_backward=int(np.sum(dt<0)),
                       sub1ms_fraction=float(np.mean(dt<.001)),elapsed_repeat_fraction=float(np.mean(de==0)),
                       max_timestamp_gap_s=float(dt.max()),max_clock_disagreement_s=float(np.max(np.abs(t-e))),
                       exported_frame_ids_contiguous=True)
        for axis, clock in (("timestamp",ts),("elapsed",elapsed)):
            bt, total, channel, valid, counts = bin_data(clock,raw)
            session[axis+"_missing_seconds"] = int(np.floor(clock[-1]-clock[0])+1-len(bt))
            session[axis+"_low_occupancy_seconds"] = int(np.sum(~valid))
            for start,end in WINDOWS:
                if clock[-1]-clock[0] < end:
                    continue
                result = window_row(sid,axis,bt,total,channel,valid,start,end,clock)
                if result:
                    rows.append(result)
            if axis == "timestamp":
                curves.append((sid,bt,total/raw.shape[1],valid,gap_intervals))
        provenance.append(dict(id=sid,path=str(path.relative_to(ROOT)),sha256=digest,matched_previous=True))
        sessions.append(session)
        print(sid,len(ts),raw.shape[1],flush=True)
    sf, wf = pd.DataFrame(sessions), pd.DataFrame(rows)
    sf.to_csv(OUT/"sessions.csv",index=False,encoding="utf-8-sig")
    wf.to_csv(OUT/"windows.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(gaps).to_csv(OUT/"gaps.csv",index=False,encoding="utf-8-sig")
    usable = wf[wf.usable]
    compare = usable[usable.axis=="timestamp"].merge(usable[usable.axis=="elapsed"],on=["id","start_s","end_s"],suffixes=("_ts","_elapsed"))
    compare["delta_clock_difference_adc"] = compare.total_delta_adc_elapsed-compare.total_delta_adc_ts
    compare["slope_clock_difference_adc_min"] = compare.ols_adc_min_elapsed-compare.ols_adc_min_ts
    compare.to_csv(OUT/"clock_sensitivity.csv",index=False,encoding="utf-8-sig")
    (OUT/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    draw(curves)
    print(usable.loc[usable.axis=="timestamp",["id","start_s","end_s","total_delta_adc","ols_adc_min","sen_adc_min","positive_slope_channels","negative_slope_channels","cancellation_fraction","slope_effective_channels"]].round(3).to_string(index=False))
    print("CLOCK_MAX", compare[["delta_clock_difference_adc","slope_clock_difference_adc_min"]].abs().max().to_dict())
    print("TOTAL",len(sf),int(sf.frames.sum()),sf.duration_timestamp_s.sum(),int(sf.timestamp_repeat.sum()))


if __name__ == "__main__":
    main()
