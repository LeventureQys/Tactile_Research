from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
ROOT = (HERE / MANIFEST["root"]).resolve()
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def mad_sigma(values, axis=None):
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(values - center), axis=axis)


def read_raw(path):
    lines = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
            if line.strip() == "##Data":
                break
        else:
            raise ValueError(f"No data marker: {path}")
    frame = pd.read_csv(path, skiprows=len(lines), encoding="utf-8-sig")
    if frame.shape[1] < 4:
        raise ValueError(f"Too few columns: {path}")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite input: {path}")
    return values[:, 0], values[:, 1], values[:, 2], values[:, 3:], lines


def consecutive_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def linear_slope(times, values):
    centered = times - np.mean(times)
    denom = np.sum(centered * centered)
    if denom <= 0:
        return np.full(values.shape[1:], np.nan)
    return np.tensordot(centered, values, axes=(0, 0)) / denom


def font(size):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_panel(draw, box, times, values, title, color, limits=None):
    left, top, right, bottom = box
    draw.text((left, top), title, font=font(17), fill="#17273b")
    plot = (left + 60, top + 35, right - 15, bottom - 30)
    x0, y0, x1, y1 = plot
    lo, hi = limits or (float(np.min(values)), float(np.max(values)))
    if hi <= lo:
        hi = lo + 1.0
    hi += (hi - lo) * 0.03
    for fraction in (0.0, 0.5, 1.0):
        yy = y1 - fraction * (y1 - y0)
        draw.line((x0, yy, x1, yy), fill="#dae2eb", width=1)
        draw.text((left, yy - 7), f"{lo + fraction * (hi-lo):.0f}", font=font(12), fill="#526071")
    end = max(float(times[-1]), 1.0)
    points = [(x0 + float(tt) / end * (x1 - x0), y1 - (float(vv) - lo) / (hi - lo) * (y1 - y0)) for tt, vv in zip(times, values)]
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    draw.text((x0, y1 + 5), "0 s", font=font(12), fill="#526071")
    draw.text((x1 - 60, y1 + 5), f"{end:.0f} s", font=font(12), fill="#526071")


def main():
    summaries, window_rows, channel_rows, provenance = [], [], [], []
    curves = {}
    all_second_rows = []
    for sid, group, relative in MANIFEST["sessions"]:
        folder = ROOT / relative
        path = folder / "device_001_raw_seg000.csv"
        ts, elapsed, frame_ids, raw, header = read_raw(path)
        time = ts - ts[0]
        if np.any(np.diff(time) < 0):
            raise ValueError(f"Backward timestamp: {sid}")
        total = raw.sum(axis=1)
        seconds = np.floor(time).astype(int)
        grouped = pd.DataFrame(raw).groupby(seconds, sort=True)
        binned = grouped.median()
        bt = binned.index.to_numpy(dtype=float) + 0.5
        bv = binned.to_numpy()
        totals_1s = pd.Series(total).groupby(seconds, sort=True).median()
        bs = totals_1s.to_numpy()
        counts = grouped.size().to_numpy()
        valid_bins = counts >= max(5, np.median(counts) * 0.5)
        raw_dt = np.diff(ts)
        positive_dt = raw_dt[raw_dt > 0]
        q05, q50, q95 = np.percentile(bs[valid_bins], [5, 50, 95])
        span = q95 - q05
        baseline = np.median(raw[time < min(1.0, time[-1] / 2)], axis=0)
        first_total = float(np.median(total[time < 1.0]))
        last_total = float(np.median(total[time >= max(time[-1] - 1.0, 0.0)]))
        high = (bs > q05 + 0.55 * span) & valid_bins
        contiguous = np.r_[True, np.diff(bt) < 1.1]
        runs = consecutive_runs(high)
        usable_runs = [(start, end) for start, end in runs if end-start >= 30 and np.all(contiguous[start+1:end])]
        longest = max(usable_runs, key=lambda pair: pair[1] - pair[0], default=None)
        sample_diff = np.diff(total)
        high_diff = sample_diff[high[np.minimum(np.searchsorted(bt, time[1:], side="right"), len(high)-1)]]
        high_jitter = float(mad_sigma(high_diff) / np.sqrt(2)) if len(high_diff) else np.nan
        first_channel = np.median(raw[time < 1.0], axis=0)
        high_channel = np.median(bv[high], axis=0) if high.any() else np.median(raw, axis=0)
        activation = np.maximum(high_channel - first_channel, 0.0)
        effective_channels = float(activation.sum() ** 2 / np.sum(activation**2)) if np.sum(activation**2) > 0 else np.nan
        top5 = float(np.sort(activation)[-5:].sum() / activation.sum()) if activation.sum() > 0 else np.nan
        summary = dict(id=sid, group=group, session=relative, frames=len(time), channels=raw.shape[1], duration_s=float(time[-1]),
                       effective_fps=(len(time)-1)/time[-1], first_total_adc=first_total, last_total_adc=last_total,
                       total_q05_adc=q05, total_q50_adc=q50, total_q95_adc=q95, excursion_adc=span,
                       high_frame_variation_adc=high_jitter, activation_effective_channels=effective_channels,
                       activation_top5_share=top5, timestamp_nonpositive=int(np.sum(raw_dt<=0)),
                       timestamp_burst_lt1ms_frac=float(np.mean(raw_dt<0.001)), timestamp_gap_gt100ms=int(np.sum(raw_dt>0.1)),
                       timestamp_max_gap_s=float(raw_dt.max()), timestamp_positive_dt_median_ms=float(np.median(positive_dt)*1000),
                       elapsed_repeat_frac=float(np.mean(np.diff(elapsed)==0)), frame_missing=int(np.sum(np.maximum(np.diff(frame_ids)-1,0))),
                       raw_min_adc=float(raw.min()), raw_max_adc=float(raw.max()), high_run_count=len(usable_runs))
        for ch in range(raw.shape[1]):
            channel_rows.append(dict(id=sid, channel=ch, first_adc=first_channel[ch], high_median_adc=high_channel[ch],
                                     activation_adc=activation[ch], raw_q05_adc=np.percentile(raw[:, ch],5),
                                     raw_q95_adc=np.percentile(raw[:, ch],95), successive_diff_sigma_adc=float(mad_sigma(np.diff(raw[:,ch]))/np.sqrt(2))))
        run_results = []
        for start, end in usable_runs:
            start = start + 5
            if end-start < 20:
                continue
            x = bt[start:end]
            y = bs[start:end]
            delta = float(np.median(y[-5:]) - np.median(y[:5]))
            slope = float(linear_slope(x, y)*60)
            changes = np.median(bv[end-5:end],axis=0) - np.median(bv[start:start+5],axis=0)
            slopes = linear_slope(x,bv[start:end])*60
            load = np.median(bv[start:end],axis=0)-baseline
            active = load > max(5.0, float(np.max(load))*0.02)
            corr = float(np.corrcoef(load[active],slopes[active])[0,1]) if active.sum()>3 and np.std(load[active])>0 and np.std(slopes[active])>0 else np.nan
            first_part = max(5,len(y)//3)
            early = float(linear_slope(x[:first_part],y[:first_part])*60)
            late = float(linear_slope(x[-first_part:],y[-first_part:])*60)
            run_results.append(dict(start_s=float(x[0]-.5),end_s=float(x[-1]+.5),duration_s=len(y),delta_adc=delta,
                                    slope_adc_min=slope,early_slope_adc_min=early,late_slope_adc_min=late,
                                    cancellation_ratio=float(abs(changes.sum())/np.abs(changes).sum()) if np.abs(changes).sum()>0 else np.nan,
                                    channel_positive=int(np.sum(changes>1)),channel_negative=int(np.sum(changes<-1)),
                                    channel_load_slope_corr=corr))
        if run_results:
            chosen=max(run_results,key=lambda row:row["duration_s"])
            summary.update({"longest_"+key:value for key,value in chosen.items()})
        for duration in (30,60,120):
            for start in range(0,len(bt)-duration+1,duration):
                end=start+duration
                if not np.all(valid_bins[start:end]) or not np.all(np.diff(bt[start:end])<1.1):
                    continue
                x,y=bt[start:end],bs[start:end]
                delta=float(np.median(y[-5:])-np.median(y[:5]))
                slope=float(linear_slope(x,y)*60)
                jumps=float(np.max(np.abs(np.diff(y))))
                candidate=bool(np.mean(high[start:end])>=.95 and jumps<=.08*span)
                changes=np.median(bv[end-5:end],axis=0)-np.median(bv[start:start+5],axis=0)
                window_rows.append(dict(id=sid,duration_s=duration,start_s=float(x[0]-.5),end_s=float(x[-1]+.5),
                                        median_adc=float(np.median(y)),delta_adc=delta,slope_adc_min=slope,
                                        delta_over_excursion=delta/span if span>0 else np.nan,
                                        max_1s_jump_adc=jumps,candidate_high_no_major_jump=candidate,
                                        channel_positive=int(np.sum(changes>1)),channel_negative=int(np.sum(changes<-1)),
                                        cancellation_ratio=float(abs(changes.sum())/np.abs(changes).sum()) if np.abs(changes).sum()>0 else np.nan))
        session=json.loads((folder/"session.json").read_text(encoding="utf-8-sig"))
        identity=[]
        for dev in session.get("devices",[]):
            identity.append({key:dev.get(key) for key in ("id","device_id","name","alias","port","device_sn","serial_number","observed_fps","first_timestamp","last_timestamp","frame_count") if key in dev})
        header_explicit=any("raw_adc" in line for line in header)
        provenance.append(dict(id=sid,raw_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               bytes=path.stat().st_size,header_explicit_raw_adc=header_explicit,header=header,
                               identity=identity,columns=raw.shape[1],high_runs=run_results,
                               metadata_note="Session algorithm results/parameters are not used as numerical evidence."))
        summaries.append(summary)
        curves[sid]=dict(t=bt,total=bs,channels=bv,high=high,valid=valid_bins)
        for tt,vv,count in zip(bt,bs,counts):
            all_second_rows.append(dict(id=sid,time_s=float(tt),raw_total_median_adc=float(vv),frame_count=int(count)))
        print(json.dumps({key:summary.get(key) for key in ("id","frames","channels","duration_s","excursion_adc","longest_delta_adc","longest_early_slope_adc_min","longest_late_slope_adc_min")},ensure_ascii=True),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/"sessions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(window_rows).to_csv(OUT/"windows.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(channel_rows).to_csv(OUT/"channels.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(all_second_rows).to_csv(OUT/"raw_total_1s.csv",index=False,encoding="utf-8-sig")
    (OUT/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(OUT/"raw_1s_channels.npz",**{sid+"_"+key:val for sid,curve in curves.items() for key,val in curve.items()})
    image=Image.new("RGB",(1590,1790),"#f8fafc")
    draw=ImageDraw.Draw(image)
    draw.text((25,15),"RAW ADC ONLY | 21 sessions | 1 s median total | independent axes",font=font(25),fill="#17273b")
    for index,(sid,group,relative) in enumerate(MANIFEST["sessions"]):
        row,col=divmod(index,3)
        curve=curves[sid]
        draw_panel(draw,(15+col*525,65+row*245,520+col*525,300+row*245),curve["t"],curve["total"],sid+" | "+str(len(curve["channels"][0]))+" channels", "#167e94")
    image.save(OUT/"raw_overview.png")
    selected=["NP1","NP2","NT1","NT2","NF1","NF2"]
    image=Image.new("RGB",(1500,970),"#f8fafc")
    draw=ImageDraw.Draw(image)
    draw.text((25,15),"RAW ADC | six October 15 sessions | relative to first-second median",font=font(24),fill="#17273b")
    for index,sid in enumerate(selected):
        row,col=divmod(index,2)
        curve=curves[sid]
        draw_panel(draw,(15+col*745,65+row*295,740+col*745,350+row*295),curve["t"],curve["total"]-curve["total"][0],sid,"#b55732")
    image.save(OUT/"six_sessions.png")
    print("DONE",len(summaries),"sessions",sum(item["frames"] for item in summaries),"frames",flush=True)


if __name__=="__main__":
    main()
