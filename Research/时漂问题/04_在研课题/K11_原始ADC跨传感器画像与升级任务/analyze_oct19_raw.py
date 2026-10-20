from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "raw_review_oct19"
WINDOWS = [(30, 120), (300, 420), (480, 600), (1200, 1500), (2400, 2700), (4800, 5100)]


def slope(t, y):
    x = t - np.mean(t)
    return np.tensordot(x, y, axes=(0, 0)) / (x @ x) * 60


def sen(t, y):
    i, j = np.triu_indices(len(t), 1)
    return float(np.median((y[j] - y[i]) / (t[j] - t[i])) * 60)


def read_raw(path):
    header = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            header.append(line.strip())
            if line.strip() == "##Data":
                break
        else:
            raise ValueError("Missing ##Data")
    fields = dict(line.split(",", 1) for line in header if "," in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    assert fields.get("数据阶段") == "raw_adc"
    assert fields.get("算法施加") == "none"
    assert fields.get("阈值已施加") == "false"
    df = pd.read_csv(path, skiprows=len(header), encoding="utf-8-sig")
    assert all(str(c).startswith("原始_通道") for c in df.columns[3:])
    a = df.to_numpy(dtype=float)
    assert np.isfinite(a).all()
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3:], fields


def bins(clock, raw):
    t = clock - clock[0]
    sec = np.floor(t).astype(int)
    g = pd.Series(raw.sum(axis=1)).groupby(sec)
    b = g.agg(["median", "mean", "count", "min", "max"])
    b["t"] = b.index + .5
    c = pd.DataFrame(raw).groupby(sec).median().to_numpy()
    return b, c


def analyze(sid, path):
    ts, elapsed, frame, raw, header = read_raw(path)
    t = ts - ts[0]
    dt, de = np.diff(ts), np.diff(elapsed)
    assert np.all(dt >= 0) and np.all(de >= 0)
    assert np.all(np.diff(frame) == 1)
    total = raw.sum(axis=1)
    b, channels = bins(ts, raw)
    b.to_csv(OUT / (sid + "_1s.csv"), index_label="second")
    np.savez_compressed(OUT / (sid + "_channels_1s.npz"), t=b.t.to_numpy(), channels=channels)
    meta = json.loads(path.with_name("session.json").read_text(encoding="utf-8"))
    targets = meta.get("device_force_mapping", {}).get("targets", [])
    ev = total[frame.astype(int) % 2 == 0]
    od = total[frame.astype(int) % 2 == 1]
    diff = np.diff(total)
    corr = float(np.corrcoef(diff[:-1], diff[1:])[0, 1]) if np.std(diff) else None
    first = (t < 10)
    last = (t >= t[-1] - 10)
    s = dict(id=sid, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             header=header, targets=targets, frames=len(t), channels=raw.shape[1], duration_s=float(t[-1]),
             elapsed_duration_s=float(elapsed[-1]-elapsed[0]), timestamp_repeat=int(np.sum(dt == 0)),
             timestamp_backward=int(np.sum(dt < 0)), max_gap_s=float(dt.max()),
             sub1ms_fraction=float(np.mean(dt < .001)), elapsed_repeat_fraction=float(np.mean(de == 0)),
             max_clock_disagreement_s=float(np.max(np.abs(t-(elapsed-elapsed[0])))),
             median_dt_ms=float(np.median(dt)*1000), frame_ids_contiguous=True,
             first10_median_total=float(np.median(total[first])), last10_median_total=float(np.median(total[last])),
             total_min=float(total.min()), total_max=float(total.max()),
             even_median_total=float(np.median(ev)), odd_median_total=float(np.median(od)),
             adjacent_difference_abs_median=float(np.median(np.abs(diff))),
             difference_lag1_correlation=corr, raw_min=float(raw.min()), raw_max=float(raw.max()),
             q99_abs_mean_minus_median_1s=float(np.quantile(np.abs(b['mean']-b['median']), .99)))
    phase = []
    for a in range(0, int(t[-1]), 60):
        m = (t >= a) & (t < a+60)
        if m.sum() < 20:
            continue
        r, f = raw[m], frame[m].astype(int)
        x, y = r[f % 2 == 0], r[f % 2 == 1]
        gap = np.median(x, axis=0) - np.median(y, axis=0)
        phase.append(dict(id=sid,start_s=a,even_total_median=float(np.median(x.sum(axis=1))),
                          odd_total_median=float(np.median(y.sum(axis=1))),
                          channel_parity_gap_l1=float(np.abs(gap).sum()),
                          max_channel_parity_gap=float(np.max(np.abs(gap))),
                          channels_parity_gap_gt10=int(np.sum(np.abs(gap) > 10))))
    rows = []
    for axis, clock in [("timestamp",ts),("elapsed",elapsed)]:
        bb, cc = (b,channels) if axis == "timestamp" else bins(clock,raw)
        x = bb.t.to_numpy()
        d = np.diff(clock)
        ct = clock[:-1]-clock[0]
        for a,z in WINDOWS:
            if clock[-1]-clock[0] < z:
                continue
            m = (x >= a)&(x < z)
            yy = bb.loc[m,'median'].to_numpy()
            xx = x[m]
            good = bool(len(xx)==z-a and bb.loc[m,'count'].min()>=max(5,.5*bb['count'].median()) and d[(ct>=a)&(ct<z)].max(initial=0)<=.25)
            row = dict(id=sid,axis=axis,start_s=a,end_s=z,usable=good)
            if good:
                cs = slope(xx,cc[m]); l1=np.abs(cs).sum()
                row.update(median_total=float(np.median(yy)),delta5s=float(np.median(yy[-5:])-np.median(yy[:5])),
                           ols_adc_min=float(slope(xx,yy)),sen_adc_min=sen(xx,yy),
                           mean_ols_adc_min=float(slope(xx,bb.loc[m,'mean'].to_numpy())),
                           per_channel_ols_adc_min=float(slope(xx,yy)/raw.shape[1]),
                           channel_slope_sum=float(cs.sum()),channel_slope_l1=float(l1),
                           cancellation=float(1-abs(cs.sum())/l1) if l1 else 0,
                           effective_channels=float(l1*l1/(cs@cs)) if cs@cs else 0,
                           top5_share=float(np.sort(np.abs(cs))[-5:].sum()/l1) if l1 else 0,
                           positive_channels_gt1=int(np.sum(cs>1)),negative_channels_lt_minus1=int(np.sum(cs<-1)),
                           median_abs_adjacent_1s=float(np.median(np.abs(np.diff(yy)))))
            rows.append(row)
    jumps = np.argsort(np.abs(np.diff(b['median'].to_numpy())))[-10:][::-1]
    jump_rows = [dict(id=sid,start_s=float(b.t.iloc[i]),end_s=float(b.t.iloc[i+1]),
                      delta=float(b['median'].iloc[i+1]-b['median'].iloc[i])) for i in jumps]
    print(sid, json.dumps({k:s[k] for k in ['frames','channels','duration_s','max_gap_s','difference_lag1_correlation','adjacent_difference_abs_median','q99_abs_mean_minus_median_1s']},ensure_ascii=False),flush=True)
    return s, rows, phase, jump_rows, b


def draw(curves):
    im = Image.new('RGB',(1600,1400),'#f8fafc'); d=ImageDraw.Draw(im)
    f=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    title=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',24)
    d.text((25,10),'RAW ADC | 1-second frame-total median (blue) and mean (orange)',font=title,fill='#17273b')
    d.text((25,43),'Independent axes; descriptive windows only; no force/load labels inferred',font=f,fill='#526071')
    for k,(sid,b) in enumerate(curves):
        col=k%2; row=k//2; l=20+col*795; top=85+row*325
        x0,x1,y0,y1=l+80,l+760,top+30,top+260
        low=min(b['median'].min(),b['mean'].min()); high=max(b['median'].max(),b['mean'].max()); span=max(high-low,1)
        d.text((l+5,top),sid+' | total ADC',font=f,fill='#17273b')
        for q in [0,.5,1]:
            yy=y1-q*(y1-y0);d.line((x0,yy,x1,yy),fill='#dce3eb');d.text((l,yy-8),f'{low+q*span:.0f}',font=f,fill='#526071')
        for name,color in [('mean','#d57831'),('median','#177eaa')]:
            pts=[(x0+t/max(b.t.iloc[-1],1)*(x1-x0),y1-(v-low)/span*(y1-y0)) for t,v in zip(b.t,b[name])]
            d.line(pts,fill=color,width=2)
        d.text((x0,y1+8),'0 s',font=f,fill='#526071');d.text((x1-95,y1+8),f'{b.t.iloc[-1]:.0f} s',font=f,fill='#526071')
    im.save(OUT/'raw_overview.png')


def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'07_原始ADC跨传感器画像_20261020/manifest.json').read_text(encoding='utf-8'))
    sources=[(sid,ROOT/'data'/rel/'device_001_raw_seg000.csv') for sid,group,rel in manifest['sessions'] if sid.startswith(('NP','NT','NF'))]
    sources += [('O19A',Path('D:/Temp/拇指指腹-2026-10-19_16-33-49/device_001_raw_seg000.csv')),('O19B',Path('D:/Temp/拇指指腹-2026-10-19_17-22-01/device_001_raw_seg000.csv'))]
    summary=[];windows=[];phases=[];jumps=[];curves=[]
    for sid,path in sources:
        s,w,p,j,b=analyze(sid,path);summary.append(s);windows+=w;phases+=p;jumps+=j;curves.append((sid,b))
    (OUT/'provenance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in s.items() if k not in ['header','targets']} for s in summary]).to_csv(OUT/'sessions.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'windows.csv',index=False)
    pd.DataFrame(phases).to_csv(OUT/'parity_60s.csv',index=False)
    pd.DataFrame(jumps).to_csv(OUT/'largest_1s_changes.csv',index=False)
    draw(curves)
    print(pd.DataFrame(windows).query("axis=='timestamp'").round(3).to_string(index=False))


if __name__=='__main__':
    main()
