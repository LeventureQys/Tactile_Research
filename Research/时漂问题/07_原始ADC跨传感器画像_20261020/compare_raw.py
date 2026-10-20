from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from analyze_raw import MANIFEST, OUT, draw_panel, font, linear_slope


def main():
    data=np.load(OUT/"raw_1s_channels.npz")
    sessions=pd.read_csv(OUT/"sessions.csv")
    rows=[]
    spatial=[]
    for item in sessions.to_dict("records"):
        sid=item["id"]
        time=data[sid+"_t"]
        total=data[sid+"_total"]
        channels=data[sid+"_channels"]
        reference=item["first_total_adc"]
        amplitude=item["total_q95_adc"]-reference
        crossing=np.flatnonzero(total>=reference+.5*amplitude)
        t0=float(time[crossing[0]]) if len(crossing) and amplitude>50 else np.nan
        row=dict(id=sid,group=item["group"],rise50_s=t0,reference_first_second_adc=reference,
                 observed_high_minus_first_adc=amplitude)
        def sample(age):
            mask=np.abs(time-(t0+age))<=2
            return float(np.median(total[mask])) if mask.sum()>=3 and time[-1]>=t0+age+1 else np.nan
        for age in (10,30,60,120,300,600,900):
            row[f"adc_at_age{age}"]=sample(age)
        if np.isfinite(row["adc_at_age120"]):
            lo,hi=t0+30,t0+120
            mask=(time>=lo)&(time<=hi)
            segment=total[mask]
            x=time[mask]
            row["delta30_120_adc"]=row["adc_at_age120"]-row["adc_at_age30"]
            row["delta30_120_pct_level"]=100*row["delta30_120_adc"]/(row["adc_at_age30"]-reference)
            row["slope30_120_adc_min"]=float(linear_slope(x,segment)*60)
            row["max_1s_jump30_120_adc"]=float(np.abs(np.diff(segment)).max())
            row["major_jump30_120"]=bool(row["max_1s_jump30_120_adc"]>.01*amplitude)
            channel_segment=channels[mask]
            chan_delta=np.median(channel_segment[-5:],axis=0)-np.median(channel_segment[:5],axis=0)
            load=np.median(channel_segment[:5],axis=0)-channels[0]
            slopes=linear_slope(x,channel_segment)*60
            active=load>max(5,.02*np.max(load))
            row["channel_delta_positive_gt1"]=int(np.sum(chan_delta>1))
            row["channel_delta_negative_ltminus1"]=int(np.sum(chan_delta<-1))
            row["absolute_change_cancellation"]=float(1-abs(chan_delta.sum())/np.abs(chan_delta).sum()) if np.abs(chan_delta).sum()>0 else np.nan
            row["load_slope_corr_active"]=float(np.corrcoef(load[active],slopes[active])[0,1]) if active.sum()>3 and np.std(slopes[active])>0 else np.nan
            for channel,(initial,delta,slope) in enumerate(zip(load,chan_delta,slopes)):
                spatial.append(dict(id=sid,channel=channel,start_s=lo,end_s=hi,initial_relative_adc=initial,delta_adc=delta,slope_adc_min=slope,active=bool(active[channel])))
        for start,end in ((30,60),(90,120),(300,420),(480,600)):
            mask=(time>=t0+start)&(time<t0+end)
            row[f"slope{start}_{end}_adc_min"]=float(linear_slope(time[mask],total[mask])*60) if mask.sum()>=end-start-2 else np.nan
        rows.append(row)
    aligned=pd.DataFrame(rows)
    aligned.to_csv(OUT/"aligned_raw_features.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(spatial).to_csv(OUT/"aligned_channel_features.csv",index=False,encoding="utf-8-sig")
    print(aligned[["id","rise50_s","observed_high_minus_first_adc","delta30_120_adc","delta30_120_pct_level","slope30_60_adc_min","slope90_120_adc_min","slope480_600_adc_min","major_jump30_120","channel_delta_positive_gt1","channel_delta_negative_ltminus1","load_slope_corr_active"]].round(3).to_string(index=False))
    quality=sessions[["id","timestamp_nonpositive","timestamp_burst_lt1ms_frac","timestamp_gap_gt100ms","timestamp_max_gap_s","elapsed_repeat_frac","frame_missing"]]
    print("QUALITY",quality.round(4).to_string(index=False))
    image=Image.new("RGB",(1540,930),"#f8fafc")
    draw=ImageDraw.Draw(image)
    draw.text((25,15),"RAW ADC | equal-age comparisons after observed 50% rise",font=font(24),fill="#17273b")
    draw.text((25,48),"1 s medians; first second is a reference, not certified zero load; no force normalization",font=font(16),fill="#526071")
    selections=[("T11","T12","T21","T22"),("NP1","NP2","NT1","NT2","NF1","NF2")]
    colors=["#187d9b","#d17636","#278258","#8958a5","#b44d5c","#555b67"]
    for index,group in enumerate(selections):
        left=25+index*760
        top=100
        right=left+725
        bottom=850
        x0,y0,x1,y1=left+65,top+50,right-20,bottom-45
        draw.text((left,top),"Thumb repeats" if index==0 else "Three locations, two recordings each",font=font(21),fill="#17273b")
        duration=240 if index==0 else 120
        plotted=[]
        for sid in group:
            info=aligned.loc[aligned.id==sid].iloc[0]
            time=data[sid+"_t"]-info.rise50_s
            mask=(time>=10)&(time<=duration)
            raw=data[sid+"_total"]
            anchor=np.median(raw[np.abs(time-30)<=2])
            yy=raw[mask]-anchor
            plotted.append((sid,time[mask],yy))
        lo=min(float(yy.min()) for _,_,yy in plotted)-20
        hi=max(float(yy.max()) for _,_,yy in plotted)+20
        for fraction in (0,.25,.5,.75,1):
            pixel=y1-fraction*(y1-y0)
            draw.line((x0,pixel,x1,pixel),fill="#dbe3eb",width=1)
            draw.text((left,pixel-8),f"{lo+fraction*(hi-lo):.0f}",font=font(14),fill="#526071")
        for number,(sid,tt,yy) in enumerate(plotted):
            points=[(x0+float(t)/duration*(x1-x0),y1-(float(v)-lo)/(hi-lo)*(y1-y0)) for t,v in zip(tt,yy)]
            if len(points)>1:
                draw.line(points,fill=colors[number],width=3)
            draw.text((x0+10+(number%3)*180,y0+8+(number//3)*25),sid,font=font(17),fill=colors[number])
        for value in (0,duration//2,duration):
            xx=x0+value/duration*(x1-x0)
            draw.text((xx-10,y1+12),str(value)+" s",font=font(15),fill="#526071")
        draw.text((left+80,bottom-5),"ADC difference relative to age 30 s",font=font(16),fill="#526071")
    image.save(OUT/"equal_age_raw_comparison.png")
    windows=pd.read_csv(OUT/"windows.csv")
    valid=windows[(windows.duration_s==60)&windows.candidate_high_no_major_jump]
    print("60S CANDIDATE WINDOWS",valid.groupby("id").agg(count=("slope_adc_min","size"),slope_min=("slope_adc_min","min"),slope_max=("slope_adc_min","max"),delta_min=("delta_adc","min"),delta_max=("delta_adc","max")).round(2).to_string())
    facts=dict(session_count=len(sessions),frame_count=int(sessions.frames.sum()),channel_counts=sorted(sessions.channels.unique().tolist()),
               total_duration_s=float(sessions.duration_s.sum()),raw_sha256_count=len(sessions),
               timestamp_nonpositive_total=int(sessions.timestamp_nonpositive.sum()),frame_missing_total=int(sessions.frame_missing.sum()),
               max_gap_s=float(sessions.timestamp_max_gap_s.max()),burst_fraction_min=float(sessions.timestamp_burst_lt1ms_frac.min()),
               burst_fraction_max=float(sessions.timestamp_burst_lt1ms_frac.max()),
               analysis_scope="21 independent raw files; no pre/seg/algoadc/mapforce/forcetotal or prior algorithm reports")
    (OUT/"facts.json").write_text(json.dumps(facts,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    main()
