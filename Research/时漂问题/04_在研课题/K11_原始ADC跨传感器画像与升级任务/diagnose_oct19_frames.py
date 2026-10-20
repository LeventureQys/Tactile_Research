import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_oct19_raw import HERE, OUT, read_raw

provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
rows=[]; local=[]
for s in provenance:
    ts,e,f,r,h=read_raw(Path(s['path']))
    t=ts-ts[0]; y=r.sum(axis=1)
    for a,z in [(30,120),(300,420),(1200,1500),(2400,2700),(4800,5100)]:
        if t[-1]<z: continue
        m=(t>=a)&(t<z); yy=y[m]; rr=r[m]; tt=t[m]
        d=np.diff(yy); d2=yy[2:]-yy[:-2]
        n=len(yy)//2; paired=yy[:2*n].reshape(-1,2).mean(axis=1)
        rows.append(dict(id=s['id'],start_s=a,end_s=z,
            median_abs_lag1=float(np.median(abs(d))),median_abs_lag2=float(np.median(abs(d2))),
            diff_lag1_corr=float(np.corrcoef(d[:-1],d[1:])[0,1]),
            reversal_fraction=float(np.mean(d[:-1]*d[1:]<0)),
            median_abs_pairmean_difference=float(np.median(abs(np.diff(paired)))),
            median_abs_channel_lag1=float(np.median(abs(np.diff(rr,axis=0)))),
            median_abs_channel_lag2=float(np.median(abs(rr[2:]-rr[:-2])))))
    for a,z in [(60,62),(1200,1202)]:
        if t[-1]<z: continue
        inds=np.flatnonzero((t>=a)&(t<z))[:30]
        for i in inds: local.append(dict(id=s['id'],t=float(t[i]),frame=int(f[i]),total=float(y[i]),ch0=float(r[i,0]),ch20=float(r[i,20])))
pd.DataFrame(rows).to_csv(OUT/'lag_diagnostics.csv',index=False)
pd.DataFrame(local).to_csv(OUT/'frame_examples.csv',index=False)
w=pd.read_csv(OUT/'windows.csv'); a=w.query("axis=='timestamp' and usable"); b=w.query("axis=='elapsed' and usable")
p=a.merge(b,on=['id','start_s','end_s'],suffixes=('_ts','_elapsed'))
print('CLOCK',float(abs(p.ols_adc_min_ts-p.ols_adc_min_elapsed).max()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print('TOTAL',sum(s['frames'] for s in provenance),sum(s['duration_s'] for s in provenance))
