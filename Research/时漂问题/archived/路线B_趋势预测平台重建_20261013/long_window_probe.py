import json
from pathlib import Path
import numpy as np
from route_b import route_b, load_folder, ROOT, CONFIG, metrics
from check_contract import PARAMS

HERE=Path(__file__).resolve().parent

def main():
    time=np.arange(6000)/100
    reference=np.full((len(time),57),8.)
    reference[time>=2]+=500
    rows=[]
    for separation in (1.,2.,3.):
        cfg=dict(separation_s=separation)
        background,_=route_b(reference,time,PARAMS,cfg)
        ramp_results=[]
        for duration in (1.,3.,5.):
            changed=reference+np.clip((time-30)/duration,0,1)[:,None]*100/57
            out,trace=route_b(changed,time,PARAMS,cfg)
            signal=(out-background).sum(1)
            mask=(time>=30+duration+10)&(time<30+duration+15)
            ramp_results.append(dict(duration_s=duration,hold_adc=float(np.median(signal[mask])),events=len(trace['events'])))
        drift_results=[]
        for rate in (5.7,20.):
            drifted=reference+np.maximum(time-15,0)[:,None]*rate/57
            _,trace=route_b(drifted,time,PARAMS,cfg)
            drift_results.append(dict(rate=rate,events=len(trace['events']),ledger=trace['ledger_final']))
        rows.append(dict(separation_s=separation,ramps=ramp_results,drift=drift_results))
        (HERE/'long_window_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(rows[-1]),flush=True)

if __name__=='__main__':main()
