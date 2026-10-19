import json
from pathlib import Path
import numpy as np
from route_b import route_b, OnlineObserver, CONFIG
from creep_observer_k9 import CreepObserverK9

HERE = Path(__file__).resolve().parent
PARAMS = dict(r_fast=.03,tau_c_fast_s=2,tau_r_fast_s=.1,r_slow_max=.4,slope_cap_frac=.02,slow_confirm_s=3,soft_unfreeze_s=2,tau_r_slow_idle_s=.2)

def main():
    time=np.arange(6000)/100
    reference=np.full((len(time),57),8.)
    reference[time>=2]+=500
    values=reference.copy()
    values[(time>=30)&(time<45)]+=100/57
    output,trace=route_b(values,time,PARAMS)
    prefix,_=route_b(values[:3700],time[:3700],PARAMS)
    disabled,_=route_b(values,time,PARAMS,dict(enabled=False))
    observer=OnlineObserver(PARAMS,dict(enabled=False))
    direct=np.array([observer.k9.process(t,row) for t,row in zip(time,values)])
    anomaly=OnlineObserver(PARAMS)
    for stamp,row in zip(time[:3100],values[:3100]): last=anomaly.process(stamp,row)
    duplicate=anomaly.process(time[3099],values[3099]+10)
    invalid=anomaly.process(time[3100],np.full(57,np.nan))
    base,_=route_b(reference,time,PARAMS)
    ramps=[]
    for duration in (.2,1.,3.,5.):
        fraction=np.clip((time-30)/duration,0,1)
        changed=reference+fraction[:,None]*100/57
        processed,diagnostics=route_b(changed,time,PARAMS)
        delta=(processed-base).sum(1)
        mask=(time>=30+duration+10)&(time<30+duration+15)
        ramps.append(dict(duration_s=duration,hold_adc=float(np.median(delta[mask])),event_count=len(diagnostics['events'])))
    result=dict(config=CONFIG,params=PARAMS,prefix_max_abs=float(np.max(np.abs(prefix-output[:3700]))),disabled_same_engine_max_abs=float(np.max(np.abs(disabled-direct))),duplicate_change=float(np.max(np.abs(duplicate-last))),nan_change=float(np.max(np.abs(invalid-last))),ramps=ramps)
    (HERE/'contract_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
