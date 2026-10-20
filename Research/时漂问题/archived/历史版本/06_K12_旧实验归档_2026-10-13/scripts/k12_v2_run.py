from pathlib import Path
import sys
import json
import numpy as np
from k12_v2_core import compensate, DEFAULTS

TOOLBOX = Path(r'D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具')
sys.path.insert(0, str(TOOLBOX))
from dptool import algorithms as alg, session_csv as scsv
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / 'data' / '\u53f3\u62c7\u6307\u6307\u8179'
OUT = ROOT / 'results' / 'v2'
OUT.mkdir(parents=True, exist_ok=True)
registry = alg.PluginRegistry()
registry.scan()
k9 = registry.get('k9_creep_observer')
params = alg.default_params(k9.specs)
params.update(r_fast=.03, tau_c_fast_s=2, tau_r_fast_s=.1, r_slow_max=.1,
              slope_cap_frac=.02, slow_confirm_s=3, soft_unfreeze_s=2,
              tau_r_slow_idle_s=.2)
core = k9.module._compensate
validation = []
time = np.arange(6000)/100
reference = np.full((len(time),57),8.)
reference[time>=2] += 500
for footprint in (1,57):
    source = reference.copy()
    source[(time>=30)&(time<45),:footprint] += 100/footprint
    base,_ = core(reference,time,params)
    old,_ = core(source,time,params)
    ref_new,_ = compensate(reference,time,params,core)
    new,trace = compensate(source,time,params,core)
    signal = (new-ref_new).sum(1)
    response = (old-base).sum(1)
    item = dict(footprint=footprint, k9_plateau=float(np.median(response[(time>=40)&(time<44)])),
                k12_plateau=float(np.median(signal[(time>=40)&(time<44)])),
                k12_released=float(np.median(signal[(time>=52)&(time<58)])),events=trace['events'])
    disabled,_ = compensate(source,time,params,core,dict(enabled=False))
    assert np.array_equal(disabled,old), 'Disabled parity'
    prefix,_ = compensate(source[:3500],time[:3500],params,core)
    assert np.array_equal(prefix,new[:3500]), 'Prefix causality'
    validation.append(item)
    print(json.dumps({key:val for key,val in item.items() if key!='events'}),flush=True)
    if footprint==57:
        np.savez_compressed(OUT/'synthetic.npz',time=time,reference=reference.sum(1),
                            step=np.where((time>=30)&(time<45),100.,0.),
                            k9_response=response,k12_response=signal,guard=trace['guard'])
# Pure linear drift must not accumulate confirmed events.
drifting=reference+np.maximum(time[:,None]-15,0)*.1
_,trace=compensate(drifting,time,params,core)
validation.append(dict(case='linear_drift_5.7_adc_per_s',events=trace['events']))
summary=[]
for manifest in sorted(DATA.glob('*/*/session.json')):
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    folder=manifest.parent
    pre=scsv.load_table(str(folder/'device_001_pre_seg0.csv'),use_cache=False)
    recorded=scsv.load_table(str(folder/'device_001_seg000.csv'),use_cache=False)
    assert pre.n_rows==recorded.n_rows
    assert np.array_equal(pre.data[:,2],recorded.data[:,2])
    values=pre.channels().astype(float)
    axis=np.asarray(pre.data[:,2],float)/meta['devices'][0]['observed_fps']
    axis-=axis[0]
    cfg=dict(params)
    cfg.update({key:val for key,val in meta['algorithm']['params'].items() if key in cfg})
    replay,_=core(values,axis,cfg)
    tuned,_=core(values,axis,dict(cfg,r_slow_max=.3))
    new,trace=compensate(values,axis,cfg,core)
    total=values.sum(1)
    jumps=np.flatnonzero(np.abs(np.diff(total))>=DEFAULTS['large_step_adc'])
    loaded=float(axis[jumps[0]+1]) if len(jumps) else 0.
    start=loaded+15.
    file=f'{folder.parent.name}_{folder.name}.npz'
    np.savez_compressed(OUT/file,time=axis,pre=total,recorded_k9=recorded.channels().sum(1),
                        replay_k9=replay.sum(1),k9_tuned=tuned.sum(1),k12=new.sum(1),
                        offset_total=trace['offset_total'],guard=trace['guard'],large_guard=trace['large_guard'])
    (OUT/(Path(file).stem+'_events.json')).write_text(json.dumps(trace['events'],ensure_ascii=False,indent=2),encoding='utf-8')
    row=dict(file=file,device=folder.parent.name,session=folder.name,
             loaded_at=loaded,analysis_start=start,events=len(trace['events']),
             final_ledger_adc=float(trace['offset_total'][-1]))
    for label,signal in [('recorded_k9',recorded.channels().sum(1)),('replay_k9',replay.sum(1)),('k12',new.sum(1)),('k9_tuned',tuned.sum(1))]:
        head=(axis>=start)&(axis<start+5)
        tail=axis>=axis[-1]-5
        row[label+'_window_change_adc']=float(np.median(signal[tail])-np.median(signal[head]))
    summary.append(row)
    print(json.dumps(row,ensure_ascii=True),flush=True)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'options.json').write_text(json.dumps(DEFAULTS,indent=2),encoding='utf-8')
