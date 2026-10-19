# -*- coding: utf-8 -*-
"""严格配对评估路线B；默认synthetic，real需显式指定session JSON。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from route_b import CONFIG, disabled_equivalent, load_folder, route_b
from plugins.k9_creep_observer import _compensate
FPS=100; CHANNELS=57; BASELINE=8.0
PARAM_DEFAULTS=dict(r_fast=.03,tau_c_fast_s=2,tau_r_fast_s=.1,r_slow_max=.4,
                    slope_cap_frac=.02,slow_confirm_s=3,soft_unfreeze_s=2,
                    tau_r_slow_idle_s=.2)
PARAMS=dict(PARAM_DEFAULTS)

def timeline(n_or_seconds):
    return np.arange(0.,float(n_or_seconds),1./FPS) if np.isscalar(n_or_seconds) else np.arange(n_or_seconds)/FPS

def square(t,start,end,total,channels):
    ch=np.asarray(channels,int); out=np.zeros((len(t),CHANNELS)); m=(t>=start)&(t<end)
    out[np.ix_(m,ch)]=total/len(ch); return out

def ramp(t,start,duration,total,channels):
    ch=np.asarray(channels,int); out=np.zeros((len(t),CHANNELS)); f=np.clip((t-start)/duration,0,1)
    out[:,ch]=f[:,None]*total/len(ch); return out

def paired(name,t,load,kind,windows):
    bg=np.full((len(t),CHANNELS),BASELINE)
    bg[t>=2.0] += 500.0
    return dict(name=name,kind=kind,time=t,background=bg,injection=bg+load,
                truth_total=load.sum(1),measure_windows=windows)

def make_cases():
    t=timeline(60.); cases=[]
    for label,ch in {'distributed57':np.arange(57),'concentrated1':np.array([0]),'distributed19':np.arange(0,57,3)}.items():
        x=square(t,30,45,100,ch)
        cases += [paired('positive_100adc_%s_square'%label,t,x,'positive_square',[(30,45)]),
                  paired('negative_100adc_%s_square'%label,t,-x,'negative_square',[(30,45)])]
    x=np.zeros((len(t),CHANNELS))
    for s in (30,35,40): x+=square(t,s,45,100,[0])
    for s in (42,43,44): x+=square(t,s,45,-100,[0])
    cases.append(paired('positive_three_add_then_sequential_remove',t,x,'positive_repeated',[(30,45)]))
    for d in (.2,1.,3.,5.): cases.append(paired('positive_ramp_%gs'%d,t,ramp(t,30,d,100,np.arange(0,57,3)),'positive_ramp',[(30+d,30+d+5)]))
    fixed=np.zeros((len(t),CHANNELS))
    for rate in (5.7,20.): cases.append(paired('negative_linear_%g_adc_s_after15'%rate,t,fixed+np.maximum(t-15,0)[:,None]*rate/CHANNELS,'negative_linear',[(15,20)]))
    for tau in (20.,80.):
        c=1000*(1-np.exp(-np.maximum(t-15,0)/tau))/CHANNELS
        cases.append(paired('negative_exponential_1000_tau%g_after15'%tau,t,fixed+c[:,None],'negative_exponential',[(15,20)]))
    return cases

def median(v,t,a,b):
    z=v[(t>=a)&(t<b)]; return float(np.median(z)) if z.size else None

def direct(values,t): return _compensate(values,t,PARAMS)[0]

def metrics(case,routed,bg,disabled,k9):
    t=case['time']; pair=routed.sum(1)-bg.sum(1); truth=case['truth_total']; a,b=case['measure_windows'][0]
    hs=max(a,b-5); expected=median(truth,t,hs,b); hold=median(pair,t,hs,b)
    n=min(len(t),4000); pref,_=route_b(case['injection'][:n],t[:n],PARAMS)
    return {'hold_delta_adc':hold,'truth_hold_adc':expected,
      'hold_retention':None if expected in (None,0) else hold/expected,
      'withdrawal_residual_7_12s_adc':None if case['kind'] in ('positive_ramp','negative_linear','negative_exponential') else median(pair,t,b+7,b+12),
      'first5s_residual_adc':None if case['kind'] in ('negative_linear','negative_exponential') else median(pair,t,b+.01,b+5),
      'prefix_max_abs_adc':float(np.max(np.abs(pref-routed[:n]))),
      'disabled_vs_direct_k9_max_abs_adc':float(np.max(np.abs(disabled-k9))),
      'direct_k9_pair_max_abs_adc':float(np.max(np.abs(k9.sum(1)-bg.sum(1)))),
      'route_vs_direct_k9_max_abs_adc':float(np.max(np.abs(routed-k9)))}

def evaluate(case):
    t=case['time']; out,tr=route_b(case['injection'],t,PARAMS); bg,_=route_b(case['background'],t,PARAMS)
    dis=disabled_equivalent(case['injection'],t,PARAMS); k9=direct(case['injection'],t)
    return {'name':case['name'],'kind':case['kind'],'event_count':len(tr.get('events',[])),
            'events':tr.get('events',[]),'metrics':metrics(case,out,bg,dis,k9),
            'pair_definition':'injection_output - matched_no_injection_background_output'}

def real_case(session_path):
    folder=Path(session_path).resolve().parent; session,values,t,_,_=load_folder(folder)
    global PARAMS; PARAMS=dict(session.get('algorithm',{}).get('params',PARAM_DEFAULTS)); PARAMS['r_slow_max']=.4
    # Treat the observed recording as the matched background, then add labelled probes.
    bg=values.astype(float); load=np.zeros_like(bg); m=(t>=30)&(t<45); load[np.ix_(m,[0])]=100
    case={'name':'real_background_plus_100adc_probe','kind':'real_paired_probe','time':t,'background':bg,'injection':bg+load,'truth_total':load.sum(1),'measure_windows':[(30,45)]}
    result=evaluate(case); result['session_json']=str(session_path); result['fps']=session.get('devices',[{}])[0].get('observed_fps')
    result['note']='录制本身无外部注入标签；此项仅为真实背景上的合成探针配对，不宣称真实10g验收。'
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('synthetic','real'),default='synthetic'); p.add_argument('--session-json',type=Path); p.add_argument('--output',type=Path,default=HERE/'strict_results.json'); a=p.parse_args()
    global PARAMS; PARAMS=dict(PARAM_DEFAULTS)
    if a.mode=='real':
        if a.session_json is None: raise SystemExit('real mode requires --session-json')
        payload={'schema':'strict-route-b-v1','mode':'real','config':CONFIG,'results':[real_case(a.session_json)],'notes':['保持率只使用真实背景配对探针差值，不使用空载绝对值。','不宣称真实10g验收。']}
    else:
        payload={'schema':'strict-route-b-v1','mode':'synthetic','fps':FPS,'channels':CHANNELS,'baseline_adc_per_channel':BASELINE,'background_500_adc_per_channel_after_2s':True,'params':PARAMS,'config':CONFIG,'results':[evaluate(c) for c in make_cases()],'notes':['所有保持率、撤载残差和前5s残差均为注入输出减匹配无注入背景输出。','合成结果不等同真实10g验收。']}
    a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
