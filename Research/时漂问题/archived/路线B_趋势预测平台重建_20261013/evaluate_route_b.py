# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from route_b import route_b, CONFIG
from validation_cases import generate_cases
from plugins.k9_creep_observer import _compensate

HERE = Path(__file__).resolve().parent
PARAMS = dict(r_fast=.03, tau_c_fast_s=2, tau_r_fast_s=.1, r_slow_max=.4,
              slope_cap_frac=.02, slow_confirm_s=3, soft_unfreeze_s=2,
              tau_r_slow_idle_s=.2)

def k9(values, time):
    out, _ = _compensate(values, time, PARAMS)
    return out.sum(axis=1)

def evaluate(case):
    t = case['time']; x = case['values']; truth = case['truth_total']
    routed_channels, trace = route_b(x, t, PARAMS)
    routed = routed_channels.sum(axis=1)
    base = k9(x, t)
    baseline_total = float(np.median(routed[t < 2.0]))
    result = {'name': case['name'], 'kind': case['kind'], 'events': len(trace['events']),
              'event_times_s': [round(e['time_s'], 4) for e in trace['events']]}
    if case['name'].startswith('positive_100adc'):
        injected = (t >= 15) & (t < 30)
        hold = (t >= 25) & (t < 30)
        after = (t >= 30) & (t < 35)
        delta = routed - baseline_total
        result.update(truth_adc=100.0, hold_delta_adc=float(np.median(delta[hold])),
                      hold_retention=float(np.median(delta[hold]) / 100.0),
                      withdrawal_residual_adc=float(np.median(delta[after])))
    elif case['kind'] == 'positive_repeated':
        result['cycle_outputs_adc'] = []
        for start, end in ((15,20),(35,40),(55,60),(75,80),(95,100),(110,115)):
            mask=(t>=start+4)&(t<end)
            result['cycle_outputs_adc'].append(float(np.median(routed[mask])))
    elif case['kind'].startswith('negative_'):
        result['negative_event_count']=len(trace['events'])
    return result

def main():
    results={'config':CONFIG,'params':PARAMS,'cases':[evaluate(c) for c in generate_cases()],
             'notes':['合成案例有真值；共模100 ADC与分散100 ADC刻意同总量，不能要求算法仅凭单路数据区分二者。',
                      '真实录制10g仍无操作标签，不能由本报告推出真实检出率。']}
    (HERE/'route_b_synthetic_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in results['cases']:
        if 'hold_retention' in row or row['kind'].startswith('negative_'):
            print(row)

if __name__=='__main__': main()
