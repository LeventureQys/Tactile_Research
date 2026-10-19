import json
from pathlib import Path
import numpy as np
from route_b import route_b, load_folder, ROOT, CONFIG

HERE = Path(__file__).resolve().parent

def probe(values, time, params, starts, footprint):
    base, _ = route_b(values, time, params)
    changed = values.copy()
    for start in starts:
        changed[(time >= start) & (time < start + 15), :footprint] += 100 / footprint
    result, trace = route_b(changed, time, params)
    delta = (result - base).sum(1)
    events = []
    for start in starts:
        events.append(dict(start=start, hold_adc=float(np.median(delta[(time >= start+10)&(time < start+15)])), release_adc=float(np.median(delta[(time >= start+22)&(time < start+27)]))))
    return dict(footprint=footprint, measurements=events, detected=trace['events'])

def main():
    folder = ROOT/'data'/'右拇指指腹'/'设备1'/'测试2-2026-10-13_16-43-59'
    session, real, rt, _, load = load_folder(folder)
    params = dict(session['algorithm']['params'], r_slow_max=.4)
    time = np.arange(9000)/100
    base = np.full((len(time),57),8.)
    base[time>=2] += 500
    rows = []
    for footprint in (1,19,57):
        rows.append(dict(kind='synthetic', **probe(base,time,params,[30.,60.],footprint)))
    rows.append(dict(kind='real_background', **probe(real,rt,params,[load+30.,float(rt[-1])-30.],57)))
    output = dict(config=CONFIG,params=params,cases=rows)
    (HERE/'final_probe_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=True),flush=True)

if __name__=='__main__': main()
