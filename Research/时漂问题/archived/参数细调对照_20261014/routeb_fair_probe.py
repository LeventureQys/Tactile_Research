import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '路线B_趋势预测平台重建_20261013'
sys.path.insert(0, str(PREVIOUS))
from route_b import route_b, load_folder, ROOT, OnlineObserver, CONFIG


def run_k9(values, time, params):
    observer = OnlineObserver(params, dict(enabled=False))
    return np.array([observer.process(float(stamp), row) for stamp, row in zip(time, values)])


def main():
    folder = ROOT / 'data' / '右拇指指腹' / '设备1' / '测试2-2026-10-13_16-43-59'
    session, values, time, _, load = load_folder(folder)
    mask = time < 69
    values, time = values[mask], time[mask]
    params = dict(session['algorithm']['params'], r_slow_max=.6)
    backgrounds = {'k9': run_k9(values, time, params), 'route_b': route_b(values, time, params)[0]}
    results = []
    for start in (30., load + 30., 40.):
        for rise in (0., 2.5):
            level = np.where(time >= start, 1., 0.) if rise == 0 else np.clip((time - start) / rise, 0., 1.)
            level[time >= start + 15] = 0.
            injected = values + level[:, None] * 100 / 57
            row = dict(start_s=start, rise_s=rise, arms={})
            for name in backgrounds:
                if name == 'k9':
                    output = run_k9(injected, time, params)
                    events = None
                else:
                    output, trace = route_b(injected, time, params)
                    events = trace['events']
                delta = (output - backgrounds[name]).sum(1)
                hold = (time >= start + 10) & (time < start + 15)
                release = (time >= start + 22) & (time < start + 27)
                row['arms'][name] = dict(hold_adc=float(np.median(delta[hold])), release_adc=float(np.median(delta[release])), events=events)
            results.append(row)
            (HERE / 'routeb_fair_results.json').write_text(json.dumps(dict(params=params, config=CONFIG, probes=results), ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(start=start, rise=rise, arms={name: {key:value for key,value in result.items() if key != 'events'} for name,result in row['arms'].items()})), flush=True)

if __name__ == '__main__':
    main()
