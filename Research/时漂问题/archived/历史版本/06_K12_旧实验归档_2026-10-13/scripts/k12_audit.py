from pathlib import Path
import json
import sys
import numpy as np

TOOLBOX = Path(r'D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具')
sys.path.insert(0, str(TOOLBOX))
from dptool import algorithms as alg
from dptool import session_csv as scsv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / 'data' / '\u53f3\u62c7\u6307\u6307\u8179'
registry = alg.PluginRegistry()
registry.scan()
k9 = registry.get('k9_creep_observer')
k12 = registry.get('k12_motion_guard')
params = alg.default_params(k9.specs)
params.update(r_fast=.03, tau_c_fast_s=2, tau_r_fast_s=.1, r_slow_max=.1,
              slope_cap_frac=.02, slow_confirm_s=3, soft_unfreeze_s=2,
              tau_r_slow_idle_s=.2)
report = {'synthetic': [], 'recording_parity': []}

# The same total step is tested with concentrated and distributed footprints.
time = np.arange(6000) / 100
for footprint in (1, 57):
    reference = np.full((len(time), 57), 8.0)
    reference[time >= 2] += 500
    perturbed = reference.copy()
    perturbed[(time >= 30) & (time < 45), :footprint] += 100 / footprint
    cfg = dict(params, motion_guard_enable=True, motion_thresh_adc=60,
               motion_slow_gain=0)
    out, stats = k12.module._compensate(perturbed, time, cfg)
    ref_out, _ = k12.module._compensate(reference, time, cfg)
    off, _ = k9.module._compensate(perturbed, time, params)
    ref_off, _ = k9.module._compensate(reference, time, params)
    band = (time >= 30) & (time < 40)
    row = {'footprint_channels': footprint, 'total_step_adc': 100,
           'per_channel_step_adc': 100 / footprint,
           'guard_fraction_30_40s': float(stats['guard_mask'][band].mean()),
           'k9_response_31_33s': float(np.median((off-ref_off).sum(1)[(time>=31)&(time<33)])),
           'k9_response_40_44s': float(np.median((off-ref_off).sum(1)[(time>=40)&(time<44)])),
           'k12_response_40_44s': float(np.median((out-ref_out).sum(1)[(time>=40)&(time<44)]))}
    report['synthetic'].append(row)
    print(json.dumps(row), flush=True)

for manifest in sorted(DATA.glob('*/*/session.json')):
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    folder = manifest.parent
    pre = scsv.load_table(str(folder/'device_001_pre_seg0.csv'), use_cache=False)
    recorded = scsv.load_table(str(folder/'device_001_seg000.csv'), use_cache=False)
    if pre.n_rows != recorded.n_rows:
        raise ValueError('Frame count mismatch')
    frames = np.asarray(pre.data[:, 2], dtype=float)
    if not np.array_equal(frames, recorded.data[:, 2]):
        raise ValueError('Frame id mismatch')
    values = pre.channels().astype(float)
    expected = recorded.channels().astype(float)
    cfg = dict(params)
    cfg.update({key: val for key, val in meta['algorithm']['params'].items() if key in cfg})
    axes = {'frame_fps': frames / meta['devices'][0]['observed_fps'],
            'timestamp': np.asarray(pre.data[:, 0], dtype=float)}
    for name, axis in axes.items():
        output, _ = k9.module._compensate(values, axis, cfg)
        error = output - expected
        total_error = error.sum(1)
        row = {'device': folder.parent.name, 'session': folder.name, 'time_mode': name,
               'channel_rmse_adc': float(np.sqrt(np.mean(error**2))),
               'channel_max_abs_adc': float(np.max(np.abs(error))),
               'total_rmse_adc': float(np.sqrt(np.mean(total_error**2))),
               'total_max_abs_adc': float(np.max(np.abs(total_error)))}
        report['recording_parity'].append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

(ROOT/'results'/'k12_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
