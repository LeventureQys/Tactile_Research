"""Research K12 v2 plugin; select this plugin directory in dptool."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from dptool.algorithms import ParamSpec, AlgorithmOutput, AlgorithmResult, Cancelled

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from k12_v2_core import compensate, DEFAULTS
BASE_PATH = Path(r'D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具\plugins\k9_creep_observer.py')
spec = importlib.util.spec_from_file_location('k12_base_k9', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PLUGIN_META = dict(id='k12_motion_ledger', name='K12 v2 Motion Ledger', version='2.0.0',
                   protocol=1, input_mode='dataset', time_modes=['frame_fps','elapsed','timestamp'],
                   description='Experimental total-domain event ledger; unchanged K9 background core.')


def param_specs():
    extra = [ParamSpec(name='ledger_enabled', label='Event ledger enabled', type='bool', default=True)]
    for name, default in DEFAULTS.items():
        if name == 'enabled':
            continue
        extra.append(ParamSpec(name='ledger_'+name, label=name, type='float', default=default,
                               min=.001, max=100000., step=.01, group='K12 v2'))
    return tuple(base.param_specs()) + tuple(extra)


class Algorithm:
    def run(self, ctx, params, progress=None, cancel=None):
        if cancel and cancel():
            raise Cancelled()
        values = np.asarray(ctx.values, dtype=float)
        time = np.asarray(ctx.time_axis, dtype=float)
        if values.ndim != 2 or len(time) != len(values):
            raise ValueError('Time/channel shape mismatch')
        cfg = {name: params.get('ledger_'+name, default) for name, default in DEFAULTS.items()}
        cfg['enabled'] = params.get('ledger_enabled', True)
        output, trace = compensate(values, time, params, base._compensate, cfg)
        if cancel and cancel():
            raise Cancelled()
        if progress:
            progress(1., 'Complete')
        items = [AlgorithmOutput(key='total', label='K12 v2 total', y=output.sum(1), unit='ADC')]
        ids = ctx.channel_ids or tuple(range(values.shape[1]))
        items.extend(AlgorithmOutput(key=f'ch{channel}', label=f'K12 v2 ch{channel}',
                                     y=output[:, index], unit='ADC') for index, channel in enumerate(ids))
        return AlgorithmResult(outputs=items, note=f"Candidate events: {len(trace['events'])}",
                               params_used=dict(params))
