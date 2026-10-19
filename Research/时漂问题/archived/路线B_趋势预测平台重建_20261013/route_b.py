# -*- coding: utf-8 -*-
"""路线B在线因果观测器：K9逐帧状态 + 可回滚平台保护。"""
from __future__ import annotations
import json
import sys
from collections import deque
from dataclasses import fields
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
K9_IMPL = ROOT / "01_当前工作_K9" / "02_现场调参" / "09-21_09-26_工程A_v3.4"
TOOLBOX = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
sys.path.insert(0, str(K9_IMPL)); sys.path.insert(0, str(TOOLBOX))
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402
from dptool.session_csv import load_table  # noqa: E402

CONFIG = dict(enabled=True, window_s=.25, separation_s=.50, confirm_s=.25,
              hold_s=2.0, threshold_adc=45., noise_multiplier=4., trend_tau_s=8.,
              cooldown_s=1.5, large_step_adc=2000., max_event_adc=250.,
              max_platform_adc=800., max_platform_rate=500., gap_s=.25,
              rollback_tolerance=.35)


def _params(raw, r_slow_max=.4):
    names={f.name for f in fields(Params)}
    p={k: float(v) for k,v in raw.items() if k in names}
    p["r_slow_max"] = r_slow_max
    return Params(**p)


class OnlineObserver:
    """每帧推进一次K9；事件账本只保存有限平台向量。"""
    def __init__(self, params, cfg=None):
        self.cfg=dict(CONFIG); self.cfg.update(cfg or {})
        self.params=params if isinstance(params, Params) else _params(params, float(params.get('r_slow_max', .4)))
        self.k9=CreepObserverK9(self.params)
        self.k9._trace_frame = lambda *args: None
        self.n=0; self.last_t=None; self.last_raw=None
        self.state="UNCERTAIN"; self.candidate_since=None; self.candidate_anchor=None
        self.hold_until=-np.inf; self.cooldown_until=-np.inf; self.trend=0.; self.sigma=0.
        self.recent=deque(); self.delayed=deque(); self.noise=deque(maxlen=500)
        self.platforms=[]; self.events=deque(maxlen=4096); self.diag=deque(maxlen=2048)
        self.compensation_history = deque()
        self.candidate_compensation = None
        self.last_output = None
        self.state_fields = ('x_fast', 'x_slow', 'applied', 'zero', 'v_lp', 'v_fast_lp', 'y_floor', 'y_max', 'load_dwell', 'ramp_dwell', 't_edge', 'total_baseline', 'total_noise')

    @property
    def ledger(self):
        if not self.platforms: return np.zeros(self.n)
        return np.sum([p["vector"] for p in self.platforms], axis=0)

    def _reset_uncertain(self, stamp, row):
        self.state="UNCERTAIN"; self.candidate_since=None; self.candidate_anchor=None
        self.recent.clear(); self.delayed.clear(); self.noise.clear(); self.trend=0.; self.sigma=0.
        self.hold_until=stamp+8.; self.cooldown_until=stamp+8.
        self.platforms.clear(); self._reanchor(row)

    def _reanchor(self, adjusted):
        if not self.k9._first:
            self.k9.v_lp=adjusted.copy(); self.k9.v_fast_lp=adjusted.copy()
            y=adjusted-self.k9.zero
            self.k9.y_floor=np.minimum(self.k9.y_floor, y)
            self.k9.y_max=np.maximum(self.k9.y_max, np.maximum(y,0.))
            self.k9.applied=self.k9.x_fast+self.k9.x_slow

    def _rollback(self, delta):
        remain=-delta.copy()
        for p in reversed(self.platforms):
            v=p["vector"]; overlap=np.minimum(np.maximum(v,0),np.maximum(remain,0))+np.minimum(np.minimum(v,0),np.minimum(remain,0))
            if np.sum(np.abs(overlap)) <= 0: continue
            p["vector"]-=overlap; remain-=overlap
            if np.sum(np.abs(p['vector'])) < 1.:
                self.platforms = [item for item in self.platforms if item is not p]
            if np.sum(np.abs(remain)) < 1.: break

    def _commit(self, vector, stamp, direction):
        total=float(np.sum(vector)); vector=np.asarray(vector,float)
        if direction < 0 and np.any(self.ledger > 0):
            before = self.ledger.copy()
            self._rollback(vector)
            unmatched = vector - (self.ledger - before)
            if float(np.sum(np.abs(unmatched))) > 1.0:
                room = max(0., self.cfg['max_platform_adc'] - float(np.sum(np.abs(self.ledger))))
                unmatched *= min(1., room / max(float(np.sum(np.abs(unmatched))), 1e-9))
                self.platforms.append({'vector': unmatched.copy(), 'time_s': stamp})
            action="rollback"
        else:
            vector=np.clip(vector, -self.cfg["max_event_adc"], self.cfg["max_event_adc"])
            room=max(0., self.cfg["max_platform_adc"]-float(np.sum(np.abs(self.ledger))))
            scale=min(1., room/max(float(np.sum(np.abs(vector))),1e-9))
            vector*=scale; rate=float(np.sum(np.abs(vector)))/max(self.cfg["confirm_s"],1e-6)
            if rate>self.cfg["max_platform_rate"]: vector*=self.cfg["max_platform_rate"]/rate
            if np.sum(np.abs(vector))<1.: return
            self.platforms.append({"vector":vector.copy(),"time_s":stamp})
            action="lock"
        self.events.append(dict(time_s=stamp, action=action, delta_total_adc=float(np.sum(vector)),
                                platform_total_adc=float(np.sum(self.ledger))))
        self.state="HOLD"; self.hold_until=stamp+self.cfg["hold_s"]; self.cooldown_until=stamp+self.cfg["cooldown_s"]

    def process(self, stamp, row):
        row=np.asarray(row,float).copy(); stamp=float(stamp)
        if not self.cfg.get("enabled", True):
            return self.k9.process(stamp, row)
        if self.n==0: self.n=row.size; self.last_t=stamp; self.last_raw=row.copy(); self.state="STABLE"
        if not np.isfinite(stamp) or not np.isfinite(row).all():
            self.state='UNCERTAIN'
            return self.last_output.copy() if self.last_output is not None else np.nan_to_num(row)
        if row.size != self.n:
            self.__init__(self.params, self.cfg)
            return self.process(stamp, row)
        dt=stamp-self.last_t if self.last_t is not None else 0.
        if dt <= 0 and self.last_output is not None:
            return self.last_output.copy()
        if dt>self.cfg['gap_s']:
            self._reset_uncertain(stamp,row); dt=0.
        if self.last_raw is not None and abs(float(np.sum(row-self.last_raw)))>=self.cfg["large_step_adc"]:
            self._reset_uncertain(stamp,row)
        self.last_t=stamp; self.last_raw=row.copy()
        self.recent.append((stamp,row.copy()))
        while self.recent and self.recent[0][0]<stamp-self.cfg["window_s"]: self.recent.popleft()
        short=np.mean(np.stack([v for _,v in self.recent]),axis=0)
        if self.state == 'UNCERTAIN' and stamp >= self.hold_until:
            self.state = 'STABLE'
        self.delayed.append((stamp,short.copy()))
        while len(self.delayed)>2 and self.delayed[1][0]<=stamp-self.cfg["separation_s"]: self.delayed.popleft()
        ref=self.delayed[0][1] if self.delayed and self.delayed[0][0]<=stamp-self.cfg["separation_s"] else None
        if ref is not None and dt > 0 and self.state == 'UNCERTAIN':
            rate = float(np.sum(short-ref))/self.cfg['separation_s']
            self.trend += (1-np.exp(-dt/self.cfg['trend_tau_s']))*(rate-self.trend)
        if ref is not None and dt>0 and self.state!="UNCERTAIN":
            delta=short-ref; total=float(np.sum(delta)); rate=total/self.cfg["separation_s"]
            if self.state not in ("CANDIDATE","HOLD"):
                a=1-np.exp(-dt/self.cfg["trend_tau_s"]); self.trend+=a*(rate-self.trend)
                self.noise.append(total-self.cfg["separation_s"]*self.trend)
                z=np.asarray(self.noise); self.sigma=float(np.median(np.abs(z-np.median(z)))/.6745) if z.size>10 else 0.
            threshold=max(self.cfg["threshold_adc"],self.cfg["noise_multiplier"]*self.sigma)
            innovation = total-self.cfg['separation_s']*self.trend
            reverse = float(np.sum(self.ledger))*innovation < 0
            available = (stamp>=self.hold_until and stamp>=self.cooldown_until) or reverse
            active = abs(innovation)>=threshold and available
            if self.state in ('STABLE', 'HOLD') and active:
                self.state='CANDIDATE'; self.candidate_since=stamp; self.candidate_anchor=ref.copy()
                self.candidate_direction = np.sign(innovation)
                self.candidate_compensation = None
                for history_t, snapshot in self.compensation_history:
                    if history_t <= stamp-self.cfg['separation_s']:
                        self.candidate_compensation = snapshot
                    else:
                        break
            elif self.state=='CANDIDATE':
                if not active or innovation*self.candidate_direction <= 0:
                    self.state='STABLE'; self.candidate_since=None
                elif stamp-self.candidate_since>=self.cfg['confirm_s']:
                    vec=short-self.candidate_anchor
                    direction=1 if float(np.sum(vec))>=0 else -1
                    if direction == self.candidate_direction:
                        self._commit(vec,stamp,direction)
                        if self.candidate_compensation is not None:
                            for name, value in self.candidate_compensation.items():
                                setattr(self.k9, name, value.copy() if isinstance(value, np.ndarray) else value)
                        self._reanchor(row-self.ledger)
                    else:
                        self.state='STABLE'
            elif self.state=='HOLD' and stamp>=self.hold_until:
                self.state='STABLE'
        adjusted=row-self.ledger
        out=self.k9.process(stamp,adjusted)+self.ledger
        snapshot = {name: getattr(self.k9, name).copy() if isinstance(getattr(self.k9, name), np.ndarray) else getattr(self.k9, name) for name in self.state_fields}
        self.compensation_history.append((stamp, snapshot))
        while self.compensation_history and self.compensation_history[0][0] < stamp-2.0:
            self.compensation_history.popleft()
        self.diag.append(dict(time_s=stamp,state=self.state,ledger_total_adc=float(np.sum(self.ledger)),trend=self.trend,sigma=self.sigma))
        self.last_output = out.copy()
        return out


def route_b(values,time,params,cfg=None):
    obs=OnlineObserver(params,cfg); out=np.empty_like(values)
    for i,(t,row) in enumerate(zip(time,values)): out[i]=obs.process(t,row)
    return out,dict(events=list(obs.events),diag=list(obs.diag),ledger_final=float(np.sum(obs.ledger)))


def disabled_equivalent(values,time,params):
    obs=OnlineObserver(params,{"enabled":False}); out=np.empty_like(values)
    for i,(t,row) in enumerate(zip(time,values)):
        if i==0: out[i]=obs.k9.process(float(t),row)
        else: out[i]=obs.k9.process(float(t),row)
    return out


def load_folder(folder):
    session=json.loads((folder/'session.json').read_text(encoding='utf-8')); fps=float(session['devices'][0]['observed_fps'])
    pre=load_table(str(folder/'device_001_pre_seg0.csv'),use_cache=False); rec=load_table(str(folder/'device_001_seg000.csv'),use_cache=False)
    frame=np.asarray(pre.data[:,2],float); time=(frame-frame[0])/fps; values=pre.channels().astype(float); recorded=rec.channels().sum(1)
    total=values.sum(1); jumps=np.flatnonzero(np.abs(np.diff(total))>=2000)
    return session,values,time,recorded,float(time[jumps[0]+1])


def metrics(time,signal,load):
    start=load+15.; first=np.median(signal[(time>=start)&(time<start+5)]); last=np.median(signal[time>=time[-1]-5]); offsets=[]
    for s in np.arange(start,time[-1]-4,1.):
        z=signal[(time>=s)&(time<s+5)];
        if z.size: offsets.append(np.median(z)-first)
    return dict(net_change_adc=float(last-first),max_abs_offset_adc=float(np.max(np.abs(offsets))))

if __name__=="__main__":
    rows=[]
    for folder in sorted((ROOT/'data'/'右拇指指腹').glob('*/*')):
        if not (folder/'session.json').exists(): continue
        session,values,time,recorded,load=load_folder(folder)
        params = dict(session['algorithm']['params'], r_slow_max=.4)
        out,trace=route_b(values,time,params)
        rows.append(dict(id=f'{folder.parent.name}/{folder.name}',kind='10g' if '10g' in folder.name else 'constant_load',load_time_s=load,metrics=metrics(time,out.sum(1),load),events=trace['events']))
    (HERE/'route_b_results.json').write_text(json.dumps({'config':CONFIG,'samples':rows},ensure_ascii=False,indent=2),encoding='utf-8')
