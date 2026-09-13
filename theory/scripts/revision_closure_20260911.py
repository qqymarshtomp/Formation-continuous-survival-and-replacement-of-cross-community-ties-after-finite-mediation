import json
import sys
import time
from pathlib import Path
import numpy as np

origin=Path(__file__).resolve().parents[1]/'inputs/original'
sys.path.insert(0,str(origin/'reproducibility'))
from model import NetworkModel,response_probability
from network_forecast import OpinionExpectation,FORECAST_METRICS

out=Path(__file__).resolve().parents[1]/'outputs/manuscript_restructured_20260911/closure'
out.mkdir(parents=True,exist_ok=True)
a=np.load(origin/'results/forecast_confirm.npz',allow_pickle=False)
tasks=json.loads(a['metadata'].item())
partition=int(sys.argv[1])
selected=list(range(partition,len(tasks),4))
started=time.perf_counter()
for task_index in selected:
    task=tasks[task_index]
    state=NetworkModel.load(origin/'results/forecast_exits'/f'confirm_{task["variant"]}_{task["seed"]}_{task["condition"]}.npz')
    c=state.cfg
    opinion_operator=OpinionExpectation(c.n,state.i,state.j)
    results=[]
    for mode in ['mean_trust','factorized']:
        rng=np.random.default_rng(np.random.SeedSequence([state.seed,19701]))
        w=np.repeat(state.w[:,None],64,axis=1)
        r=state.r[:,None].copy() if mode=='mean_trust' else np.repeat(state.r[:,None],64,axis=1)
        x=state.x.copy()
        cohort=state.w[state.cx]>0
        alive=np.repeat(cohort[:,None],64,axis=1)
        result=np.empty((401,len(FORECAST_METRICS)))
        for tick in range(401):
            distance=np.abs(x[state.i]-x[state.j])[:,None]
            p=response_probability(c.intercept+c.trust_effect*r-c.distance_effect*distance,c.response)
            lam=c.natural_rate/c.opportunity_degree*(c.exploration+(1-c.exploration)*w)
            mean_lam=lam.mean(axis=1,keepdims=True)
            mean_p=p.mean(axis=1,keepdims=True)
            result[tick]=[100+tick,2/c.n*np.mean(w[state.cx]>0,axis=1).sum(),2/c.n*np.mean(w[state.cx],axis=1).sum(),r[state.cx].mean(),x[state.groups==1].mean()-x[state.groups==0].mean(),p[state.cx].mean(),alive.sum()/(cohort.sum()*64)]
            if tick==400:
                break
            dx=opinion_operator.increment(x,(mean_lam*mean_p).ravel())
            draws=rng.random((3,state.m,64))
            success=draws[0]<lam*mean_p
            birth=success & (w==0)
            reinforce=success & (w>0)
            w[birth]=c.birth_weight
            w[reinforce]+=c.reinforcement*(1-w[reinforce])
            if mode=='mean_trust':
                r+=c.trust_learning*mean_lam*(p-r)
            else:
                attempted=draws[1]<mean_lam
                accepted=draws[2]<p
                r+=c.trust_learning*attempted*(accepted-r)
            r+=c.trust_relaxation*(state.r0[:,None]-r)
            x+=c.opinion_learning*dx
            x+=c.opinion_anchoring*(state.anchor-x)
            w*=1-c.decay
            w[w<c.threshold]=0
            alive &= w[state.cx]>0
        results.append(result)
    np.savez_compressed(out/f'exit_{task_index:03d}.npz',predictions=np.stack(results),modes=np.array(['mean_trust','factorized']),task=json.dumps(task),metrics=json.dumps(FORECAST_METRICS))
    print(partition,task_index,task['variant'],task['condition'],f'{time.perf_counter()-started:.1f}s',flush=True)
print('COMPLETED',partition,len(selected),'exit states',flush=True)
