"""Separate particle variation, quadrature sensitivity and conditional-future noise."""
import json
from pathlib import Path
import numpy as np
from model import NetworkModel
from network_forecast import OpinionExpectation

root=Path(__file__).resolve().parents[1]
cases=['high_degree','heterogeneous','strong_opinion']
policies=['none','context'];modes=['coupled','mean_trust'];sizes=[64,128,256]
metrics=json.loads(np.load(root/'inputs/main.npz')['metrics'].item())
bix=metrics.index('b')
rows=[];quadrature=[];contrasts=[];state_rows=[]
draws=np.random.default_rng([20260912,6]).integers(0,32,(6000,32))
particle_draws=np.random.default_rng([20260912,7]).integers(0,4,(6000,4))
future_all=[];pred_all=[]
for case in cases:
    capacity=12 if case=='high_degree' else 6
    future=np.stack([[np.load(root/'results'/f'future_{case}_{p}_{r:02d}.npz')['trajectories']
                      for r in range(32)] for p in policies])
    pred=np.stack([[[[np.load(root/'results'/f'numerics_{case}_{p}_{mode}_{size}_{r}.npz')['prediction']
                      for r in range(1,5)] for size in sizes] for mode in modes] for p in policies])
    future_all.append(future);pred_all.append(pred)
    for history, f, p in [('paired',future[1,:,:,bix]-future[0,:,:,bix],pred[1,:,:,:,:,1]-pred[0,:,:,:,:,1]),
                          ('none',future[0,:,:,bix],pred[0,:,:,:,:,1]),
                          ('context',future[1,:,:,bix],pred[1,:,:,:,:,1])]:
        reference=f.mean(axis=0)
        fse=100/capacity*np.sqrt(np.mean(f[:,1:].var(axis=0,ddof=1)/32))
        for m,mode in enumerate(modes):
            for j,size in enumerate(sizes):
                ensemble=p[m,j].mean(axis=0)
                rmse=100/capacity*np.sqrt(np.mean((ensemble[1:]-reference[1:])**2))
                particle_se=100/capacity*np.sqrt(np.mean(p[m,j,:,1:].var(axis=0,ddof=1)/4))
                size_diff=100/capacity*np.sqrt(np.mean((ensemble[1:]-p[m,2].mean(axis=0)[1:])**2))
                rows.append(dict(case=case,history=history,mode=mode,particles=size,rmse_percent=float(rmse),
                    future_mean_se_rms_percent=float(fse),particle_mean_se_rms_percent=float(particle_se),
                    difference_from_256_percent=float(size_diff)))
        # Condition on the six selected exits. Particle and future resamples are independent.
        difference_future=np.empty(6000);difference_both=np.empty(6000)
        for start in range(0,6000,100):
            fmean=f[draws[start:start+100]].mean(axis=1)
            pmean=p[:,2].mean(axis=1)
            errs=100/capacity*np.sqrt(np.mean((pmean[None,:,1:]-fmean[:,None,1:])**2,axis=2))
            difference_future[start:start+100]=errs[:,0]-errs[:,1]
            pboot=p[:,2,particle_draws[start:start+100]].mean(axis=2).transpose(1,0,2)
            errs=100/capacity*np.sqrt(np.mean((pboot[:,:,1:]-fmean[:,None,1:])**2,axis=2))
            difference_both[start:start+100]=errs[:,0]-errs[:,1]
        point=[r['rmse_percent'] for r in rows if r['case']==case and r['history']==history and r['particles']==256]
        contrasts.append(dict(case=case,history=history,contrast='coupled-minus-mean_trust at 256 particles, four replicates',
            difference_pp=point[0]-point[1],future_only_interval=np.quantile(difference_future,[.025,.975]).tolist(),
            future_and_particle_interval=np.quantile(difference_both,[.025,.975]).tolist()))
    for policy in policies:
        for mode in modes:
            q=np.load(root/'results'/f'quadrature_{case}_{policy}_{mode}.npz')
            for j,order in enumerate(q['orders'][1:],1):
                diff=q['predictions'][j]-q['predictions'][0]
                quadrature.append(dict(case=case,policy=policy,mode=mode,order=int(order),
                    maximum_degree=int(q['maximum_degree']),max_degree_difference=float(np.max(np.abs(diff[:,1]))),
                    max_opinion_gap_difference=float(np.max(np.abs(diff[:,4]))),
                    max_survival_difference=float(np.max(np.abs(diff[:,6])))))
        prefix={'high_degree':root/'inputs/forecast_exits/confirm_more_opportunities_10000',
                'heterogeneous':root/'inputs/forecast_exits/confirm_heterogeneous_10000',
                'strong_opinion':root/'exits/stress_strong_opinion_16000'}[case]
        state=NetworkModel.load(str(prefix)+f'_{policy}.npz')
        max_degree=int(np.bincount(np.r_[state.i,state.j],minlength=state.cfg.n).max())
        order=max(24,(max_degree+1)//2)
        q12=OpinionExpectation(state.cfg.n,state.i,state.j,12)
        qhigh=OpinionExpectation(state.cfg.n,state.i,state.j,order)
        state.rng_nat=np.random.default_rng(np.random.SeedSequence([state.seed,21001,0]))
        differences=[]
        for t in range(401):
            s=state.natural_probability()*state.probability(np.arange(state.m))
            differences.append(np.max(np.abs(q12.increment(state.x,s)-qhigh.increment(state.x,s))))
            if t<400:
                state.step()
        state_rows.append(dict(case=case,policy=policy,maximum_degree=max_degree,exact_order=order,
            maximum_one_step_increment_difference=float(max(differences)),
            scope='401 states along prespecified conditional future replicate 0; unscaled opinion increment'))
report=dict(units='percent of cross-community opportunity capacity; intervals in percentage points',
            conditional_scope='Six fixed exit states; 32 future replicates and four particle replicates per resolution; not new graph samples',
            precision=rows,method_contrasts=contrasts,quadrature=quadrature,one_step_quadrature=state_rows)
(root/'results/precision_analysis.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(root/'results/precision_analysis.npz',futures=np.stack(future_all),predictions=np.stack(pred_all),
                    bootstrap_future_indices=draws,bootstrap_particle_indices=particle_draws,cases=cases,modes=modes,sizes=sizes)
print('PRECISION',json.dumps(report,indent=2))
