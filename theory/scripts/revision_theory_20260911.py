import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.stats import t as student

origin = Path(__file__).resolve().parents[1]/'inputs/original'
sys.path.insert(0, str(origin / 'reproducibility'))
from theory import transfer_operator
from model import Config

out = Path(__file__).resolve().parents[1]/'outputs/manuscript_restructured_20260911/theory'
out.mkdir(parents=True, exist_ok=True)
plan = {
 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'operator_cases':[[p,d] for p in [.15,.3,.5,.7] for d in [.0125,.025,.05]],
 'operator_grids':[601,1201],
 'first_passage_cases':[[.15,.025],[.5,.025],[.15,.05],[.5,.05]],
 'first_passage_blocks':16,'episodes_per_block':512,
 'first_passage_seed_sequence':'[15000, case_index]; independent entries of a vectorized simulation; 16 blocks of 512 episodes',
 'first_passage_start':'post-decay birth weight (1-delta)*w_b; stop each episode at first deletion; retain every lifetime, no censoring or replacement',
 'comparisons':'Renewal occupancy vs original independent Monte Carlo; mean first-passage lifetime vs new independent direct episodes; report numerical residual and 601/1201 sensitivity',
 'bootstrap_cohort_analysis':{'resamples':6000,'seed':65000,'unit':'24 original paired seeds; within-seed cohort decomposition first, averaging second'},
 'closure_diagnostic':{'exits':'all 144 saved confirmation exit states, six conditions x 12 seeds x two histories','new_modes':['weight-marginal with mean trust','factorized weight and trust marginals'],'particles':64,'followup':400,'random_seed_sequence':'[network_seed, 19701] reset for each diagnostic mode','original_truth':'unchanged saved full-network futures','interpretation':'Retrospective closure diagnostic on existing confirmation cases; no claim of new independent confirmation'},
 'new_gpu_calls':0,'original_files_modified':False,'no_posthoc_pass_threshold':True
}
(out/'execution_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cfg = Config()
old = json.loads((origin/'results/theory_benchmark.json').read_text())
results=[]
for case in old:
    entry={'p':case['p'],'delta':case['decay']}
    for cells in plan['operator_grids']:
        grid, P = transfer_operator(case['p'],case['decay'],cells,cfg)
        K=P[1:,1:]
        lifetime=spsolve(eye(cells-1,format='csc')-K.T,np.ones(cells-1))
        activation=float(P[1:,0].sum())
        birth=np.asarray(P[1:,0].toarray()).ravel()/activation
        mean_lifetime=float(birth@lifetime)
        occupancy=mean_lifetime/(1/activation+mean_lifetime)
        mass=spsolve(eye(cells-1,format='csc')-K,birth)
        stationary=np.r_[1/activation,mass];stationary/=stationary.sum()
        entry[str(cells)]={'mean_lifetime':mean_lifetime,'waiting_time':1/activation,'renewal_occupancy':occupancy,'old_operator_occupancy':case[f'operator_{cells}']['active'],'lifetime_equation_max_residual':float(np.max(np.abs(lifetime-K.T@lifetime-1))),'stationarity_max_residual':float(np.max(np.abs(P@stationary-stationary)))}
        np.savez_compressed(out/f'operator_p{case["p"]}_d{case["decay"]}_n{cells}.npz',grid=grid,lifetime=np.r_[0.,lifetime],birth_distribution=birth,stationary=stationary)
    mc=np.array(case['mc_active'])
    half=student.ppf(.975,15)*mc.std(ddof=1)/np.sqrt(16)
    entry['original_mc_occupancy']={'mean':float(mc.mean()),'95_interval':[float(mc.mean()-half),float(mc.mean()+half)]}
    results.append(entry)
(out/'renewal_results.json').write_text(json.dumps(results,indent=2)+'\n')

episode_results=[]
for i,(p,delta) in enumerate(plan['first_passage_cases']):
    rng=np.random.default_rng(np.random.SeedSequence([15000,i]))
    weights=np.full((16,512),(1-delta)*cfg.birth_weight)
    lifetimes=np.zeros((16,512),dtype=np.int64)
    alive=np.ones((16,512),dtype=bool)
    while alive.any():
        lifetimes+=alive
        success=alive & (rng.random(weights.shape)<cfg.natural_rate/cfg.opportunity_degree*(cfg.exploration+(1-cfg.exploration)*weights)*p)
        weights[success]+=cfg.reinforcement*(1-weights[success])
        weights[alive]*=1-delta
        alive &= weights>=cfg.threshold
    means=lifetimes.mean(axis=1)
    half=student.ppf(.975,15)*means.std(ddof=1)/4
    prediction=next(r['1201']['mean_lifetime'] for r in results if r['p']==p and r['delta']==delta)
    episode_results.append({'p':p,'delta':delta,'mean_lifetime':float(means.mean()),'95_block_interval':[float(means.mean()-half),float(means.mean()+half)],'operator_lifetime':prediction,'absolute_error':float(abs(means.mean()-prediction)),'relative_error':float((means.mean()-prediction)/prediction),'block_means':means.tolist(),'max_lifetime':int(lifetimes.max()),'episodes':int(lifetimes.size),'censored':0})
    np.savez_compressed(out/f'episodes_p{p}_d{delta}.npz',lifetimes=lifetimes)
    print('First passage',p,delta,float(means.mean()),'prediction',prediction,flush=True)
(out/'first_passage_results.json').write_text(json.dumps(episode_results,indent=2)+'\n')

data=np.load(origin/'results/main.npz',allow_pickle=False)
meta=json.loads(data['metadata'].item());metrics=json.loads(data['metrics'].item())
cohorts={}
for policy in ['none','context']:
    ids=[i for i,m in enumerate(meta) if m['condition']==policy]
    histories=data['trajectories'][ids,100:,:]
    degree=histories[:,:,metrics.index('b')]
    surviving=degree[:,0,None]*histories[:,:,metrics.index('survival')]
    cohorts[policy]=np.stack([degree,surviving,degree-surviving],axis=1)
effect=cohorts['context']-cohorts['none']
boot=np.random.default_rng(65000).integers(0,24,(6000,24))
stats=[]
for j,label in enumerate(['total','continuous_cohort','replacement']):
    for endpoint in ['area','tick400']:
        x=np.trapezoid(effect[:,j],dx=1,axis=1)/400 if endpoint=='area' else effect[:,j,-1]
        stats.append({'component':label,'outcome':endpoint,'mean':float(x.mean()),'95_interval':np.quantile(x[boot].mean(axis=1),[.025,.975]).tolist(),'seed_values':x.tolist()})
np.savez_compressed(out/'cohort_decomposition.npz',none=cohorts['none'],context=cohorts['context'],paired_effect=effect,bootstrap_indices=boot,seeds=np.array([m['seed'] for m in meta if m['condition']=='none']))
(out/'cohort_results.json').write_text(json.dumps({'statistics':stats,'max_identity_residual':float(np.max(np.abs(effect[:,0]-effect[:,1]-effect[:,2]))),'interpretation':'Own-exit-cohort accounting, not causal mediation decomposition; reborn edges enter replacement.'},indent=2)+'\n')
print('Cohort statistics',json.dumps(stats[:2]),flush=True)
