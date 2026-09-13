"""Analyze the completed, fixed MR2 simulations, without selecting conditions."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
metrics = json.loads(np.load(root/'inputs/main.npz')['metrics'].item())
bix, six = metrics.index('b'), metrics.index('survival')
draws = np.random.default_rng([20260912,3]).integers(0,24,(6000,24))
truth = np.stack([[np.load(root/'results'/f'baseline_{seed}_{policy}.npz')['trajectories']
                   for policy in ['none','context']] for seed in range(1000,1024)])
pred = np.stack([[np.load(root/'results'/f'baseline_{seed}_{policy}.npz')['prediction']
                  for policy in ['none','context']] for seed in range(1000,1024)])
relax = np.stack([[np.load(root/'results'/f'baseline_{seed}_{policy}.npz')['relaxation']
                   for policy in ['none','context']] for seed in range(1000,1024)])
replay = np.array([[np.load(root/'results'/f'baseline_{seed}_{policy}.npz')['replay_max_difference']
                    for policy in ['none','context']] for seed in range(1000,1024)])
true_components = np.stack([truth[:,:,:,bix],truth[:,:,0,bix,None]*truth[:,:,:,six],
                            truth[:,:,:,bix]-truth[:,:,0,bix,None]*truth[:,:,:,six]],axis=2)
pred_components = np.stack([pred[:,:,:,1],truth[:,:,0,bix,None]*pred[:,:,:,6],
                            pred[:,:,:,1]-truth[:,:,0,bix,None]*pred[:,:,:,6]],axis=2)
e = pred_components-true_components
baseline = {'scope':'Same configuration and graph seeds as original, consistently re-executed histories; not extra independent seeds',
            'replay_maximum':float(replay.max()),'cohort_forecast':{},'restricted_mean_remaining':[]}
for h,error in [('paired',e[:,1]-e[:,0]),('none',e[:,0]),('context',e[:,1])]:
    baseline['cohort_forecast'][h] = dict(zip(['total','continuous','replacement'],
        (100/6*np.sqrt(np.mean(error.mean(axis=0)[:,1:]**2,axis=1))).tolist()))
for p,policy in enumerate(['none','context']):
    life = truth[:,p,:400,six].sum(axis=1)
    predicted = pred[:,p,:400,6].sum(axis=1)
    baseline['restricted_mean_remaining'].append(dict(policy=policy,observed=float(life.mean()),
        predicted=float(predicted.mean()),observed_interval=np.quantile(life[draws].mean(axis=1),[.025,.975]).tolist(),
        difference=float((predicted-life).mean()),difference_interval=np.quantile((predicted-life)[draws].mean(axis=1),[.025,.975]).tolist()))
# All relaxation variants retain identical exits. Curves are paired before averaging.
relax_b = np.concatenate([truth[:,:,None,:,bix],relax[:,:,:,:,bix]],axis=2)
relax_d = relax_b[:,1]-relax_b[:,0]
names = ['baseline','no_trust_return','fast_trust_return','no_opinion_anchoring','fast_opinion_anchoring']
relaxation = []
for v,name in enumerate(names):
    d = relax_d[:,v]
    initial = d[:,0];area=np.trapezoid(d,axis=1)/400
    aa=area[draws].mean(axis=1);dd=initial[draws].mean(axis=1)
    paired_area=area-np.trapezoid(relax_d[:,0],axis=1)/400
    relaxation.append(dict(variant=name,D0=float(initial.mean()),A400=float(area.mean()),
        A400_interval=np.quantile(aa,[.025,.975]).tolist(),Teff=float(400*area.mean()/initial.mean()),
        Teff_interval=np.quantile(400*aa/dd,[.025,.975]).tolist(),
        area_change=float(paired_area.mean()),area_change_interval=np.quantile(paired_area[draws].mean(axis=1),[.025,.975]).tolist()))
np.savez_compressed(root/'results/baseline_analysis.npz',truth=truth,prediction=pred,
                    true_components=true_components,pred_components=pred_components,relaxation=relax,
                    relaxation_effects=relax_d,bootstrap_indices=draws,replay_differences=replay)
(root/'results/baseline_analysis.json').write_text(json.dumps(baseline,indent=2))
(root/'results/relaxation_analysis.json').write_text(json.dumps(relaxation,indent=2))
print('BASELINE',json.dumps(baseline));print('RELAXATION',json.dumps(relaxation))

variants = ['baseline','no_trust_learning','no_opinion_learning','no_learning']
learning = np.stack([[np.load(root/'results'/f'shuffle_{variant}_{seed}.npz')['trajectories']
                      for seed in range(8000,8016)] for variant in variants])[:,:,:,100:]
draws = np.random.default_rng([20260912,4]).integers(0,16,(6000,16))
learn_rows = []
for v,variant in enumerate(variants):
    for contrast,left,right in [('matched-minus-shuffled',1,2),('matched-minus-none',1,0),('shuffled-minus-none',2,0)]:
        d=learning[v,:,left,:,bix]-learning[v,:,right,:,bix]
        initial=d[:,0];area=np.trapezoid(d,axis=1)/400
        aa=area[draws].mean(axis=1);dd=initial[draws].mean(axis=1)
        learn_rows.append(dict(variant=variant,contrast=contrast,D0=float(initial.mean()),
            D0_interval=np.quantile(dd,[.025,.975]).tolist(),A400=float(area.mean()),
            A400_interval=np.quantile(aa,[.025,.975]).tolist(),Teff=float(400*area.mean()/initial.mean()),
            Teff_interval=np.quantile(400*aa/dd,[.025,.975]).tolist(),minimum_bootstrap_denominator=float(dd.min())))
np.savez_compressed(root/'results/learning_analysis.npz',trajectories=learning,variants=variants,
                    bootstrap_indices=draws,policies=['none','context','shuffled'])
(root/'results/learning_analysis.json').write_text(json.dumps(learn_rows,indent=2))
duration_boot=[];duration_point=[];duration_changes=[]
for v in range(4):
    d=learning[v,:,1,:,bix]-learning[v,:,2,:,bix]
    area=np.trapezoid(d,axis=1);initial=d[:,0]
    duration_boot.append(area[draws].mean(axis=1)/initial[draws].mean(axis=1))
    duration_point.append(area.mean()/initial.mean())
for v in range(1,4):
    duration_changes.append(dict(variant=variants[v],Teff_change_from_baseline=float(duration_point[v]-duration_point[0]),
        interval=np.quantile(duration_boot[v]-duration_boot[0],[.025,.975]).tolist()))
(root/'results/learning_paired_duration_changes.json').write_text(json.dumps(duration_changes,indent=2))
print('LEARNING',json.dumps(learn_rows))

variants=['baseline','strong_opinion','strong_trust']
modes=['coupled','frozen','mean_trust','factorized']
truth=np.stack([[[np.load(root/'results'/f'stress_{v}_{s}_{p}_future.npz')['trajectories']
                  for p in ['none','context']] for s in range(16000,16012)] for v in variants])
pred=np.stack([[[np.load(root/'results'/f'stress_{v}_{s}_{p}_pred.npz')['predictions']
                 for p in ['none','context']] for s in range(16000,16012)] for v in variants])
e=(pred[:,:,:,:,:,1]-truth[:,:,:,None,:,bix])/6*100
rows=[]
draws=np.random.default_rng([20260912,5]).integers(0,12,(6000,12))
for v,variant in enumerate(variants):
    for h,error in [('paired',e[v,:,1]-e[v,:,0]),('none',e[v,:,0]),('context',e[v,:,1])]:
        rmse=np.sqrt(np.mean(error.mean(axis=0)[:,1:]**2,axis=1))
        for m,mode in enumerate(modes):
            rows.append(dict(variant=variant,history=h,mode=mode,rmse_percent=float(rmse[m])))
np.savez_compressed(root/'results/stress_analysis.npz',truth=truth,prediction=pred,variants=variants,modes=modes,
                    bootstrap_indices=draws)
(root/'results/stress_analysis.json').write_text(json.dumps(rows,indent=2))
print('STRESS',json.dumps(rows))
