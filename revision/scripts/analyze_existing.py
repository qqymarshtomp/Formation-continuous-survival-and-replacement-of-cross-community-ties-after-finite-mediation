"""MR2 reanalysis of original shared-seed confirmation and formation data."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
z = np.load(root/'inputs/forecast_confirm.npz')
meta = json.loads(z['metadata'].item())
metrics = json.loads(z['metrics'].item())
pm = json.loads(z['prediction_metrics'].item())
pred = np.concatenate([z['predictions'], np.stack([
    np.load(root/'inputs/closure'/f'exit_{i:03d}.npz')['predictions'] for i in range(144)])], axis=1)
modes = ['coupled','frozen','mean_trust','factorized']
variants = list(dict.fromkeys(t['variant'] for t in meta))
seeds = sorted({t['seed'] for t in meta})
truth = np.empty((6,12,2,3,401))
prediction = np.empty((6,12,2,4,3,401))
capacity = []
for v, variant in enumerate(variants):
    for s, seed in enumerate(seeds):
        for p, policy in enumerate(['none','context']):
            ix = next(i for i,t in enumerate(meta) if (t['variant'],t['seed'],t['condition'])==(variant,seed,policy))
            b = z['trajectories'][ix,:,metrics.index('b')]
            survival = z['trajectories'][ix,:,metrics.index('survival')]
            truth[v,s,p] = np.stack([b,b[0]*survival,b-b[0]*survival])
            bpred = pred[ix,:,:,pm.index('b')]
            spred = pred[ix,:,:,pm.index('survival')]
            prediction[v,s,p] = np.stack([bpred,b[0]*spred,bpred-b[0]*spred],axis=1)
    cfg = meta[next(i for i,t in enumerate(meta) if t['variant']==variant)]['config']
    capacity.append(cfg['opportunity_degree']*cfg['mixing'])
capacity = np.array(capacity)
errors = (prediction-truth[:,:,:,None])/capacity[:,None,None,None,None,None]*100
paired_error = errors[:,:,1]-errors[:,:,0]
all_errors = np.concatenate([paired_error[:,:,None],errors],axis=2)
point = np.sqrt(np.mean(all_errors.mean(axis=1)[...,1:]**2,axis=(0,4)))
draws = np.random.default_rng([20260912,1]).integers(0,12,(6000,12))
weights = (draws[:,:,None]==np.arange(12)[None,None,:]).sum(axis=1)/12
boot = np.empty((6000,3,4,3))
for start in range(0,6000,100):
    mean_error = np.einsum('bs,vshmct->bvhmct',weights[start:start+100],all_errors[...,1:], optimize=True)
    boot[start:start+100] = np.sqrt(np.mean(mean_error**2,axis=(1,5)))
report = {'units':'Capacity-normalized RMSE in percent; differences in percentage points',
          'uncertainty':'6000 shared seed-block resamples, conditional on existing particles; no time resampling',
          'modes':modes,'variants':variants,'histories':['paired','none','context'],
          'components':['total','continuous','replacement'],'results':[],'differences':[]}
for h,history in enumerate(report['histories']):
    for m,mode in enumerate(modes):
        for c,component in enumerate(report['components']):
            report['results'].append(dict(history=history,mode=mode,component=component,
                rmse_percent=float(point[h,m,c]),interval=np.quantile(boot[:,h,m,c],[.025,.975]).tolist()))
    for m in [2,3]:
        for c,component in enumerate(report['components']):
            report['differences'].append(dict(history=history,component=component,contrast='coupled-minus-'+modes[m],
                difference_pp=float(point[h,0,c]-point[h,m,c]),
                interval=np.quantile(boot[:,h,0,c]-boot[:,h,m,c],[.025,.975]).tolist()))
report['restricted_mean_remaining'] = []
for v,variant in enumerate(variants):
    for p,policy in enumerate(['none','context']):
        survival=truth[v,:,p,1]/truth[v,:,p,0,0,None]
        psurvival=prediction[v,:,p,:,1]/truth[v,:,p,0,0,None,None]
        report['restricted_mean_remaining'].append(dict(variant=variant,policy=policy,
            observed=float(survival[:,:400].sum(axis=1).mean()),
            predicted=psurvival[:,:,:400].sum(axis=2).mean(axis=0).tolist()))
(root/'results/confirmation_analysis.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(root/'results/confirmation_analysis.npz',truth=truth,prediction=prediction,
                    bootstrap_indices=draws,bootstrap_rmse=boot,capacity=capacity,variants=variants,modes=modes)
print(json.dumps(report['differences'],indent=2))

z = np.load(root/'inputs/main.npz')
meta = json.loads(z['metadata'].item())
metrics = json.loads(z['metrics'].item())
main = {}
for policy in ['none','context','shuffled','contact_only']:
    indices = sorted([i for i,t in enumerate(meta) if t['condition']==policy],key=lambda i:meta[i]['seed'])
    main[policy] = z['trajectories'][indices,100:,metrics.index('b')]
draws = np.random.default_rng([20260912,2]).integers(0,24,(6000,24))
response = []
for policy in ['none','shuffled','contact_only']:
    d = main['context']-main[policy]
    mean = d.mean(axis=0)
    initial = d[:,0]
    area = np.trapezoid(d,axis=1)/400
    initial_boot = initial[draws].mean(axis=1)
    area_boot = area[draws].mean(axis=1)
    response.append(dict(contrast='context-minus-'+policy,D0=float(mean[0]),
        D0_interval=np.quantile(initial_boot,[.025,.975]).tolist(),A400=float(area.mean()),
        A400_interval=np.quantile(area_boot,[.025,.975]).tolist(),
        Teff=float(400*area.mean()/mean[0]),
        Teff_interval=np.quantile(400*area_boot/initial_boot,[.025,.975]).tolist(),
        minimum_bootstrap_denominator=float(initial_boot.min())))
np.savez_compressed(root/'results/main_normalized.npz',**main,bootstrap_indices=draws)
(root/'results/main_normalized.json').write_text(json.dumps(response,indent=2))
print(json.dumps(response,indent=2))
