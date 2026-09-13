import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root/'outputs/manuscript_restructured_20260911'
latex = out/'latex'
latex.mkdir(parents=True, exist_ok=True)
origin = Path(__file__).resolve().parents[1]/'inputs/original'
a = np.load(origin/'results/forecast_confirm.npz',allow_pickle=False)
meta = json.loads(a['metadata'].item())
pm = json.loads(a['prediction_metrics'].item())
metrics = json.loads(a['metrics'].item())
new = np.stack([np.load(out/'closure'/f'exit_{i:03d}.npz',allow_pickle=False)['predictions'] for i in range(144)])
pred = np.concatenate([a['predictions'],new],axis=1)
modes = ['coupled','frozen','mean_trust','factorized']
variants = list(dict.fromkeys(m['variant'] for m in meta))
errors = []
curves = []
for variant in variants:
    indices = {p:[i for i,m in enumerate(meta) if m['variant']==variant and m['condition']==p] for p in ['none','context']}
    cfg = meta[indices['none'][0]]['config']
    capacity = cfg['opportunity_degree']*cfg['mixing']
    true = {p:a['trajectories'][ix].mean(axis=0) for p,ix in indices.items()}
    predicted = {p:pred[ix].mean(axis=0) for p,ix in indices.items()}
    effect = true['context'][:,metrics.index('b')]-true['none'][:,metrics.index('b')]
    predicted_effect = predicted['context'][:,:,pm.index('b')]-predicted['none'][:,:,pm.index('b')]
    curves.append({'variant':variant,'capacity':capacity,'true_effect':effect.tolist(),'predicted_effect':predicted_effect.tolist()})
    for j,mode in enumerate(modes):
        row = {'variant':variant,'mode':mode,'paired_degree_rmse_percent':float(100*np.sqrt(np.mean(((predicted_effect[j,1:]-effect[1:])/capacity)**2))),
               'area_signed_error':float(np.trapezoid(predicted_effect[j]-effect)/400)}
        for p in ['none','context']:
            row[p+'_degree_rmse_percent']=float(100*np.sqrt(np.mean(((predicted[p][j,1:,pm.index('b')]-true[p][1:,metrics.index('b')])/capacity)**2)))
        errors.append(row)
overall=[]
for mode in modes:
    rows=[r for r in errors if r['mode']==mode]
    overall.append({'mode':mode,**{k:float(np.sqrt(np.mean([r[k]**2 for r in rows]))) for k in ['paired_degree_rmse_percent','none_degree_rmse_percent','context_degree_rmse_percent']}})
(out/'theory/closure_results.json').write_text(json.dumps({'unit':'12 original seed blocks across six conditions; new modes retrospective','modes':modes,'overall':overall,'per_condition':errors,'curves':curves},indent=2)+'\n')
print('CLOSURE',json.dumps(overall))

# An independent Cartesian-product calculation checks the marginal transition
# interpretation at a heterogeneous state, including nonlinear response.
w=np.array([0.,.08,.4,.85]);r=np.array([.1,.3,.55,.9]);distance=.73
lam=.15*(.1+.9*w);p=1/(1+np.exp(-(-1+2.8*r-distance)))
reinforced=np.where(w==0,.4,w+.3*(1-w))
dw0=np.where(.975*w>=.05,.975*w,0)
dw1=np.where(.975*reinforced>=.05,.975*reinforced,0)
direct_w=0.;direct_r=0.;direct_active=0.
for wi in range(4):
    for ri in range(4):
        probabilities=[1-lam[wi],lam[wi]*(1-p[ri]),lam[wi]*p[ri]]
        weights=[dw0[wi],dw0[wi],dw1[wi]]
        trusts=[r[ri],r[ri]+.15*(0-r[ri]),r[ri]+.15*(1-r[ri])]
        for probability,weight,trust in zip(probabilities,weights,trusts):
            direct_w+=probability*weight/16
            direct_r+=probability*(.995*trust+.005*.2)/16
            direct_active+=probability*(weight>0)/16
factor_w=np.mean((1-lam*p.mean())*dw0+lam*p.mean()*dw1)
factor_r=np.mean(.995*(r+.15*lam.mean()*(p-r))+.005*.2)
factor_active=np.mean((1-lam*p.mean())*(dw0>0)+lam*p.mean()*(dw1>0))
verification={'cartesian_product_expectations':[direct_w,direct_r,direct_active],
              'factorized_marginal_expectations':[float(factor_w),float(factor_r),float(factor_active)],
              'max_difference':float(np.max(np.abs(np.array([direct_w,direct_r,direct_active])-[factor_w,factor_r,factor_active]))),
              'scope':'One-step expected weight, trust and occupancy; enumeration independent of diagnostic particle code; no test of full-network factorization accuracy'}
(out/'theory/closure_one_step_verification.json').write_text(json.dumps(verification,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
colors=['#20639b','#df6b37','#299b80','#8a62a5']
renewal=json.loads((out/'theory/renewal_results.json').read_text())
episodes=json.loads((out/'theory/first_passage_results.json').read_text())
fig=plt.figure(figsize=(6.7,5.4),layout='constrained')
gs=fig.add_gridspec(2,2)
ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
for k,delta in enumerate([.0125,.025,.05]):
    cells=[r for r in renewal if r['delta']==delta]
    x=[r['p'] for r in cells]
    ax[0].plot(x,[r['1201']['renewal_occupancy'] for r in cells],color=colors[k],label=rf'$\delta={delta}$')
    y=np.array([r['original_mc_occupancy']['mean'] for r in cells]);ci=np.array([r['original_mc_occupancy']['95_interval'] for r in cells])
    ax[0].errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',ms=3,color=colors[k],capsize=2)
    ax[1].plot(x,[r['1201']['mean_lifetime'] for r in cells],'-o',ms=3,color=colors[k])
ax[0].set(xlabel='Fixed response $p$',ylabel='Active fraction',title='(a) Renewal occupancy',ylim=(0,1))
ax[0].legend(frameon=False,fontsize=9)
ax[1].set(xlabel='Fixed response $p$',ylabel='Mean lifetime (ticks)',title='(b) Active episode',yscale='log')
for i,r in enumerate(episodes):
    lo,hi=r['95_block_interval'];mean=r['mean_lifetime']
    ax[2].errorbar(r['operator_lifetime'],mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i],capsize=3,label=rf"$p={r['p']},\ \delta={r['delta']}$")
minimum=35;maximum=225
ax[2].plot([minimum,maximum],[minimum,maximum],':',color='.5')
ax[2].set(xlabel='Operator mean (ticks)',ylabel='Direct episode mean (ticks)',title='(c) First-passage check',xlim=(minimum,maximum),ylim=(minimum,maximum))
ax[2].legend(frameon=False,fontsize=9,loc='upper left')
fig.savefig(latex/'fig_renewal.pdf');plt.close(fig)

c=np.load(out/'theory/cohort_decomposition.npz',allow_pickle=False)
t=np.arange(401)
fig=plt.figure(figsize=(6.7,5.6),layout='constrained')
gs=fig.add_gridspec(2,2)
ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
labels=['All active ties','Continuous exit cohort','Replacement episodes']
for j,label in enumerate(labels):
    values=c['paired_effect'][:,j];mean=values.mean(axis=0);half=student.ppf(.975,23)*values.std(axis=0,ddof=1)/np.sqrt(24)
    ax[0].plot(t,mean,color=colors[j],label=label);ax[0].fill_between(t,mean-half,mean+half,color=colors[j],alpha=.15)
ax[0].axhline(0,color='.7',lw=.8);ax[0].set(xlabel='Ticks after withdrawal',ylabel='Paired excess degree',title='(a) Paired decomposition');ax[0].legend(frameon=False,fontsize=8)
for j in [1,2]:
    for policy,ls in [('context','-'),('none','--')]:
        ax[1].plot(t,c[policy][:,j].mean(axis=0),color=colors[j],ls=ls,label=('Matched' if policy=='context' else 'Unmediated')+(' cohort' if j==1 else ' replacement'))
ax[1].set(xlabel='Ticks after withdrawal',ylabel='Degree by component',title='(b) Separate policy histories');ax[1].legend(frameon=False,fontsize=8)
stats=json.loads((out/'theory/cohort_results.json').read_text())['statistics']
for j,component in enumerate(['total','continuous_cohort','replacement']):
    r=next(r for r in stats if r['component']==component and r['outcome']=='area');mean=r['mean'];lo,hi=r['95_interval']
    ax[2].bar(j,mean,color=colors[j],alpha=.85);ax[2].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],color='black',capsize=3)
ax[2].axhline(0,color='.5',lw=.8);ax[2].set(xticks=[0,1,2],xticklabels=['Total','Continuous cohort','Replacement'],ylabel='Mean excess degree',title='(c) Mean contributions over 400 ticks')
fig.savefig(latex/'fig_cohorts.pdf');plt.close(fig)

fig,ax=plt.subplots(3,2,figsize=(6.7,7.1),layout='constrained')
display={'slow_decay':'Slow decay','faster_decay':'Faster decay','high_decay':'High decay','heterogeneous':'Heterogeneous','fewer_opportunities':'Fewer opportunities','more_opportunities':'More opportunities'}
for j,c in enumerate(curves):
    aa=ax.ravel()[j];aa.plot(t,np.array(c['true_effect'])/c['capacity'],color='black',label='Full network',lw=1.8)
    for k in [0,2,3]:
        aa.plot(t,np.array(c['predicted_effect'][k])/c['capacity'],color=colors[k],ls=['--',':','-.','--'][k],label={'coupled':'Joint','mean_trust':'Mean trust','factorized':'Factorized'}[modes[k]])
    aa.set_title(f'({chr(97+j)}) '+display[c['variant']]);aa.set_xlabel('Ticks after withdrawal');aa.set_ylabel('Excess / capacity')
ax[0,0].legend(frameon=False,fontsize=8)
fig.savefig(latex/'fig_closure_diagnostic.pdf');plt.close(fig)

rows=['\\begin{tabular}{lrrr}\\toprule','Forecast & Paired increment & Unmediated & Matched \\\\','\\midrule']
for r in overall:
    name={'coupled':'Joint weight--trust','frozen':'Frozen response','mean_trust':'Mean trust','factorized':'Factorized marginals'}[r['mode']]
    rows.append(f"{name} & {r['paired_degree_rmse_percent']:.3f} & {r['none_degree_rmse_percent']:.3f} & {r['context_degree_rmse_percent']:.3f} \\\\")
rows+=['\\bottomrule\\end{tabular}']
(latex/'table_closure_new.tex').write_text('\n'.join(rows)+'\n')
rows=['\\begin{tabular}{rrrrrr}\\toprule','$p$ & $\\delta$ & Episodes & Direct mean [95\\% interval] & Operator & Error (\\%) \\\\','\\midrule']
for r in episodes:
    lo,hi=r['95_block_interval']
    rows.append(f"{r['p']:.2f} & {r['delta']:.3f} & {r['episodes']} & {r['mean_lifetime']:.2f} [{lo:.2f}, {hi:.2f}] & {r['operator_lifetime']:.2f} & {100*r['relative_error']:.2f} \\\\")
rows+=['\\bottomrule\\end{tabular}']
(latex/'table_lifetimes.tex').write_text('\n'.join(rows)+'\n')
summary={'max_renewal_vs_old_operator':max(abs(r['1201']['renewal_occupancy']-r['1201']['old_operator_occupancy']) for r in renewal),
 'max_renewal_vs_original_mc':max(abs(r['1201']['renewal_occupancy']-r['original_mc_occupancy']['mean']) for r in renewal),
 'max_occupancy_grid_change':max(abs(r['1201']['renewal_occupancy']-r['601']['renewal_occupancy']) for r in renewal),
 'max_lifetime_relative_grid_change':max(abs(r['1201']['mean_lifetime']-r['601']['mean_lifetime'])/r['1201']['mean_lifetime'] for r in renewal),
 'max_lifetime_equation_residual':max(r['1201']['lifetime_equation_max_residual'] for r in renewal),
 'max_stationarity_residual':max(r['1201']['stationarity_max_residual'] for r in renewal),
 'max_new_lifetime_relative_error':max(abs(r['relative_error']) for r in episodes),
 'all_four_predictions_in_pointwise_intervals':all(r['95_block_interval'][0]<=r['operator_lifetime']<=r['95_block_interval'][1] for r in episodes)}
(out/'theory/numerical_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('NUMERICS',json.dumps(summary));print('FIRST PASSAGE',json.dumps([{k:v for k,v in r.items() if k!='block_means'} for r in episodes]))
