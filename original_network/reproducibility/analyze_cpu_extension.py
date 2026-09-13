"""Statistics and publication figures for the prespecified CPU extension."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from cpu_experiments import LOW_CELLS, RULES
from forecast_experiments import CASES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"manuscript"
(ROOT/"qa").mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"STIXGeneral", "mathtext.fontset":"stix",
    "font.size":10, "axes.labelsize":11, "axes.titlesize":11, "legend.fontsize":9,
    "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.7,
    "pdf.fonttype":42, "savefig.dpi":200})
COLORS = {"context":"#126D94", "contact_only":"#C47A27", "shuffled":"#BF5554", "random_pair":"#498A76"}
LABELS = {"context":"Context matched", "contact_only":"Contact only", "shuffled":"Shuffled inputs", "random_pair":"Random pairs"}
RULE_LABELS = ["Baseline", "Lower deletion threshold", "Higher deletion threshold",
               "Fewer opportunities", "More opportunities", "No trust learning",
               "No opinion learning", "Neither learning process"]
CASE_LABELS = ["Slow decay", "Faster decay", "High decay", "Unequal node weights",
               "Fewer opportunities", "More opportunities"]


def load(suite):
    with np.load(ROOT/"results"/f"{suite}.npz", allow_pickle=False) as z:
        return {k: (json.loads(str(z[k])) if k in ["metadata", "metrics", "prediction_metrics", "prediction_methods"] else z[k].copy()) for k in z.files}


def select(data, condition, **filters):
    ids = [i for i,m in enumerate(data['metadata']) if m['condition']==condition and
           all(m.get(k,m['config'].get(k))==v for k,v in filters.items())]
    return sorted(ids, key=lambda i:data['metadata'][i]['seed'])


def interval(x):
    x = np.asarray(x)
    rng = np.random.default_rng(12001)
    boots = x[rng.integers(0,len(x),(6000,len(x)))].mean(axis=1)
    lo,hi = np.quantile(boots,[.025,.975])
    return dict(mean=float(x.mean()),lo=float(lo),hi=float(hi),n=len(x))


def difference(data, pa, pb, filters_a, filters_b=None, metric="b"):
    ia = select(data,pa,**filters_a)
    ib = select(data,pb,**(filters_a if filters_b is None else filters_b))
    assert [data['metadata'][i]['seed'] for i in ia] == [data['metadata'][i]['seed'] for i in ib]
    j = data['metrics'].index(metric)
    return data['trajectories'][ia,:,j]-data['trajectories'][ib,:,j]


def measures(diff, off=100):
    return {"off":interval(diff[:,off]), "A":interval(np.trapezoid(diff[:,off:],axis=1)/400),
            "final":interval(diff[:,-1])}


def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(ROOT/"qa"/f"{name}.png",bbox_inches="tight")
    plt.close(fig)


def points(ax,x,values,color,label):
    y = np.array([v['mean'] for v in values])
    lo = np.array([v['lo'] for v in values]); hi = np.array([v['hi'] for v in values])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="o-",capsize=3,color=color,label=label,lw=1.3,ms=4)


low, rules = load('low_intensity'), load('mechanism_rules')
summary = {"low_intensity":[],"mechanism_rules":[]}
for decay,intensity in LOW_CELLS:
    filters = dict(decay=decay,mediation_intensity=intensity)
    cell = dict(decay=decay,intensity=intensity, attempts_per_tick=round(intensity*250/2),
                contrasts={},effects={},occupancy={},survival={},event_contributions={})
    for policy in COLORS:
        ids=select(low,policy,**filters)
        a=low['trajectories'][ids]
        assert np.all(np.nansum(a[:,51:101,low['metrics'].index('mediation_attempts')],axis=1)==cell['attempts_per_tick']*50)
        cell['effects'][policy]=measures(difference(low,policy,'none',filters,dict(decay=decay)))
        cell['occupancy'][policy]=interval(a[:,100,low['metrics'].index('b')]/6)
        cell['survival'][policy]=interval(a[:,-1,low['metrics'].index('survival')])
        cell['event_contributions'][policy]={m:interval(np.nansum(a[:,51:101,low['metrics'].index(m)],axis=1)*2/250)
            for m in ['natural_births','mediation_births','deletions']}
        if policy!='context':
            cell['contrasts'][policy]=measures(difference(low,'context',policy,filters))
    summary['low_intensity'].append(cell)

reference=difference(rules,'context','none',dict(variant='baseline'))
for variant in RULES:
    d=difference(rules,'context','none',dict(variant=variant))
    ids=select(rules,'context',variant=variant)
    c=rules['metadata'][ids[0]]['config'];cap=c['opportunity_degree']*c['mixing']
    summary['mechanism_rules'].append(dict(variant=variant,capacity=cap,effect=measures(d),
        normalized_effect=measures(d/cap),difference_from_baseline=measures(d-reference)))

fig,axes=plt.subplots(2,2,figsize=(7.2,6.0))
us=[.05,.10,.20]
base_cells=[next(r for r in summary['low_intensity'] if r['decay']==.025 and r['intensity']==u) for u in us]
for p in COLORS:
    points(axes[0,0],us,[r['occupancy'][p] for r in base_cells],COLORS[p],LABELS[p])
axes[0,0].set(xlabel='Nominal intensity $u$',ylabel='Fraction of opportunity capacity',
              title='(a) Occupancy at withdrawal',ylim=(0,1))
for p in ['contact_only','shuffled','random_pair']:
    points(axes[0,1],us,[r['contrasts'][p]['A'] for r in base_cells],COLORS[p],LABELS[p])
axes[0,1].set(xlabel='Nominal intensity $u$',ylabel='Context advantage in $A_{400}$',
              title='(b) Equal contact budgets')
ds=[.015,.025,.04]
for p in ['contact_only','shuffled','random_pair']:
    cells=[next(r for r in summary['low_intensity'] if r['decay']==d and r['intensity']==.1) for d in ds]
    points(axes[1,0],ds,[r['contrasts'][p]['A'] for r in cells],COLORS[p],LABELS[p])
axes[1,0].set(xlabel=r'Decay fraction $\delta$',ylabel='Context advantage in $A_{400}$',
              title='(c) Decay at $u=0.10$')
for u,col in zip(us,['#7C71A3','#126D94','#498A76']):
    d=difference(low,'context','shuffled',dict(decay=.025,mediation_intensity=u))[:,100:]
    half=student_t.ppf(.975,23)*d.std(axis=0,ddof=1)/np.sqrt(24)
    axes[1,1].plot(np.arange(401),d.mean(axis=0),color=col,label=f'$u={u:.2f}$')
    axes[1,1].fill_between(np.arange(401),d.mean(axis=0)-half,d.mean(axis=0)+half,color=col,alpha=.12,lw=0)
axes[1,1].set(xlabel='Time since withdrawal (ticks)',ylabel='Context minus shuffled degree',
              title='(d) Input assignment after withdrawal')
axes[1,1].legend(frameon=False,fontsize=9)
for ax in axes.ravel():ax.grid(alpha=.13,axis='y',lw=.5)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,fontsize=9.5)
fig.tight_layout(rect=[0,0,1,.945]);save(fig,'fig8_low_intensity')

fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),sharey=True)
for ax,key,xlabel in zip(axes,['effect','normalized_effect'],['$A_{400}$',r'$A_{400}/(k_o\mu)$']):
    for i,r in enumerate(summary['mechanism_rules']):
        v=r[key]['A']
        ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['lo']],[v['hi']-v['mean']]],fmt='o',capsize=3,color='#126D94')
    ax.set(xlabel=xlabel);ax.axvline(0,color='.65',ls=':',lw=.8);ax.grid(alpha=.13,axis='x')
axes[0].set_yticks(range(8),RULE_LABELS);axes[0].invert_yaxis()
axes[0].set_title('(a) Absolute intervention gain');axes[1].set_title('(b) Capacity-normalized gain')
fig.tight_layout();save(fig,'figS2_mechanism_rules')

pilot=load('forecast_pilot')
pj=pilot['prediction_metrics'].index('b')
pd=pilot['predictions'][:,0,1:,pj]-pilot['predictions'][:,1,1:,pj]
summary['forecast_numerics']={'particle_comparison':'32 versus 64, six development exit states',
    'normalized_rmse_by_state':(np.sqrt(np.mean(pd*pd,axis=1))/6).tolist(),
    'max_normalized_rmse':float(np.max(np.sqrt(np.mean(pd*pd,axis=1))/6))}
data=load('forecast_confirm');j=data['metrics'].index('b');pj=data['prediction_metrics'].index('b')
rng=np.random.default_rng(12002)
resamples=rng.integers(0,12,(6000,12))
weights=np.array([np.bincount(r,minlength=12)/12 for r in resamples])
summary['forecast']={};curve_data={};normalized_errors={method:[] for method in ['coupled','frozen']}
for case in CASES:
    ia=select(data,'context',variant=case);ib=select(data,'none',variant=case)
    assert [data['metadata'][i]['seed'] for i in ia]==[data['metadata'][i]['seed'] for i in ib]
    cfg=data['metadata'][ia[0]]['config'];cap=cfg['opportunity_degree']*cfg['mixing']
    actual=data['trajectories'][ia,:,j]-data['trajectories'][ib,:,j]
    predictions=data['predictions'][ia,:,:,pj]-data['predictions'][ib,:,:,pj]
    case_result={'capacity':cap,'config':cfg,'actual_effect':measures(actual,off=0),'methods':{}}
    curve_data[case]=(actual,predictions)
    for method,mode in enumerate(['coupled','frozen']):
        error=predictions[:,method]-actual
        normalized=error[:,1:]/cap
        normalized_errors[mode].append(normalized)
        boot=np.sqrt(np.mean((weights@normalized)**2,axis=1))
        case_result['methods'][mode]={
            'nrmse':float(np.sqrt(np.mean(normalized.mean(axis=0)**2))),
            'nrmse_interval':np.quantile(boot,[.025,.975]).tolist(),
            'A_error':interval(np.trapezoid(error,axis=1)/400),
            'predicted_effect':measures(predictions[:,method],off=0),
            'absolute_degree_rmse':float(np.sqrt(np.mean((data['predictions'][ia,method,1:,pj].mean(axis=0)-data['trajectories'][ia,1:,j].mean(axis=0))**2))),
        }
    summary['forecast'][case]=case_result
pooled={}
for mode in ['coupled','frozen']:
    errors=np.stack(normalized_errors[mode])
    pooled[mode]=float(np.sqrt(np.mean(errors.mean(axis=1)**2)))
    pooled[mode+'_bootstrap']=np.sqrt(np.mean(np.stack([(weights@x)**2 for x in errors]),axis=(0,2)))
summary['forecast_pooled']={mode:pooled[mode] for mode in ['coupled','frozen']}
summary['forecast_pooled']['reduction_percent']=100*(1-pooled['coupled']/pooled['frozen'])
summary['forecast_pooled']['difference_interval']=np.quantile(pooled['coupled_bootstrap']-pooled['frozen_bootstrap'],[.025,.975]).tolist()

fig,axes=plt.subplots(3,2,figsize=(7.0,6.6),sharex=True)
for ax,case,label in zip(axes.ravel(),CASES,CASE_LABELS):
    actual,pred=curve_data[case]
    mean=actual.mean(axis=0)
    half=student_t.ppf(.975,11)*actual.std(axis=0,ddof=1)/np.sqrt(12)
    ax.fill_between(np.arange(401),mean-half,mean+half,color='#656B73',alpha=.2,lw=0)
    ax.plot(mean,color='#656B73',lw=1.7,label='Full network')
    ax.plot(pred[:,0].mean(axis=0),color='#126D94',lw=1.4,ls='--',label='Coupled forecast')
    ax.plot(pred[:,1].mean(axis=0),color='#C47A27',lw=1.2,ls=':',label='Frozen response')
    ax.set_title(label);ax.grid(alpha=.13,axis='y',lw=.5)
for ax in axes[:,0]:ax.set_ylabel('Paired degree increment')
for ax in axes[-1]:ax.set_xlabel('Ticks after withdrawal')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False,fontsize=10)
fig.tight_layout(rect=[0,0,1,.945]);save(fig,'fig9_forecast')

lines=[r'\begin{tabular}{lrrr}',r'\toprule',
    r'Condition & Joint error (\%) & Frozen error (\%) & $\Delta A_{400}$ \\',r'\midrule']
for case,label in zip(CASES,CASE_LABELS):
    r=summary['forecast'][case]['methods'];a=r['coupled']['A_error']
    lines.append(f"{label} & {100*r['coupled']['nrmse']:.2f} & {100*r['frozen']['nrmse']:.2f} & {a['mean']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}]"+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'table_forecast.tex').write_text('\n'.join(lines))

macros={}
for i,cell in enumerate(base_cells):
    key=['Low','Mid','Higher'][i]
    macros[key+'Occupancy']=100*cell['occupancy']['context']['mean']
    for p,label in [('contact_only','Contact'),('shuffled','Shuffle'),('random_pair','Random')]:
        for stat in ['mean','lo','hi']:
            macros[key+'Vs'+label+stat.title()]=cell['contrasts'][p]['A'][stat]
for rule,label in [('baseline','RuleBase'),('no_trust_learning','NoTrust'),('no_opinion_learning','NoOpinion'),('no_learning','NoLearning')]:
    row=next(r for r in summary['mechanism_rules'] if r['variant']==rule)
    macros[label+'Area']=row['effect']['A']['mean']
    for stat in ['mean','lo','hi']:
        macros[label+'Difference'+stat.title()]=row['difference_from_baseline']['A'][stat]
macros['ForecastCoupledPercent']=100*pooled['coupled'];macros['ForecastFrozenPercent']=100*pooled['frozen']
macros['ForecastReductionPercent']=summary['forecast_pooled']['reduction_percent']
macros['ParticleDifferencePercent']=100*summary['forecast_numerics']['max_normalized_rmse']
(OUT/'cpu_results_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.3f}'+'}' for k,v in macros.items()))
(ROOT/'results'/'cpu_extension_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'low_intensity_cells':len(summary['low_intensity']),
                  'rule_variants':len(summary['mechanism_rules']),
                  'forecast_conditions':len(summary['forecast']),
                  'forecast_pooled':summary['forecast_pooled']},indent=2))
