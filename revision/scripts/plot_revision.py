"""Scientific figures for MR2, derived only from archived arrays."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1];figdir=root/'latex';t=np.arange(401)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
colors=['#20639b','#d86a35','#269b81','#8b63a6','#666666']
labels=['Joint','Frozen','Mean trust','Factorized']
metrics=json.loads(np.load(root/'inputs/main.npz')['metrics'].item());bix=metrics.index('b')

z=np.load(root/'results/main_normalized.npz')
learn=np.load(root/'results/learning_analysis.npz')['trajectories']
lr=[r for r in json.loads((root/'results/learning_analysis.json').read_text()) if r['contrast']=='matched-minus-shuffled']
relax=np.load(root/'results/baseline_analysis.npz')['relaxation_effects']
fig,ax=plt.subplots(2,2,figsize=(6.8,6.1),layout='constrained')
for k,(left,right,label) in enumerate([('context','none','Matched vs none'),('shuffled','none','Shuffled vs none'),
                                      ('context','shuffled','Matched vs shuffled')]):
    d=(z[left]-z[right]).mean(axis=0)
    ax[0,0].plot(t,d/d[0],color=colors[k],label=label)
ax[0,0].set(title='(a) Formation-normalized gain',xlabel='Ticks after withdrawal',ylabel=r'$G(\tau)$')
ax[0,0].legend(frameon=False,fontsize=8)
names=['All learning','No trust learning','No opinion learning','Neither']
for k,name in enumerate(names):
    d=(learn[k,:,1,:,bix]-learn[k,:,2,:,bix]).mean(axis=0)
    ax[0,1].plot(t,d/d[0],color=colors[k],label=name)
    r=lr[k];x=r['D0'];y=r['Teff'];xl,xh=r['D0_interval'];yl,yh=r['Teff_interval']
    ax[1,0].errorbar(x,y,xerr=[[x-xl],[xh-x]],yerr=[[y-yl],[yh-y]],fmt='o',color=colors[k],capsize=2,label=name)
ax[0,1].set(title='(b) Assignment effect: learning',xlabel='Ticks after withdrawal',ylabel=r'$G_{\mathrm{matched-shuffled}}$')
ax[0,1].legend(frameon=False,fontsize=8)
ax[1,0].set(title='(c) Amplitude and retention',xlabel='Initial assignment gain $D_b(0)$',ylabel=r'$T_{\mathrm{eff}}$ (ticks)',xlim=(.50,.66))
for k,label in enumerate(['Baseline',r'$\omega_r=0$',r'$\omega_r=0.01$',r'$\omega_x=0$',r'$\omega_x=0.04$']):
    d=relax[:,k].mean(axis=0)
    ax[1,1].plot(t,d/d[0],color=colors[k],label=label)
ax[1,1].set(title='(d) State return: identical exits',xlabel='Ticks after withdrawal',ylabel=r'$G_{\mathrm{matched-none}}$')
ax[1,1].legend(frameon=False,fontsize=8,ncol=2)
for a in [ax[0,0],ax[0,1],ax[1,1]]:a.axhline(0,color='.75',lw=.6)
fig.savefig(figdir/'fig_mr2_retention.pdf');plt.close(fig)

old=np.load(root/'inputs/cohort_decomposition.npz')
confirm=np.load(root/'results/confirmation_analysis.npz')
truth=confirm['truth'];pred=confirm['prediction'];cap=confirm['capacity']
d=old['paired_effect'];mean=d.mean(axis=0)
fig,ax=plt.subplots(2,2,figsize=(6.8,6.1),layout='constrained')
for c,label in enumerate(['Total','Continuous cohort','Replacement']):
    se=d[:,c].std(axis=0,ddof=1)/np.sqrt(24)*student.ppf(.975,23)
    ax[0,0].plot(t,mean[c],color=colors[c],label=label)
    ax[0,0].fill_between(t,mean[c]-se,mean[c]+se,color=colors[c],alpha=.13,lw=0)
    area=np.trapezoid(d[:,c],axis=1)/400
    boots=area[old['bootstrap_indices']].mean(axis=1)
    lo,hi=np.quantile(boots,[.025,.975]);avg=area.mean()
    ax[0,1].bar(c,avg,color=colors[c],width=.6)
    ax[0,1].errorbar(c,avg,yerr=[[avg-lo],[hi-avg]],fmt='none',color='black',capsize=3)
ax[0,0].set(title='(a) Original baseline accounting',xlabel='Ticks after withdrawal',ylabel='Paired degree gain')
ax[0,0].legend(frameon=False,fontsize=8);ax[0,0].axhline(0,color='.6',lw=.7)
ax[0,1].set(title='(b) Time-averaged contribution',xticks=range(3),xticklabels=['Total','Cohort','Replacement'],ylabel='$A_{400}$')
ax[0,1].axhline(0,color='.6',lw=.7)
for c,a,title in [(1,ax[1,0],'(c) Original exits: continuous cohort'),(2,ax[1,1],'(d) Original exits: replacement')]:
    observed=((truth[:,:,1,c]-truth[:,:,0,c])/cap[:,None,None]*100).mean(axis=(0,1))
    predicted=((pred[:,:,1,:,c]-pred[:,:,0,:,c])/cap[:,None,None,None]*100).mean(axis=(0,1))
    a.plot(t,observed,color='black',lw=2,label='Network')
    for m in [0,2,3]:a.plot(t,predicted[m],color=colors[m],ls=['--',':','-.'][[0,2,3].index(m)],label=labels[m])
    a.set(title=title,xlabel='Ticks after withdrawal',ylabel='Paired capacity gain (%)')
    a.axhline(0,color='.7',lw=.6)
ax[1,0].legend(frameon=False,fontsize=8)
fig.savefig(figdir/'fig_mr2_cohort.pdf');plt.close(fig)

variants=confirm['variants'].tolist()
fig,axes=plt.subplots(2,3,figsize=(7.1,5.1),layout='constrained')
short=['Slow decay','Faster decay','High decay','Heterogeneous','Fewer opportunities','More opportunities']
for v,a in enumerate(axes.flat):
    observed=(truth[v,:,1,0]-truth[v,:,0,0]).mean(axis=0)
    predicted=(pred[v,:,1,:,0]-pred[v,:,0,:,0]).mean(axis=0)
    for m in range(4):a.plot(t[1:],(predicted[m,1:]-observed[1:])*100/cap[v],color=colors[m],lw=1,label=labels[m])
    a.set(title=f'({chr(97+v)}) {short[v]}',xlabel='Ticks after withdrawal',yscale='symlog')
    a.set_yscale('symlog',linthresh=.15)
    a.axhline(0,color='.6',lw=.7)
for a in axes[:,0]:a.set_ylabel('Signed capacity error (%)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.055),ncol=4,frameon=False,fontsize=9)
fig.savefig(figdir/'fig_mr2_hierarchy.pdf');plt.close(fig)

# Supplementary stress curves expose separate histories as well as cancellation.
stress=np.load(root/'results/stress_analysis.npz')
st=stress['truth'];sp=stress['prediction']
fig,axes=plt.subplots(2,3,figsize=(7.1,5.2),layout='constrained')
for v,title in enumerate(['Baseline','Strong opinion learning','Strong trust response']):
    for p in range(2):
        observed=st[v,:,p,:,bix].mean(axis=0);predicted=sp[v,:,p,:,:,1].mean(axis=0)
        a=axes[p,v]
        for m in range(4):a.plot(t[1:],(predicted[m,1:]-observed[1:])*100/6,color=colors[m],lw=1,label=labels[m])
        a.set(title=title if p==0 else '',xlabel='Ticks after withdrawal',yscale='symlog')
        a.set_yscale('symlog',linthresh=.2);a.axhline(0,color='.6',lw=.6)
axes[0,0].set_ylabel('Unmediated error (%)');axes[1,0].set_ylabel('Matched error (%)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.055),ncol=4,frameon=False,fontsize=9)
fig.savefig(figdir/'fig_mr2_stress.pdf');plt.close(fig)
print('Wrote four MR2 scientific figures.')
