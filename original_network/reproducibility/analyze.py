"""Rebuild all manuscript figures, tables, and numeric statements from raw runs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import t as student_t
from model import Config

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"manuscript"
OUT.mkdir(exist_ok=True)
(ROOT/"qa").mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix",
    "font.size":10,"axes.labelsize":11,"axes.titlesize":11,"legend.fontsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.7,
    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":220})
COLORS={"none":"#656B73","contact_only":"#C47A27","action_only":"#9C5C82",
        "fixed_action":"#977638","shuffled":"#BF5554","random_pair":"#498A76",
        "context":"#126D94"}
LABELS={"none":"No mediation","contact_only":"Contact only","action_only":"Action only",
        "fixed_action":"Fixed action","shuffled":"Shuffled inputs","random_pair":"Random pairs",
        "context":"Context matched"}
DATA={}
for suite in ["main","withdrawal","withdrawal_reference","scan","scale","structure","sensitivity","long_horizon"]:
    z=np.load(ROOT/"results"/f"{suite}.npz",allow_pickle=False)
    DATA[suite]=(z["trajectories"],json.loads(str(z["metadata"])),json.loads(str(z["metrics"])))
METRICS=DATA["main"][2]
J={k:i for i,k in enumerate(METRICS)}


def get(suite,condition,**filters):
    a,meta,_=DATA[suite]
    ids=[i for i,m in enumerate(meta) if m["condition"]==condition and all(
         m.get(k,m["config"].get(k))==v for k,v in filters.items())]
    ids.sort(key=lambda i:meta[i]["seed"])
    return a[ids]


def ci(values):
    values=np.asarray(values,float)
    rng=np.random.default_rng(9001)
    means=values[rng.integers(0,len(values),(6000,len(values)))].mean(axis=1)
    lo,hi=np.quantile(means,[0.025,0.975])
    return {"mean":float(values.mean()),"lo":float(lo),"hi":float(hi),"n":len(values)}


def effect(a,b,metric="b",off=100):
    diff=a[:,:,J[metric]]-b[:,:,J[metric]]
    return {"off":ci(diff[:,off]),"A":ci(np.trapezoid(diff[:,off:],axis=1)/400),
            "final":ci(diff[:,-1])}


def band(ax,arr,metric,label,color,shift=0,ls="-"):
    x=arr[0,:,J["tick"]]-shift
    y=arr[:,:,J[metric]]
    mean=y.mean(axis=0)
    half=student_t.ppf(.975,len(y)-1)*y.std(axis=0,ddof=1)/np.sqrt(len(y))
    ax.plot(x,mean,label=label,color=color,lw=1.55,ls=ls)
    ax.fill_between(x,mean-half,mean+half,color=color,alpha=.13,lw=0)


def save(fig,name):
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(ROOT/"qa"/f"{name}.png",bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    ax.grid(alpha=.13,axis="y",lw=.5)
    ax.tick_params(labelsize=9)


base=get("main","none")
summary={"main":{p:effect(get("main",p),base) for p in LABELS if p!="none"}}
summary["contrasts"]={p:effect(get("main","context"),get("main",p)) for p in
    ["contact_only","shuffled","fixed_action","random_pair"]}
summary["absolute"]={p:{metric:{"off":ci(get("main",p)[:,100,J[metric]]),
    "final":ci(get("main",p)[:,-1,J[metric]])} for metric in
    ["b","strength","coverage","opinion_gap","trust"]} for p in ["none","context"]}
summary["withdrawal"]={}
normal=get("withdrawal","normal")
for p in ["normal","no_reinforcement","no_edge_gain","decay_only","slow_decay","fast_decay","continued"]:
    a=get("withdrawal",p)
    summary["withdrawal"][p]={"b_final":ci(a[:,-1,J["b"]]),
        "survival_final":ci(a[:,-1,J["survival"]]),
        "strength_final":ci(a[:,-1,J["strength"]]),
        "normal_minus_branch":effect(normal,a,off=0)}
med_nr=get('withdrawal','no_reinforcement')
base_nr=get('withdrawal_reference','no_reinforcement')
policy_normal=normal-base[:,100:,:]
policy_nr=med_nr-base_nr
summary['reinforcement_interaction']={metric:effect(policy_normal,policy_nr,metric,off=0)
    for metric in ['b','strength']}

# Figure 1: exact event sequence and the two response channels.
fig,ax=plt.subplots(figsize=(8.8,3.35))
ax.set(xlim=(0,10),ylim=(0,3.7));ax.axis("off")
boxes=[(.1,2.28,2.0,.95,"State at tick $t$\nOpinions, trust,\ntie weights"),
       (2.7,2.28,2.2,.95,"Natural contacts\nWeight-dependent\ncontact probability"),
       (5.5,2.28,2.1,.95,"Active mediation\nPair selection\nFraming + action"),
       (7.8,.55,2.0,.95,"State at tick $t+1$\nRetained and\nnewly formed ties"),
       (3.65,.55,3.2,.95,"Relaxation and decay\nOpinion and trust baselines\nWeight loss and tie deletion")]
for x,y,w,h,txt in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07",fc="#EEF4F7",ec="#49778A",lw=1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9.2)
for start,end in [((2.16,2.75),(2.62,2.75)),((4.97,2.75),(5.42,2.75)),
                   ((6.5,2.2),(5.4,1.56)),((6.95,1.0),(7.72,1.0))]:
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,color="#36596B"))
ax.text(.1,1.75,"Successful responses\ncreate or strengthen ties;\nall attempts update trust.",ha="left",va="top",fontsize=9)
ax.text(.05,.1,"Each contact stage updates trust, tie weights, and opinions before the next stage.",fontsize=9.5)
save(fig,"fig1_model")

# Figure 2: main time courses.
fig,axes=plt.subplots(2,2,figsize=(7.0,5.6),sharex=True)
for ax,metric,ylabel,panel in zip(axes.ravel(),["b","strength","coverage","opinion_gap"],
    ["Mean cross-community degree $b$","Cross-community strength $s_\\times$",
     "Cross-community node coverage","Between-community opinion gap"],"abcd"):
    for p in ["none","contact_only","shuffled","context"]:
        band(ax,get("main",p),metric,LABELS[p],COLORS[p])
    ax.axvspan(50,100,color="#C7D7C5",alpha=.33,lw=0)
    ax.axvline(100,color="#50604C",ls=":",lw=1)
    ax.set_ylabel(ylabel);ax.set_title(f"({panel})",loc="left");clean(ax)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,fontsize=10)
for ax in axes[1]:ax.set_xlabel("Time (ticks)")
fig.tight_layout(rect=[0,0,1,.91])
save(fig,"fig2_trajectories")

# Figure 3: policy contrasts and measured topology accounting.
fig=plt.figure(figsize=(7.0,6.0))
gs=fig.add_gridspec(2,2,height_ratios=[1.1,1.0])
axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
policies=["contact_only","action_only","fixed_action","shuffled","random_pair","context"]
for ax,key,title in zip(axes[:2],["off","A"],["(a) Gain at withdrawal","(b) Mean gain over 400 ticks"]):
    for k,p in enumerate(policies):
        v=summary["main"][p][key]
        ax.errorbar(v["mean"],k,xerr=[[v["mean"]-v["lo"]],[v["hi"]-v["mean"]]],
                    fmt="o",color=COLORS[p],capsize=3,ms=5)
    ax.set_yticks(range(len(policies)),[LABELS[p] for p in policies] if ax is axes[0] else [])
    ax.invert_yaxis();ax.axvline(0,color=".7",lw=.8);ax.set_title(title)
    ax.set_xlabel("$D_b(0)$" if key=="off" else "$A_{400}$");clean(ax)
ax=axes[2]
for k,p in enumerate(["none","context"]):
    ar=get("main",p)[:,51:101,:]
    vals=[ar[:,:,J[m]].sum(axis=1)*2/250 for m in ["natural_births","mediation_births","deletions"]]
    vals[-1]*=-1
    for j,v in enumerate(vals):
        ax.bar(k+(j-1)*.23,v.mean(),width=.21,color=["#478F7A","#126D94","#BF5554"][j],
               label=["Natural creation","Mediated creation","Deletion"][j] if k==0 else None)
ax.axhline(0,c=".4",lw=.7);ax.set_xticks([0,1],["No mediation","Context matched"])
ax.set_title("(c) Changes during intervention");ax.set_ylabel("Contribution to $b$")
ax.legend(frameon=False,fontsize=10,loc="upper left");clean(ax)
fig.tight_layout(w_pad=1.4)
save(fig,"fig3_mechanisms")

# Figure 4: paired parameter grid.
deltas=[.008,.015,.025,.04,.06,.09];ints=[0,.05,.10,.20,.40,.80]
hm1=np.zeros((6,6));hm2=np.zeros((6,6));scan=[]
for i,d in enumerate(deltas):
    control=get("scan","none",decay=d)
    for j,u in enumerate(ints):
        a=control if not u else get("scan","context",decay=d,mediation_intensity=u)
        e=effect(a,control);hm1[i,j]=e["off"]["mean"];hm2[i,j]=e["A"]["mean"]
        scan.append({"decay":d,"intensity":u,**e})
summary["scan"]=scan
fig,axes=plt.subplots(1,2,figsize=(7.2,3.8))
for ax,arr,title in zip(axes,[hm1,hm2],["(a) Gain at withdrawal $D_b(0)$","(b) Mean post-withdrawal gain $A_{400}$"]):
    im=ax.imshow(arr,origin="lower",aspect="auto",cmap="viridis",vmin=0)
    ax.set_xticks(range(6),[str(x) for x in ints]);ax.set_yticks(range(6),[str(x) for x in deltas])
    ax.set_xlabel("Mediation intensity $u$");ax.set_ylabel("Weight-decay fraction $\\delta$")
    ax.set_title(title)
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=10,
                    color="white" if arr[i,j]<arr.max()*.6 else "#17313A")
    fig.colorbar(im,ax=ax,fraction=.045,pad=.025)
fig.tight_layout()
save(fig,"fig4_parameter_regions")

# Figure 5: withdrawal and cohort survival.
fig,axes=plt.subplots(2,2,figsize=(7.0,6.2),sharex=True)
branches=[("normal","Normal withdrawal","#126D94"),
          ("no_reinforcement","No strengthening","#C47A27"),
          ("slow_decay","Half decay","#498A76"),
          ("fast_decay","Double decay","#BF5554"),
          ("continued","Continued mediation","#8C729A")]
for name,label,color in branches:
    a=get("withdrawal",name)
    band(axes[0,0],a,"b",label,color,shift=100)
    band(axes[1,0],a,"survival",label,color,shift=100)
band(axes[0,0],base[:,100:,:],"b","Never mediated","#555B62",shift=100,ls="--")
band(axes[1,0],get("withdrawal","decay_only"),"survival","Decay only","#555B62",shift=100,ls="--")
for metric,color,label in [("b","#126D94","Degree increment"),("strength","#C47A27","Strength increment")]:
    a=normal.copy()
    a[:,:,J[metric]]=policy_normal[:,:,J[metric]]-policy_nr[:,:,J[metric]]
    band(axes[0,1],a,metric,label,color,shift=100)
for name,label,color in [("normal","Normal withdrawal","#126D94"),
                         ("no_edge_gain","No natural tie gains","#BF5554"),
                         ("decay_only","Decay only","#555B62")]:
    band(axes[1,1],get("withdrawal",name),"strength",label,color,shift=100,
         ls="--" if name=="decay_only" else "-")
for ax,title,y in zip(axes.ravel(),["(a) Cross-community degree","(b) Strengthening and mediation",
    "(c) Survival of withdrawal ties","(d) Weight retained after withdrawal"],
    ["$b$","Difference in mediation effects","Fraction of initial cohort","$s_\\times$"]):
    ax.set_title(title,loc="left");ax.set_ylabel(y);clean(ax)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,fontsize=10,frameon=False,ncol=3,loc="upper center")
axes[0,1].legend(fontsize=9,frameon=False);axes[1,1].legend(fontsize=9,frameon=False)
for ax in axes[1]:ax.set_xlabel("Time since withdrawal (ticks)")
fig.tight_layout(rect=[0,0,1,.91])
save(fig,"fig5_withdrawal")

# Figure 6: scale, opportunity structure, and alternative response assumptions.
fig=plt.figure(figsize=(7.0,6.5))
gs=fig.add_gridspec(2,2,height_ratios=[1,1.2])
axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
summary["scale"]=[]
for n in [100,250,500,1000]:
    e=effect(get("scale","context",n=n),get("scale","none",n=n))
    summary["scale"].append({"n":n,**e});v=e["A"]
    axes[0].errorbar(n,v["mean"],yerr=[[v["mean"]-v["lo"]],[v["hi"]-v["mean"]]],fmt="o",capsize=3,color="#126D94")
axes[0].set(xscale="log",xlabel="Nodes $N$",ylabel="$A_{400}$",title="(a) Network size")
axes[0].set_xticks([100,250,500,1000],["100","250","500","1000"])
summary["structure"]=[]
for sig,label,color in [(0,"Uniform node weights","#126D94"),(.8,"Unequal node weights","#C47A27")]:
    vv=[]
    for mu in [.1,.3,.5]:
        e=effect(get("structure","context",mixing=mu,degree_sigma=sig),get("structure","none",mixing=mu,degree_sigma=sig))
        summary["structure"].append({"mixing":mu,"degree_sigma":sig,**e})
        vv.append({k:(v/(20*mu) if k in ["mean","lo","hi"] else v) for k,v in e["A"].items()})
    axes[1].errorbar([.1,.3,.5],[v['mean'] for v in vv],
        yerr=[[v['mean']-v['lo'] for v in vv],[v['hi']-v['mean'] for v in vv]],
        fmt="o-",capsize=3,color=color,label=label)
axes[1].set(xlabel="Cross-community opportunity fraction",ylabel="$A_{400}/(k_o\\mu)$",title="(b) Fraction of available capacity")
axes[1].legend(frameon=False,fontsize=9,loc="upper right")
variants=[("baseline","Baseline"),("probit","Probit response"),("linear","Linear response"),
          ("no_input_effect","Input effects = 0"),("weak_input","Weaker input effects"),
          ("weak_trust","Weaker trust effect"),("fast_trust_learning","Faster trust learning"),
          ("high_exploration","Higher exploration")]
summary["sensitivity"]=[]
for k,(key,label) in enumerate(variants):
    e=effect(get("sensitivity","context",variant=key),get("sensitivity","contact_only",variant=key))
    ebase=effect(get("sensitivity","context",variant=key),get("sensitivity","none",variant=key))
    summary["sensitivity"].append({"variant":key,"vs_contact":e,"vs_none":ebase})
    v=e["A"]
    axes[2].errorbar(v["mean"],k,xerr=[[v['mean']-v['lo']],[v['hi']-v['mean']]],fmt="o",capsize=3,color="#126D94")
axes[2].set_yticks(range(8),[v[1] for v in variants]);axes[2].invert_yaxis()
axes[2].set(xlabel="Context minus contact-only $A_{400}$",title="(c) Response and feedback assumptions")
axes[2].axvline(0,color=".6",lw=.8)
for ax in axes:clean(ax)
fig.tight_layout(w_pad=2)
save(fig,"fig6_robustness")

# Figure 7: independent frozen-response benchmark and local drift condition.
theory=json.loads((ROOT/"results"/"theory_benchmark.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(7.0,3.5))
for delta,color in [(.0125,"#498A76"),(.025,"#126D94"),(.05,"#BF5554")]:
    tt=sorted([t for t in theory if t['decay']==delta],key=lambda r:r['p'])
    axes[0].plot([t['p'] for t in tt],[t['operator_1201']['active'] for t in tt],color=color,lw=1.4)
    axes[0].errorbar([t['p'] for t in tt],[np.mean(t['mc_active']) for t in tt],
        yerr=[student_t.ppf(.975,15)*np.std(t['mc_active'],ddof=1)/4 for t in tt],
        fmt="o",capsize=3,color=color,label=f"$\\delta={delta}$")
axes[0].set(xlabel="Frozen acceptance probability $p$",ylabel="Stationary active fraction",
            title="(a) Transfer operator and Monte Carlo")
axes[0].legend(frameon=False)
w=np.linspace(.06,.95,250);c=Config()
for delta,color in [(.0125,"#498A76"),(.025,"#126D94"),(.05,"#BF5554")]:
    pc=delta*w/((1-delta)*(c.natural_rate/c.opportunity_degree)*c.reinforcement*
               (c.exploration+(1-c.exploration)*w)*(1-w))
    axes[1].plot(w,pc,color=color,label=f"$\\delta={delta}$")
axes[1].axhline(1,c=".6",ls=":",lw=1)
axes[1].set(xlabel="Current tie weight $w$",ylabel="Acceptance required for positive drift",
            ylim=(0,1.6),title="(b) Reinforcement-decay balance")
axes[1].legend(frameon=False)
for ax in axes:clean(ax)
fig.tight_layout()
save(fig,"fig7_theory")
summary["theory"]={"max_active_error":max(abs(t['operator_1201']['active']-np.mean(t['mc_active'])) for t in theory),
    "max_grid_change":max(abs(t['operator_1201']['active']-t['operator_601']['active']) for t in theory),
    "benchmark_conditions":len(theory)}

# Extended follow-up uses the same realizations, not additional independent seeds.
late=get('long_horizon','context')-get('long_horizon','none')
late[:,:,J['tick']]=get('long_horizon','context')[:,:,J['tick']]
late=late[:,100:,:]
summary['long_horizon']={'last_point':ci(late[:,-1,J['b']]),
    'mean_1200_1600':ci(np.trapezoid(late[:,1200:,J['b']],axis=1)/400)}
fig,axes=plt.subplots(1,2,figsize=(7.0,3.3))
band(axes[0],late,'b','Paired degree increment','#126D94',shift=100)
band(axes[1],late[:,900:,:],'b','Paired degree increment','#126D94',shift=100)
for ax in axes:
    ax.axhline(0,color='.5',ls=':',lw=.8);ax.set_xlabel('Time since withdrawal (ticks)')
    ax.set_ylabel('$D_b(\\tau)$');clean(ax)
axes[0].set_title('(a) Extended observation');axes[1].set_title('(b) Late interval')
fig.tight_layout();save(fig,'figS1_long_horizon')

# Tables and values are emitted directly from the same summaries used in plots.
lines=[r"\begin{tabular}{lrrr}",r"\toprule",r"Policy & $D_b(0)$ & $A_{400}$ & $D_b(400)$ \\",r"\midrule"]
for p,e in summary['main'].items():
    values=[f"{e[k]['mean']:.3f} [{e[k]['lo']:.3f}, {e[k]['hi']:.3f}]" for k in ["off","A","final"]]
    lines.append(LABELS[p]+" & "+" & ".join(values)+r" \\")
lines.extend([r"\bottomrule",r"\end{tabular}"])
(OUT/"table_effects.tex").write_text("\n".join(lines))
macros={}
for name,key in [("Off","off"),("Area","A"),("Final","final")]:
    for part in ['mean','lo','hi']:
        macros['Context'+name+part.title()]=summary['main']['context'][key][part]
for name,p in [("Shuffle","shuffled"),("Fixed","fixed_action"),("Random","random_pair"),("Contact","contact_only")]:
    for part in ['mean','lo','hi']:
        macros['Vs'+name+'Area'+part.title()]=summary['contrasts'][p]['A'][part]
for name,p in [("Normal","normal"),("Slow","slow_decay"),("Continued","continued")]:
    macros[name+'Survival']=summary['withdrawal'][p]['survival_final']['mean']*100
    macros[name+'FinalB']=summary['withdrawal'][p]['b_final']['mean']
for name,p in [('Context','context'),('Base','none')]:
    macros[name+'OffB']=summary['absolute'][p]['b']['off']['mean']
    macros[name+'OffGap']=summary['absolute'][p]['opinion_gap']['off']['mean']
    macros[name+'OffCoverage']=summary['absolute'][p]['coverage']['off']['mean']*100
macros['TheoryError']=summary['theory']['max_active_error']
macros['GridError']=summary['theory']['max_grid_change']
macros['ReinforceArea']=summary['withdrawal']['no_reinforcement']['normal_minus_branch']['A']['mean']
for part in ['mean','lo','hi']:
    macros['ReinforceInteraction'+part.title()]=summary['reinforcement_interaction']['b']['A'][part]
(OUT/'results_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.3f}'+'}' for k,v in macros.items()))
(ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'main':summary['main'],'contrasts':summary['contrasts'],
                  'theory':summary['theory']},indent=2))
