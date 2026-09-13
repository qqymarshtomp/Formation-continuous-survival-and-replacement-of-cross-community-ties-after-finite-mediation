"""Run the new experiments and save every trajectory with its configuration."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from model import Config, NetworkModel, run_condition, run_withdrawal, run_segment

ROOT=Path(__file__).resolve().parents[1]
METRICS=["tick","b","strength","coverage","cross_fraction","degree","trust",
         "opinion_gap","natural_acceptance","survival","natural_births",
         "mediation_births","deletions","natural_weight","mediation_weight",
         "decay_weight","mediation_attempts","mediation_successes",
         "frame_alignment","action_alignment","natural_attempts",
         "natural_successes","count_residual","weight_residual"]


def as_array(rows):
    return np.array([[row.get(k,np.nan) for k in METRICS] for row in rows],dtype=float)


def execute(task):
    c=Config(**task["config"])
    if task["suite"] == "withdrawal_reference":
        model=NetworkModel(c,task["seed"])
        run_segment(model,c.off)
        model.mark_withdrawal()
        return [(task,as_array(run_segment(model,c.followup,natural_reinforcement=False)))]
    if task["suite"] == "withdrawal":
        checkpoint=ROOT/"results"/"withdrawal_state_seed1000.npz" if task["seed"]==1000 else None
        _,bs=run_withdrawal(c,task["seed"],checkpoint)
        return [(task|{"condition":key},as_array(rows)) for key,rows in bs.items()]
    return [(task,as_array(run_condition(c,task["seed"],task["condition"])))]


def designs():
    base=Config()
    out={}
    def task(suite,seed,condition,c,**meta):
        return dict(suite=suite,seed=seed,condition=condition,config=c.to_dict(),**meta)
    policies=["none","contact_only","action_only","fixed_action","shuffled",
              "random_pair","context"]
    out["main"]=[task("main",s,p,base) for s in range(1000,1024) for p in policies]
    out["withdrawal"]=[task("withdrawal",s,"context",base) for s in range(1000,1024)]
    out["withdrawal_reference"]=[task("withdrawal_reference",s,"no_reinforcement",base)
        for s in range(1000,1024)]
    out["long_horizon"]=[task("long_horizon",s,p,replace(base,followup=1600))
        for s in range(1000,1024) for p in ["none","context"]]
    out["scan"]=[]
    for delta in [0.008,0.015,0.025,0.04,0.06,0.09]:
        for s in range(2000,2010):
            out["scan"].append(task("scan",s,"none",replace(base,decay=delta,mediation_intensity=0)))
            for intensity in [0.05,0.10,0.20,0.40,0.80]:
                out["scan"].append(task("scan",s,"context",replace(base,decay=delta,mediation_intensity=intensity)))
    out["scale"]=[task("scale",s,p,replace(base,n=n))
        for n in [100,250,500,1000] for s in range(3000,3012) for p in ["none","context"]]
    out["structure"]=[task("structure",s,p,replace(base,mixing=mu,degree_sigma=sigma))
        for mu in [0.10,0.30,0.50] for sigma in [0.0,0.8]
        for s in range(4000,4012) for p in ["none","context"]]
    changes={"baseline":{},"probit":{"response":"probit"},
             "linear":{"response":"linear"},
             "no_input_effect":{"frame_effect":0,"action_effect":0},
             "weak_input":{"frame_effect":0.5,"action_effect":0.5},
             "weak_trust":{"trust_effect":1.4},
             "fast_trust_learning":{"trust_learning":0.30},
             "high_exploration":{"exploration":0.30}}
    out["sensitivity"]=[task("sensitivity",s,p,replace(base,**change),variant=name)
        for name,change in changes.items() for s in range(5000,5012)
        for p in ["none","contact_only","context"]]
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--suite",choices=["all",*designs()],default="all")
    args=parser.parse_args()
    (ROOT/"results").mkdir(exist_ok=True)
    suites=designs()
    manifest={"model":"context_mediation_v1","pilot_seed":101,
              "metric_names":METRICS,"base_config":Config().to_dict(),
              "designs":suites}
    (ROOT/"results"/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    for suite,tasks in suites.items():
        if args.suite not in ("all",suite):
            continue
        start=time.perf_counter()
        result=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(execute,t):idx for idx,t in enumerate(tasks)}
            for count,job in enumerate(as_completed(jobs),1):
                result.append((jobs[job],job.result()))
                if count%50==0 or count==len(tasks):
                    print(f"{suite}: {count}/{len(tasks)} tasks, {time.perf_counter()-start:.1f}s",flush=True)
        ordered=[r for _,batch in sorted(result) for r in batch]
        metadata=[r[0] for r in ordered]
        trajectories=np.stack([r[1] for r in ordered])
        np.savez_compressed(ROOT/"results"/f"{suite}.npz",trajectories=trajectories,
                            metadata=json.dumps(metadata),metrics=json.dumps(METRICS))
        print(f"Saved {suite}: {len(ordered)} trajectories",flush=True)


if __name__ == "__main__":
    main()
