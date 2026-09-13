"""Frozen-response transfer operator and independent Monte Carlo benchmark."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.sparse import csc_matrix
from model import Config

ROOT=Path(__file__).resolve().parents[1]


def transfer_operator(p,delta,cells=601,cfg=None):
    c=cfg or Config()
    grid=np.r_[0.0,np.linspace(c.threshold,1.0,cells-1)]
    success=c.natural_rate/c.opportunity_degree*(c.exploration+(1-c.exploration)*grid)*p
    failure=(1-delta)*grid
    gain=np.where(grid==0,c.birth_weight,grid+c.reinforcement*(1-grid))*(1-delta)
    rows=[];cols=[];vals=[]
    for j in range(cells):
        for y,prob in [(failure[j],1-success[j]),(gain[j],success[j])]:
            if y < c.threshold:
                rows.append(0);cols.append(j);vals.append(prob)
            else:
                k=min(np.searchsorted(grid,y,side="right")-1,cells-2)
                frac=(y-grid[k])/(grid[k+1]-grid[k])
                rows.extend([k,k+1]);cols.extend([j,j]);vals.extend([prob*(1-frac),prob*frac])
    return grid,csc_matrix((vals,(rows,cols)),shape=(cells,cells))


def stationary(p,delta,cells=601):
    grid,T=transfer_operator(p,delta,cells)
    dist=np.zeros(len(grid));dist[0]=1
    for it in range(60000):
        nxt=T@dist
        if np.abs(nxt-dist).sum()<1e-12:
            dist=nxt;break
        dist=nxt
    return {"active":float(1-dist[0]),"weight":float(grid@dist),
            "iterations":it+1,"mass_error":float(abs(dist.sum()-1))}


def benchmark(task):
    p,delta,seed=task
    c=Config()
    rng=np.random.default_rng(seed)
    blocks,dyads=16,256
    w=np.zeros((blocks,dyads))
    act=np.zeros(blocks);wei=np.zeros(blocks)
    burn,steps=6000,2000
    for t in range(burn+steps):
        prob=c.natural_rate/c.opportunity_degree*(c.exploration+(1-c.exploration)*w)*p
        ok=rng.random(w.shape)<prob
        new=ok&(w==0)
        live=ok&(w>0)
        w[new]=c.birth_weight
        w[live]+=c.reinforcement*(1-w[live])
        w*=1-delta
        w[w<c.threshold]=0
        if t>=burn:
            act+=(w>0).mean(axis=1)/steps
            wei+=w.mean(axis=1)/steps
    return {"p":p,"decay":delta,"seed":seed,"blocks":blocks,"dyads_per_block":dyads,
            "burnin":burn,"measurement":steps,"operator_601":stationary(p,delta,601),
            "operator_1201":stationary(p,delta,1201),
            "mc_active":act.tolist(),"mc_weight":wei.tolist()}


if __name__=="__main__":
    tasks=[(p,d,6000+i*3+j) for i,p in enumerate([0.15,0.30,0.50,0.70])
           for j,d in enumerate([0.0125,0.025,0.05])]
    with ProcessPoolExecutor(max_workers=4) as ex:
        result=list(ex.map(benchmark,tasks))
    (ROOT/"results"/"theory_benchmark.json").write_text(json.dumps(result,indent=2))
    for r in result:
        print(r['p'],r['decay'],round(r['operator_1201']['active'],4),
              round(np.mean(r['mc_active']),4),flush=True)
