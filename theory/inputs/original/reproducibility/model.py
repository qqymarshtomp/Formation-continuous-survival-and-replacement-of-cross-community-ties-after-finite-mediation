"""Adaptive ties under finite state-conditioned mediation.

The released experiments use an explicit stochastic input policy. No language
model is called. One tick is a simultaneous natural-contact sweep followed by
mediation, opinion anchoring, trust relaxation, weight decay and tie deletion.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.special import expit, ndtr


@dataclass(frozen=True)
class Config:
    n: int = 250
    opportunity_degree: int = 20
    initial_degree: int = 8
    mixing: float = 0.30
    degree_sigma: float = 0.0
    initial_cross_active: float = 0.04
    natural_rate: float = 3.0
    exploration: float = 0.10
    decay: float = 0.025
    threshold: float = 0.05
    birth_weight: float = 0.40
    reinforcement: float = 0.30
    trust_learning: float = 0.15
    trust_relaxation: float = 0.005
    trust_cross: float = 0.20
    trust_within: float = 0.60
    intercept: float = -1.0
    trust_effect: float = 2.8
    distance_effect: float = 1.0
    frame_effect: float = 1.25
    action_effect: float = 1.25
    policy_noise: float = 0.08
    opinion_learning: float = 0.025
    opinion_anchoring: float = 0.02
    response: str = "logistic"
    mediation_intensity: float = 0.40
    burnin: int = 50
    intervention: int = 50
    followup: int = 400

    @property
    def off(self):
        return self.burnin + self.intervention

    def to_dict(self):
        return asdict(self)


def response_probability(score, family="logistic"):
    if family == "logistic":
        return expit(score)
    if family == "probit":
        return ndtr(score / 1.6)
    if family == "linear":
        return np.clip(0.5 + score / 4.0, 0.01, 0.99)
    raise ValueError(f"Unknown response family: {family}")


class NetworkModel:
    def __init__(self, cfg: Config, seed: int):
        self.cfg, self.seed, self.tick = cfg, int(seed), 0
        n = cfg.n
        if n % 2 or not (0 < cfg.mixing < 1):
            raise ValueError("Use an even node count and mixing in (0,1).")
        init = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.rng_nat = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.rng_med = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.groups = np.repeat([0, 1], n // 2)
        ii, jj = np.triu_indices(n, 1)
        cross = self.groups[ii] != self.groups[jj]
        theta = np.exp(cfg.degree_sigma * init.standard_normal(n))
        weights = theta[ii] * theta[jj]
        m = n * cfg.opportunity_degree // 2
        mx = int(round(m * cfg.mixing))
        chosen = []
        for mask, count in [(cross, mx), (~cross, m - mx)]:
            candidates = np.flatnonzero(mask)
            pp = weights[candidates]
            chosen.append(init.choice(candidates, count, replace=False,
                                      p=pp / pp.sum()))
        chosen = np.sort(np.concatenate(chosen))
        self.i, self.j = ii[chosen], jj[chosen]
        self.cross = cross[chosen]
        self.cx = np.flatnonzero(self.cross)
        self.m = len(chosen)
        self.w = np.zeros(self.m)
        self.r0 = np.where(self.cross, cfg.trust_cross, cfg.trust_within)
        self.r = self.r0.copy()
        self.anchor = np.clip(np.where(self.groups == 0, -0.60, 0.60)
                              + init.normal(0, 0.12, n), -1, 1)
        self.x = self.anchor.copy()
        self.context = init.random(self.m)
        active_x = int(round(len(self.cx) * cfg.initial_cross_active))
        total_active = n * cfg.initial_degree // 2
        active = np.concatenate([
            init.choice(self.cx, active_x, replace=False),
            init.choice(np.flatnonzero(~self.cross), total_active-active_x,
                        replace=False)])
        self.w[active] = init.uniform(0.30, 0.60, len(active))
        self.cohort = None
        self.alive = None
        self.initial_cx = self.w[self.cx] > 0
        self.verify_state()

    def clone(self):
        return deepcopy(self)

    def mark_withdrawal(self):
        self.cohort = self.w[self.cx] > 0
        self.alive = self.cohort.copy()

    def probability(self, idx):
        c = self.cfg
        score = c.intercept + c.trust_effect*self.r[idx]
        score -= c.distance_effect*np.abs(self.x[self.i[idx]]-self.x[self.j[idx]])
        return response_probability(score, c.response)

    def natural_probability(self):
        c = self.cfg
        return c.natural_rate / c.opportunity_degree * (
            c.exploration + (1-c.exploration)*self.w)

    def _apply_success(self, idx, reinforce=True, allow_birth=True):
        c = self.cfg
        old = self.w[idx].copy()
        birth = old == 0
        new = old.copy()
        if allow_birth:
            new[birth] = c.birth_weight
        if reinforce:
            new[~birth] += c.reinforcement*(1-old[~birth])
        self.w[idx] = new
        is_cross = self.cross[idx]
        return int(np.count_nonzero(birth & (new > 0) & is_cross)), float(
            np.sum((new-old)[is_cross]))

    def _trust(self, attempted, success):
        c = self.cfg
        self.r[attempted] += c.trust_learning*(success-self.r[attempted])

    def _opinions(self, success_idx):
        if len(success_idx) == 0:
            return
        i, j = self.i[success_idx], self.j[success_idx]
        diff = self.x[j]-self.x[i]
        changes = np.bincount(np.r_[i,j], weights=np.r_[diff,-diff],
                              minlength=self.cfg.n)
        count = np.bincount(np.r_[i,j], minlength=self.cfg.n)
        self.x += self.cfg.opinion_learning*changes / np.maximum(count,1)

    def step(self, policy="none", natural_reinforcement=True,
             natural_birth=True, natural_contact=True):
        c = self.cfg
        b0 = int(np.count_nonzero(self.w[self.cx]))
        w0 = float(self.w[self.cx].sum())
        log = {k: 0.0 for k in ["natural_births", "mediation_births", "deletions",
               "natural_weight", "mediation_weight", "decay_weight",
               "mediation_attempts", "mediation_successes", "frame_alignment",
               "action_alignment", "natural_attempts", "natural_successes"]}
        draws = self.rng_nat.random((2, self.m))
        contact = draws[0] < self.natural_probability()
        if not natural_contact:
            contact[:] = False
        idx = np.flatnonzero(contact)
        successes = draws[1,idx] < self.probability(idx)
        good = idx[successes]
        log["natural_attempts"] = int(np.sum(self.cross[idx]))
        log["natural_successes"] = int(np.sum(self.cross[good]))
        log["natural_births"], log["natural_weight"] = self._apply_success(
            good, natural_reinforcement, natural_birth)
        self._trust(idx, successes.astype(float))
        self._opinions(good)

        if policy != "none" and c.mediation_intensity > 0:
            m = min(int(round(c.mediation_intensity*c.n/2)), len(self.cx))
            # Exponential keys implement weighted sampling without replacement.
            priority = np.ones(len(self.cx)) if policy == "random_pair" else (
                0.10 + (1-self.w[self.cx])*(1-0.5*self.r[self.cx]))
            keys = -np.log(np.maximum(self.rng_med.random(len(self.cx)),1e-15))/priority
            idx = self.cx[np.argpartition(keys,m-1)[:m]] if m else self.cx[:0]
            demand = 0.5*self.context[idx] + 0.5*(1-self.r[idx])
            frame = np.clip(self.context[idx]+self.rng_med.normal(0,c.policy_noise,m),0,1)
            action = np.clip(demand+self.rng_med.normal(0,c.policy_noise,m),0,1)
            permutation = self.rng_med.permutation(m)
            if policy == "shuffled":
                frame, action = frame[permutation], action[permutation]
            if policy == "fixed_action":
                action[:] = 0.5
            q = 1-2*np.abs(frame-self.context[idx])
            a = 1-2*np.abs(action-demand)
            if policy in ("contact_only", "action_only"):
                q[:] = 0.0
            if policy == "contact_only":
                a[:] = 0.0
            score = c.intercept + c.trust_effect*self.r[idx]
            score -= c.distance_effect*np.abs(self.x[self.i[idx]]-self.x[self.j[idx]])
            score += c.frame_effect*q + c.action_effect*a
            success = self.rng_med.random(m) < response_probability(score,c.response)
            good = idx[success]
            log["mediation_births"],log["mediation_weight"] = self._apply_success(good)
            self._trust(idx,success.astype(float))
            self._opinions(good)
            log["mediation_attempts"] = m
            log["mediation_successes"] = int(success.sum())
            log["frame_alignment"] = float(q.sum())
            log["action_alignment"] = float(a.sum())
        self.x += c.opinion_anchoring*(self.anchor-self.x)
        self.r += c.trust_relaxation*(self.r0-self.r)
        before = self.w[self.cx].sum()
        self.w *= 1-c.decay
        deleted = (self.w > 0) & (self.w < c.threshold)
        log["deletions"] = int(np.count_nonzero(deleted[self.cx]))
        self.w[deleted] = 0.0
        log["decay_weight"] = float(self.w[self.cx].sum()-before)
        if self.alive is not None:
            self.alive &= self.w[self.cx] > 0
        self.tick += 1
        b1 = int(np.count_nonzero(self.w[self.cx]))
        log["count_residual"] = b1-b0-log["natural_births"]-log["mediation_births"]+log["deletions"]
        log["weight_residual"] = self.w[self.cx].sum()-w0-sum(
            log[k] for k in ["natural_weight","mediation_weight","decay_weight"])
        return log

    def observe(self):
        wx = self.w[self.cx]
        active = self.cx[wx > 0]
        coverage = np.unique(np.r_[self.i[active],self.j[active]]).size/self.cfg.n
        nactive = np.count_nonzero(self.w)
        survival = np.nan if self.cohort is None or not np.any(self.cohort) else (
            np.count_nonzero(self.alive)/np.count_nonzero(self.cohort))
        return {"tick":self.tick,"b":2*len(active)/self.cfg.n,
                "strength":2*wx.sum()/self.cfg.n,"coverage":coverage,
                "cross_fraction":len(active)/nactive if nactive else np.nan,
                "degree":2*nactive/self.cfg.n,"trust":self.r[self.cx].mean(),
                "opinion_gap":self.x[self.groups==1].mean()-self.x[self.groups==0].mean(),
                "natural_acceptance":self.probability(self.cx).mean(),
                "survival":survival}

    def verify_state(self):
        assert len(np.unique(self.i*self.cfg.n+self.j)) == self.m
        assert np.all(self.i < self.j)
        assert np.all((self.w == 0) | ((self.w >= self.cfg.threshold)&(self.w <= 1)))
        assert np.all((self.r >= 0)&(self.r <= 1))
        assert np.all(np.abs(self.x) <= 1+1e-12)

    def save(self, path):
        arrays = {k:v for k,v in self.__dict__.items() if isinstance(v,np.ndarray)}
        metadata = {"config":self.cfg.to_dict(),"seed":self.seed,"tick":self.tick,
                    "natural_rng":self.rng_nat.bit_generator.state,
                    "mediation_rng":self.rng_med.bit_generator.state,
                    "cohort_present":self.cohort is not None}
        np.savez_compressed(path,**arrays,metadata=json.dumps(metadata))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]))
            obj=cls(Config(**meta["config"]),meta["seed"])
            for key in z.files:
                if key != "metadata":
                    setattr(obj,key,z[key].copy())
        obj.tick=meta["tick"]
        obj.rng_nat.bit_generator.state=meta["natural_rng"]
        obj.rng_med.bit_generator.state=meta["mediation_rng"]
        return obj


def run_segment(model, duration, policy="none", **switches):
    rows=[model.observe()]
    for _ in range(duration):
        log=model.step(policy,**switches)
        rows.append(model.observe() | log)
    return rows


def run_condition(cfg,seed,policy):
    model=NetworkModel(cfg,seed)
    rows=run_segment(model,cfg.burnin)
    rows += run_segment(model,cfg.intervention,policy)[1:]
    model.mark_withdrawal()
    rows[-1]["survival"]=1.0 if np.any(model.cohort) else np.nan
    rows += run_segment(model,cfg.followup)[1:]
    return rows


def run_withdrawal(cfg,seed,save_checkpoint=None):
    model=NetworkModel(cfg,seed)
    pre=run_segment(model,cfg.burnin)
    pre+=run_segment(model,cfg.intervention,"context")[1:]
    model.mark_withdrawal()
    if save_checkpoint:
        model.save(save_checkpoint)
    branches={}
    designs=[("normal",{},"none",{}),
             ("no_reinforcement",{},"none",{"natural_reinforcement":False}),
             ("no_edge_gain",{},"none",{"natural_reinforcement":False,"natural_birth":False}),
             ("decay_only",{},"none",{"natural_contact":False}),
             ("slow_decay",{"decay":cfg.decay/2},"none",{}),
             ("fast_decay",{"decay":cfg.decay*2},"none",{}),
             ("continued",{},"context",{})]
    for name,changes,policy,switches in designs:
        branch=model.clone()
        branch.cfg=replace(cfg,**changes)
        branches[name]=run_segment(branch,cfg.followup,policy,**switches)
    return pre,branches
