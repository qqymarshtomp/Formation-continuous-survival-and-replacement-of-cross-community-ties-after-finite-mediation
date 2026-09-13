"""Withdrawal forecast from an exit state, without future network observations.

Edge weight and trust distributions are propagated with independent particles.
Node opinions follow the conditional expected simultaneous update, closing
correlations between stochastic edge histories and node opinions.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from model import response_probability

FORECAST_METRICS = ["tick", "b", "strength", "trust", "opinion_gap",
                    "natural_acceptance", "survival"]


class OpinionExpectation:
    """Expected mean of successful-neighbor differences for independent edges."""
    def __init__(self, n, edge_i, edge_j, quadrature=12):
        degree = np.bincount(np.r_[edge_i, edge_j], minlength=n)
        self.edge = np.zeros((n, degree.max()), dtype=int)
        self.neighbor = np.zeros_like(self.edge)
        self.mask = np.zeros_like(self.edge, dtype=bool)
        count = np.zeros(n, dtype=int)
        for e, (i, j) in enumerate(zip(edge_i, edge_j)):
            for node, other in [(i, j), (j, i)]:
                k = count[node]
                self.edge[node, k] = e
                self.neighbor[node, k] = other
                self.mask[node, k] = True
                count[node] += 1
        z, weights = leggauss(quadrature)
        self.z, self.weights = (z+1)/2, weights/2

    def increment(self, opinions, success_probabilities):
        s = success_probabilities[self.edge]*self.mask
        difference = opinions[self.neighbor]-opinions[:, None]
        factors = 1-s[:, :, None]+s[:, :, None]*self.z
        product = factors.prod(axis=1)
        contribution = (s[:, :, None]*difference[:, :, None]/factors).sum(axis=1)
        return ((product*contribution)*self.weights).sum(axis=1)


def forecast(exit_state, duration=400, particles=32, mode="coupled", replicate=0, quadrature=12):
    """Return a conditional forecast; never steps or mutates exit_state.

    mode='frozen' holds each edge's acceptance probability at its exit value.
    mode='coupled' updates acceptance from the propagated trust and opinions.
    Particle random streams are disjoint from the realized network streams.
    """
    if mode not in {"coupled", "frozen", "mean_trust", "factorized"}:
        raise ValueError("unknown forecast mode")
    c = exit_state.cfg
    rng = np.random.default_rng(np.random.SeedSequence([exit_state.seed, 9701, replicate]))
    if mode in {"mean_trust", "factorized"}:
        entropy = [exit_state.seed, 19701] if replicate == 0 else [exit_state.seed, 19701, replicate]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
    w = np.repeat(exit_state.w[:, None], particles, axis=1)
    trust = np.repeat(exit_state.r[:, None], particles, axis=1)
    if mode == "mean_trust":
        trust = exit_state.r[:, None].copy()
    x = exit_state.x.copy()
    cx = exit_state.cx
    cohort = exit_state.w[cx] > 0
    alive = np.repeat(cohort[:, None], particles, axis=1)
    opinion_operator = OpinionExpectation(c.n, exit_state.i, exit_state.j, quadrature)
    fixed_p = exit_state.probability(np.arange(exit_state.m))[:, None]
    result = np.empty((duration+1, len(FORECAST_METRICS)))
    for tick in range(duration+1):
        difference = np.abs(x[exit_state.i]-x[exit_state.j])[:, None]
        p = fixed_p if mode == "frozen" else response_probability(
            c.intercept+c.trust_effect*trust-c.distance_effect*difference, c.response)
        result[tick] = [exit_state.tick+tick,
            2/c.n*np.mean(w[cx] > 0, axis=1).sum(),
            2/c.n*np.mean(w[cx], axis=1).sum(),
            trust[cx].mean(),
            x[exit_state.groups == 1].mean()-x[exit_state.groups == 0].mean(),
            p[cx].mean(), alive.sum()/(cohort.sum()*particles) if cohort.any() else np.nan]
        if tick == duration:
            break
        lam = c.natural_rate/c.opportunity_degree*(c.exploration+(1-c.exploration)*w)
        mean_lam = lam.mean(axis=1, keepdims=True)
        mean_p = p.mean(axis=1, keepdims=True)
        success_probability = np.mean(lam*p, axis=1) if mode in {"coupled", "frozen"} else (mean_lam*mean_p).ravel()
        dx = opinion_operator.increment(x, success_probability)
        if mode in {"coupled", "frozen"}:
            draws = rng.random((2, exit_state.m, particles))
            contact = draws[0] < lam
            success = contact & (draws[1] < p)
        else:
            draws = rng.random((3, exit_state.m, particles))
            success = draws[0] < lam*mean_p
        birth = success & (w == 0)
        strengthen = success & (w > 0)
        w[birth] = c.birth_weight
        w[strengthen] += c.reinforcement*(1-w[strengthen])
        if mode in {"coupled", "frozen"}:
            trust += c.trust_learning*contact*(success-trust)
        elif mode == "mean_trust":
            trust += c.trust_learning*mean_lam*(p-trust)
        else:
            attempted = draws[1] < mean_lam
            accepted = draws[2] < p
            trust += c.trust_learning*attempted*(accepted-trust)
        trust += c.trust_relaxation*(exit_state.r0[:, None]-trust)
        x += c.opinion_learning*dx
        x += c.opinion_anchoring*(exit_state.anchor-x)
        w *= 1-c.decay
        w[w < c.threshold] = 0
        alive &= w[cx] > 0
    return result
