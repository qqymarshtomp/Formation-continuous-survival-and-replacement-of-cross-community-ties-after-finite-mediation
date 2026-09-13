"""Prespecified CPU extension: low intensity and mechanism assumptions."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from model import Config, run_condition
from experiments import METRICS, as_array

ROOT = Path(__file__).resolve().parents[1]
LOW_CELLS = [(0.025, 0.05), (0.025, 0.10), (0.025, 0.20),
             (0.015, 0.10), (0.040, 0.10)]
RULES = {
    "baseline": {},
    "lower_threshold": {"threshold": 0.025},
    "higher_threshold": {"threshold": 0.10},
    "fewer_opportunities": {"opportunity_degree": 12},
    "more_opportunities": {"opportunity_degree": 40},
    "no_trust_learning": {"trust_learning": 0.0},
    "no_opinion_learning": {"opinion_learning": 0.0},
    "no_learning": {"trust_learning": 0.0, "opinion_learning": 0.0},
}


def designs():
    result = {"low_intensity": [], "mechanism_rules": []}
    base = Config()
    def add(suite, seed, policy, cfg, **extra):
        result[suite].append(dict(suite=suite, seed=seed, condition=policy,
                                  config=cfg.to_dict(), **extra))
    for seed in range(7000, 7024):
        for delta in sorted({d for d, u in LOW_CELLS}):
            add("low_intensity", seed, "none", replace(base, decay=delta,
                mediation_intensity=0))
        for delta, intensity in LOW_CELLS:
            cfg = replace(base, decay=delta, mediation_intensity=intensity)
            for policy in ["contact_only", "shuffled", "random_pair", "context"]:
                add("low_intensity", seed, policy, cfg)
    for variant, changes in RULES.items():
        cfg = replace(base, mediation_intensity=0.10, **changes)
        for seed in range(8000, 8016):
            for policy in ["none", "context"]:
                add("mechanism_rules", seed, policy, cfg, variant=variant)
    return result


def execute(task):
    rows = run_condition(Config(**task["config"]), task["seed"], task["condition"])
    return as_array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--suite", choices=["all", "low_intensity", "mechanism_rules"], default="all")
    args = parser.parse_args()
    suites = designs()
    manifest = {"design_fixed_before_execution": True,
        "primary_metric": "paired trapezoidal mean degree increment over 400 withdrawal ticks",
        "secondary_metrics": ["degree increment at withdrawal", "capacity occupancy",
                              "withdrawal-cohort survival", "creation and deletion fluxes"],
        "low_intensity_primary_contrasts": ["context-contact_only", "context-shuffled", "context-random_pair"],
        "rule_contrast": "context-none within each rule, then difference from the baseline-rule contrast",
        "uncertainty": "6000 paired seed bootstrap resamples; intervals are pointwise, not simultaneous",
        "shared_baselines": "one never-mediated trajectory per seed and decay in low_intensity",
        "model_version": "unchanged original numerical controller and transition rules",
        "designs": suites}
    (ROOT/"results"/"cpu_extension_manifest.json").write_text(json.dumps(manifest, indent=2))
    for suite, tasks in suites.items():
        if args.suite not in ["all", suite]:
            continue
        started = time.perf_counter()
        result = [None]*len(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute, task): j for j, task in enumerate(tasks)}
            for count, future in enumerate(as_completed(futures), 1):
                result[futures[future]] = future.result()
                if count % 40 == 0 or count == len(tasks):
                    print(f"{suite}: {count}/{len(tasks)}, {time.perf_counter()-started:.1f} s", flush=True)
        np.savez_compressed(ROOT/"results"/f"{suite}.npz", trajectories=np.stack(result),
                            metadata=json.dumps(tasks), metrics=json.dumps(METRICS))


if __name__ == "__main__":
    main()
