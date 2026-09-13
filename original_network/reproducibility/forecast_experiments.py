"""Independent withdrawal forecasts and their prospective CPU validation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from model import Config, NetworkModel, run_segment
from experiments import METRICS, as_array
from network_forecast import FORECAST_METRICS, forecast

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "slow_decay": {"decay": .0125, "mediation_intensity": .08},
    "faster_decay": {"decay": .035, "mediation_intensity": .16},
    "high_decay": {"decay": .055, "mediation_intensity": .24},
    "heterogeneous": {"degree_sigma": .6, "mediation_intensity": .12},
    "fewer_opportunities": {"opportunity_degree": 12, "mediation_intensity": .16},
    "more_opportunities": {"opportunity_degree": 40, "mediation_intensity": .16},
}


def designs(suite):
    tasks = []
    if suite == "pilot":
        for seed in [9000, 9001]:
            for policy, intensity in [("none", 0.), ("context", .1), ("context", .4)]:
                cfg = replace(Config(), mediation_intensity=intensity)
                tasks.append(dict(seed=seed, condition=policy, variant=f"u{intensity}",
                                  suite=suite, config=cfg.to_dict()))
    else:
        for variant, changes in CASES.items():
            for seed in range(10000, 10012):
                for policy in ["none", "context"]:
                    tasks.append(dict(seed=seed, condition=policy, variant=variant,
                        suite=suite, config=replace(Config(), **changes).to_dict()))
    return tasks


def execute(task):
    cfg = Config(**task["config"])
    state = NetworkModel(cfg, task["seed"])
    run_segment(state, cfg.burnin)
    run_segment(state, cfg.intervention, task["condition"])
    state.mark_withdrawal()
    if task["suite"] == "pilot":
        predictions = [forecast(state, cfg.followup, particles=p, mode="coupled") for p in [32, 64]]
    else:
        predictions = [forecast(state, cfg.followup, particles=64, mode=mode) for mode in ["coupled", "frozen"]]
    # Forecasts are complete before generating the realized future.
    checkpoint = ROOT/"results"/"forecast_exits"/f"{task['suite']}_{task['variant']}_{task['seed']}_{task['condition']}.npz"
    state.save(checkpoint)
    actual = as_array(run_segment(state, cfg.followup))
    return np.stack(predictions), actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pilot", "confirm"], default="confirm")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = designs(args.suite)
    (ROOT/"results"/"forecast_exits").mkdir(exist_ok=True)
    specification = {"development_conditions": "baseline model, u=0,0.1,0.4; seeds 9000,9001",
        "confirmation_cases": CASES, "confirmation_seeds": list(range(10000,10012)),
        "particles": 64, "quadrature_points": 12, "fitted_coefficients": [],
        "forecast_information": ["exit weights and trust", "exit node opinions", "opportunity graph",
                                 "anchors and trust baselines", "known model parameters"],
        "future_observations_used": False,
        "primary_error": "RMSE of the ensemble mean paired degree increment over withdrawal ticks 1..400, divided by opportunity capacity",
        "secondary_errors": ["signed A400 error", "absolute degree trajectory RMSE", "strength increment RMSE"],
        "comparison": "each edge acceptance frozen at its own withdrawal value",
        "numerical_check": "32 versus 64 particles in development conditions only",
        "tasks": tasks}
    (ROOT/"results"/f"forecast_{args.suite}_manifest.json").write_text(json.dumps(specification, indent=2))
    results = [None]*len(tasks)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute,t): j for j,t in enumerate(tasks)}
        for count, future in enumerate(as_completed(futures),1):
            results[futures[future]] = future.result()
            if count % 12 == 0 or count == len(tasks):
                print(f"forecast {args.suite}: {count}/{len(tasks)}, {time.perf_counter()-start:.1f} s", flush=True)
    np.savez_compressed(ROOT/"results"/f"forecast_{args.suite}.npz",
        predictions=np.stack([r[0] for r in results]),
        trajectories=np.stack([r[1] for r in results]),
        metadata=json.dumps(tasks), metrics=json.dumps(METRICS),
        prediction_metrics=json.dumps(FORECAST_METRICS),
        prediction_methods=json.dumps(["particles32", "particles64"] if args.suite=="pilot" else ["coupled", "frozen"]))


if __name__ == "__main__":
    main()
