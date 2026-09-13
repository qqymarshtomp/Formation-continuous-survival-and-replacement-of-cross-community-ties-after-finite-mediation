"""Run the frozen MR2 design. Invoke stage and partition 0..3."""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from model import Config, NetworkModel, run_segment, run_condition
from experiments import as_array, METRICS
from network_forecast import forecast, FORECAST_METRICS

root = Path(__file__).resolve().parents[1]
(root / 'exits').mkdir(exist_ok=True)
(root / 'results').mkdir(exist_ok=True)
stage, partition = sys.argv[1], int(sys.argv[2])
started = time.perf_counter()
modes = ['coupled', 'frozen', 'mean_trust', 'factorized']

if stage == 'baseline':
    original = np.load(root/'inputs/main.npz')
    metadata = json.loads(original['metadata'].item())
    tasks = [(i, t) for i, t in enumerate(metadata) if t['condition'] in ['none', 'context']]
    for k in range(partition, len(tasks), 4):
        original_index, task = tasks[k]
        state = NetworkModel(Config(**task['config']), task['seed'])
        run_segment(state, 50)
        run_segment(state, 50, task['condition'])
        state.mark_withdrawal()
        name = f"baseline_{task['seed']}_{task['condition']}"
        state.save(root/'exits'/f'{name}.npz')
        prediction = forecast(state, particles=64)
        replay = as_array(run_segment(state.clone(), 400))
        observations = [METRICS.index(key) for key in state.observe()]
        difference = np.max(np.abs(replay[:, observations]-original['trajectories'][original_index, 100:, observations].T))
        relaxation = []
        changes = [{'trust_relaxation': 0.0}, {'trust_relaxation': 0.01},
                   {'opinion_anchoring': 0.0}, {'opinion_anchoring': 0.04}]
        for change in changes:
            branch = state.clone()
            branch.cfg = replace(state.cfg, **change)
            relaxation.append(as_array(run_segment(branch, 400)))
        np.savez_compressed(root/'results'/f'{name}.npz', prediction=prediction,
                            trajectories=replay, relaxation=np.stack(relaxation),
                            relaxation_changes=json.dumps(changes), metadata=json.dumps(task),
                            metrics=json.dumps(METRICS), prediction_metrics=json.dumps(FORECAST_METRICS),
                            replay_max_difference=difference)
        print(stage, partition, k, name, round(time.perf_counter()-started, 1), flush=True)

elif stage == 'shuffle':
    original = np.load(root/'inputs/mechanism_rules.npz')
    metadata = json.loads(original['metadata'].item())
    tasks = [t for t in metadata if t['condition']=='context' and t['variant'] in
             ['baseline', 'no_trust_learning', 'no_opinion_learning', 'no_learning']]
    for k in range(partition, len(tasks), 4):
        task = tasks[k] | {'condition': 'shuffled', 'suite': 'mr2_learning_shuffle'}
        result = np.stack([as_array(run_condition(Config(**task['config']), task['seed'], policy))
                           for policy in ['none','context','shuffled']])
        name = f"shuffle_{task['variant']}_{task['seed']}"
        np.savez_compressed(root/'results'/f'{name}.npz', trajectories=result,
                            metadata=json.dumps(task), policies=['none','context','shuffled'], metrics=json.dumps(METRICS))
        print(stage, partition, k, name, round(time.perf_counter()-started, 1), flush=True)

elif stage == 'stress':
    specs = [('baseline', {}), ('strong_opinion', {'opinion_learning': 0.2}),
             ('strong_trust', {'trust_effect': 5.6})]
    tasks = [(variant, changes, seed, policy) for variant, changes in specs
             for seed in range(16000, 16012) for policy in ['none', 'context']]
    for k in range(partition, len(tasks), 4):
        variant, changes, seed, policy = tasks[k]
        cfg = replace(Config(), mediation_intensity=0.1, **changes)
        state = NetworkModel(cfg, seed)
        run_segment(state, 50)
        run_segment(state, 50, policy)
        state.mark_withdrawal()
        name = f'stress_{variant}_{seed}_{policy}'
        state.save(root/'exits'/f'{name}.npz')
        task = dict(variant=variant, seed=seed, condition=policy, config=cfg.to_dict())
        prediction = np.stack([forecast(state, particles=64, mode=mode) for mode in modes])
        np.savez_compressed(root/'results'/f'{name}_pred.npz', predictions=prediction,
                            modes=modes, metadata=json.dumps(task), metrics=json.dumps(FORECAST_METRICS))
        # Future simulation begins only after all four forecasts are on disk.
        result = as_array(run_segment(state, 400))
        np.savez_compressed(root/'results'/f'{name}_future.npz', trajectories=result,
                            metadata=json.dumps(task), metrics=json.dumps(METRICS))
        print(stage, partition, k, name, round(time.perf_counter()-started, 1), flush=True)

elif stage == 'numerics':
    cases = [('high_degree', root/'inputs/forecast_exits/confirm_more_opportunities_10000'),
             ('heterogeneous', root/'inputs/forecast_exits/confirm_heterogeneous_10000'),
             ('strong_opinion', root/'exits/stress_strong_opinion_16000')]
    tasks = [(case, prefix, policy, mode, particles, replicate)
             for case, prefix in cases for policy in ['none', 'context']
             for mode in ['coupled', 'mean_trust'] for particles in [64, 128, 256] for replicate in range(1, 5)]
    for k in range(partition, len(tasks), 4):
        case, prefix, policy, mode, particles, replicate = tasks[k]
        state = NetworkModel.load(str(prefix)+f'_{policy}.npz')
        prediction = forecast(state, particles=particles, mode=mode, replicate=replicate)
        name = f'numerics_{case}_{policy}_{mode}_{particles}_{replicate}'
        np.savez_compressed(root/'results'/f'{name}.npz', prediction=prediction, metrics=json.dumps(FORECAST_METRICS))
        if particles == 256 and replicate == 1:
            maximum_degree = int(np.bincount(np.r_[state.i,state.j], minlength=state.cfg.n).max())
            orders = sorted({12, 24, max(24, (maximum_degree+1)//2)})
            high = [prediction] + [forecast(state, particles=256, mode=mode, replicate=1, quadrature=q) for q in orders[1:]]
            np.savez_compressed(root/'results'/f'quadrature_{case}_{policy}_{mode}.npz',
                                predictions=np.stack(high), orders=orders, maximum_degree=maximum_degree)
        print(stage, partition, k, name, round(time.perf_counter()-started, 1), flush=True)

elif stage == 'futures':
    cases = [('high_degree', root/'inputs/forecast_exits/confirm_more_opportunities_10000'),
             ('heterogeneous', root/'inputs/forecast_exits/confirm_heterogeneous_10000'),
             ('strong_opinion', root/'exits/stress_strong_opinion_16000')]
    tasks = [(case, prefix, policy, replicate) for case, prefix in cases
             for policy in ['none', 'context'] for replicate in range(32)]
    for k in range(partition, len(tasks), 4):
        case, prefix, policy, replicate = tasks[k]
        state = NetworkModel.load(str(prefix)+f'_{policy}.npz')
        state.rng_nat = np.random.default_rng(np.random.SeedSequence([state.seed, 21001, replicate]))
        result = as_array(run_segment(state, 400))
        np.savez_compressed(root/'results'/f'future_{case}_{policy}_{replicate:02d}.npz', trajectories=result,
                            metrics=json.dumps(METRICS), future_entropy=[state.seed, 21001, replicate])
        print(stage, partition, k, case, policy, replicate, round(time.perf_counter()-started, 1), flush=True)
else:
    raise ValueError(f'Unknown command-line stage: {stage}')
print('COMPLETED', stage, partition, flush=True)
