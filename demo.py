"""CPU illustration on one graph seed; not a paper ensemble reproduction."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'revision/scripts'))
from model import Config, run_condition

cfg = Config()
seed = 20260913
policies = [('none', 'No mediation'), ('context', 'Matched'), ('shuffled', 'Shuffled')]
output = root / 'demo_output'
output.mkdir(exist_ok=True)
(output / 'config.json').write_text(json.dumps({
    'purpose': 'Single-seed demonstration; excluded from paper estimates',
    'seed': seed, 'policies': dict(policies), 'config': cfg.to_dict()
}, indent=2) + '\n')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
with (output / 'trajectories.csv').open('w', newline='') as trajectory_file, (output / 'cohorts.csv').open('w', newline='') as cohort_file:
    trajectories = csv.writer(trajectory_file)
    cohorts = csv.writer(cohort_file)
    trajectories.writerow(['policy', 'tick', 'cross_community_degree'])
    cohorts.writerow(['policy', 'ticks_after_withdrawal', 'total_degree', 'continuous_degree', 'replacement_degree'])
    for policy, label in policies:
        rows = run_condition(cfg, seed, policy)
        time = np.array([row['tick'] for row in rows])
        degree = np.array([row['b'] for row in rows])
        post = degree[cfg.off:]
        continuous = degree[cfg.off] * np.array([row['survival'] for row in rows[cfg.off:]])
        replacement = post - continuous
        trajectories.writerows((policy, int(tick), value) for tick, value in zip(time, degree))
        cohorts.writerows((policy, tick, total, survivor, newer) for tick, (total, survivor, newer) in enumerate(zip(post, continuous, replacement)))
        axes[0].plot(time, degree, label=label)
        axes[1].plot(np.arange(cfg.followup + 1), continuous)
        axes[2].plot(np.arange(cfg.followup + 1), replacement)
        print(f'{label}: withdrawal degree = {degree[cfg.off]:.3f}; final degree = {degree[-1]:.3f}')

axes[0].axvspan(cfg.burnin, cfg.off, color='gray', alpha=0.13)
axes[0].set(title='Total cross-community degree', xlabel='Tick', ylabel='Mean degree')
axes[1].set(title='Continuously surviving exit ties', xlabel='Ticks after withdrawal', ylabel='Mean degree')
axes[2].set(title='Replacement ties', xlabel='Ticks after withdrawal', ylabel='Mean degree')
fig.suptitle('One graph seed · shaded interval denotes mediation', y=0.99, fontsize=12)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.82))
fig.savefig(output / 'demo.png', dpi=150)
plt.close(fig)
print('Saved demo_output/trajectories.csv, cohorts.csv, config.json, and demo.png')
