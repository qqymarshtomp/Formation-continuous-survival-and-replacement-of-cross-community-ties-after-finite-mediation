# Cross-community ties after finite mediation

Research code for **“Formation, continuous survival, and replacement of cross-community ties after finite mediation”** (manuscript version: 13 September 2026).

An adaptive network model examines how finite mediation forms cross-community ties, how those ties survive continuously, and how replacement ties sustain connectivity after mediation ends.

## Quick start

Use Python 3.12. From the repository root:

```sh
python -m pip install -r requirements.txt
python demo.py
```

The demo runs three policies on one shared graph seed, using the paper's baseline parameters. It saves `demo_output/trajectories.csv`, `cohorts.csv`, `config.json`, and `demo.png`. The plot shows cross-community degree and its continuous-cohort/replacement decomposition after withdrawal. This is a runnable illustration, not a reproduction of the paper's ensemble estimates. **No GPU, API key, model download, or external data is needed for the demo.**

Dependencies: NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. The tested platform is macOS arm64 with Python 3.12.

## Contents

| Directory | Purpose |
|---|---|
| `revision/scripts/` | Model, four forecast methods, learning/feedback experiments, analysis and current figures |
| `original_network/reproducibility/` | Original network suites and analyses |
| `theory/` | First-passage, renewal and cohort calculations |
| `docs/` | Reproduction instructions, limitations, experiment inventory and figure/data map |

## Reproduce the paper

Full reanalysis requires the separate **`PhysicaA_Supplementary_Data_20260912.zip`** archive, which contains the original observations, trajectories, saved states and human rating forms. Large data and generated outputs are intentionally excluded from this repository.

Follow [Reproduction instructions](docs/REPRODUCING.md). See the [figure/data map](docs/FIGURE_DATA_MAP.md), [experiment inventory](docs/experiment_inventory.json) and [reproducibility limitations](docs/LIMITATIONS.md). CPU execution is sufficient for all included experiments; full reruns are much more expensive than the demo.

## Existing scientific checks

```sh
python -m unittest discover -s revision/scripts -p 'test_*.py' -v
python language/calibration_bridge_20260911/code/verify_batch.py
python language/calibration_bridge_B2_20260911/code/verify_batch.py
```

These check model accounting, saved-state futures, forecast identities and the controlled-text policy extensions. They do not establish exact cross-platform replay of the historical experiments; see the limitations above.
