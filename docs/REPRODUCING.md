# Reproducing the experiments

Run commands from the repository root unless stated otherwise. Install the pinned dependencies in `requirements.txt` using Python 3.12. All computations here use CPUs.

## Add the data archive

The lightweight repository contains programs and frozen B1/B2 plans. To reproduce the paper's numerical estimates, obtain the separately supplied `PhysicaA_Supplementary_Data_20260912.zip` (about 334 MB).

Extract it outside the repository. Inside the extracted `PhysicaA_Reproducibility/` folder, copy the `revision/`, `original_network/`, `theory/`, and `language/` folders into the repository root, **merging their contents**. Retain this repository's source files. The data archive supplies inputs and records; it contains no Python programs.

For example, on macOS/Linux, replace `/path/to/` with the actual archive location:

```sh
python -m zipfile -e /path/to/PhysicaA_Supplementary_Data_20260912.zip ../paper-data
cp -R ../paper-data/PhysicaA_Reproducibility/. .
```

After merging, `revision/inputs/main.npz` and `original_network/results/main.npz` must be present. No public repository link or DOI has been assigned in this prepared package; the archive filename identifies the required local/submission companion, not a download URL.

The supplied `.gitignore` excludes imported data and generated outputs while retaining source and the two frozen plans. Keep the original ZIP as the unchanged reference: reanalysis writes derived files in the working copy. Do not drag the data-filled working tree into GitHub's web uploader; use the clean program package for that upload.

## Reanalyse archived observations

```sh
python revision/scripts/analyze_existing.py
python revision/scripts/analyze_new.py
python revision/scripts/plot_revision.py
python revision/scripts/plot_parameter_grid.py
python language/calibration_bridge_20260911/code/analyze_batch.py
python language/calibration_bridge_B2_20260911/code/analyze_batch.py
python language/calibration_bridge_B2_20260911/code/plot_batch.py
python language/controlled_calibration_20260911/结果分析/本批分析脚本.py
```

Current revised figures are written to `revision/latex/`. B1/B2 outputs remain in their corresponding `language/` directories. Human scoring reuses the supplied A/B labels separately; it performs no new rating or adjudication.

Earlier scientific figures and theory summaries:

```sh
python theory/scripts/revision_analysis_20260911.py
python original_network/reproducibility/analyze.py
python original_network/reproducibility/analyze_cpu_extension.py
```

The theory script writes to `theory/outputs/manuscript_restructured_20260911/`; the original network scripts write to `original_network/manuscript/`. Directory dates identify frozen experimental stages. Manuscript assembly and bibliography-writing utilities are excluded from this code repository. Use the separately supplied LaTeX archive to compile the paper.

## Rerun simulations

Run these in a separate working copy if you want to retain the imported records. Rerunning writes to the same experiment output paths.

For the original suite, the design is defined in code and requires no data download:

```sh
python original_network/reproducibility/experiments.py --suite main --workers 4
```

Other original suites are selected with `--suite`; use `--help` to list them. Full-suite reruns do not guarantee exact recovery of historical mediated trajectories from seeds alone.

For the targeted revision experiments, **add the data archive first**: these runs reuse historical configuration metadata and saved confirmation exits.

```sh
OPENBLAS_NUM_THREADS=1 python revision/scripts/run_all.py
python revision/scripts/analyze_precision.py
```

The driver runs baseline/return-rate, learning, feedback stress, numerical precision, and repeated conditional futures in five stages with four processes. Logs go to `revision/logs/`; failed workers stop the driver with a nonzero exit. The precision analysis also simulates a future for each selected exit; it is not just a table formatter.

B1 and B2 need only their included frozen plans to generate new trajectories. Run all four partitions, then analyse the resulting records:

```sh
for part in 0 1 2 3; do
  python language/calibration_bridge_20260911/code/run_batch.py "$part"
done
python language/calibration_bridge_20260911/code/analyze_batch.py

for part in 0 1 2 3; do
  python language/calibration_bridge_B2_20260911/code/run_batch.py "$part"
done
python language/calibration_bridge_B2_20260911/code/analyze_batch.py
python language/calibration_bridge_B2_20260911/code/plot_batch.py
```

B1 has 144 trajectories; B2 has 1716. These are numerical realizations, not new human samples. The provided human scoring code needs the original forms and mappings in the data archive; the numerical B1/B2 drivers use the already frozen coordinates.
