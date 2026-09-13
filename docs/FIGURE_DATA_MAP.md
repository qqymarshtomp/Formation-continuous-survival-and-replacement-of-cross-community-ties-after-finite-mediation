# Figure, table and configuration map

Paths are relative to the combined PhysicaA_Reproducibility directory after extracting the Program and Data archives together. `revision/` holds the targeted revision experiments; `original_network/`, `theory/` and `language/` hold their predecessors. All trajectory archives store metric names and configuration/seed metadata. The exact random-stream namespaces, event order and four closure algorithms are documented in Supplementary §S6.2 and in the supplied model source.

## Main article

| Item | Scientific input and seed/configuration scope | Generator or source |
|---|---|---|
| Fig. 1 | `original_network/results/main.npz`; 24 seeds 1000–1023, seven policies, baseline u=0.4; four representative curves shown | Original `reproducibility/analyze.py`, preserved `fig2_trajectories.pdf` |
| Fig. 2(a) | `revision/results/main_normalized.npz`; same original 24 paired seeds | `revision/scripts/analyze_existing.py`, `plot_revision.py` |
| Fig. 2(b,c) | `revision/results/shuffle_*`; four rules × 16 seeds 8000–8015 × three policies, u=0.1; consistently re-executed | `run_revision.py shuffle`, `analyze_new.py`, `plot_revision.py` |
| Fig. 2(d) | `revision/exits/baseline_*`, `results/baseline_*`; 24 graph seeds, two histories, four return-rate branches plus normal | `run_revision.py baseline`, `analyze_new.py`, `plot_revision.py` |
| Fig. 3 | `theory/`: reduced-operator 12 cells and 32768 uncensored episodes; p={0.15,0.50}, δ={0.025,0.05} for direct checks | Prior `revision_theory_20260911.py`, `revision_analysis_20260911.py`; numerical block seeds in its frozen plan and arrays |
| Fig. 4(a,b) | `revision/inputs/cohort_decomposition.npz`; original 24 paired baseline histories and original 6000 resamples | Prior cohort calculation, new `plot_revision.py` |
| Fig. 4(c,d) | `revision/inputs/forecast_confirm.npz` plus `inputs/closure/exit_*`; 144 original exits, six conditions, 12 seeds 10000–10011, two histories | `analyze_existing.py`, `plot_revision.py`; per-condition capacity normalization precedes equal condition averaging |
| Fig. 5, Table 1 | Same original 144 exits; joint/frozen prospective predictions plus mean-trust/factorized retrospective predictions | `analyze_existing.py`, `plot_revision.py` (table values are in the analysis output; manuscript typesetting is excluded) |

Original confirmation conditions: slow decay (δ=0.0125,u=0.08), faster decay (0.035,0.16), high decay (0.055,0.24), heterogeneous degree (σ=0.6,u=0.12), fewer opportunities (k=12,u=0.16), more opportunities (k=40,u=0.16). Unspecified values use baseline. Every exact configuration is in the input metadata and saved exit.

## Supplementary material

| Item | Input, scope and generator |
|---|---|
| Fig. S1; Tables S1–S2 | Model event order, Config defaults and seven controller definitions; original model.py and manuscript source. No sampled estimate. |
| Tables S3–S4 | Prior fixed-response operator and direct first-passage records in `theory/`. |
| Fig. S2; Tables S5–S6 | Original confirmation forecasts and prior retrospective closure records; same six conditions and 12 seed blocks as main Table 1. |
| Table S7 | Original 24-seed cohort decomposition in `revision/inputs/cohort_decomposition.npz`. |
| Fig. S3; Table S8 | Original main-policy effects and event accounting: `original_network/results/main.npz`, `summary.json`, original `analyze.py`. |
| Fig. S4 | `withdrawal.npz`, `withdrawal_reference.npz`; original 24 seeds, matched common exits and rule-matched never-mediated references. |
| Fig. S5 | `original_network/results/scan.npz` (also `revision/inputs/scan.npz`); original 6×6 intensity/decay grid, 10 seeds per decay. Exact seed values in raw metadata and run_manifest.json. Current generator: `revision/scripts/plot_parameter_grid.py`. |
| Fig. S6 | `scale.npz`, `structure.npz`, `sensitivity.npz`; 12 paired seeds per condition, exact configurations/seed ranges in raw metadata. |
| Fig. S7 | Original `mechanism_rules.npz`; eight rules, 16 paired seeds 8000–8015; this original figure is distinct from the MR2 re-executed learning comparison. |
| Fig. S8 | Original `low_intensity.npz`; 24 seeds 7000–7023 and five primary cells, matched reference within seed and decay. Transferred from former main Fig. 2. |
| Fig. S9 | Original `long_horizon.npz`; same 24 original graph seeds, follow-up to 1600 ticks. |
| Table S9; six controlled texts | `language/`, controlled_calibration_20260911: original A/B ratings, 42 message items plus two card items per rater, six messages, 23 unique sentence strings. Human item counts are not simulation sample sizes. |
| Table S10 | Same archive, calibration_bridge_20260911: B1, 12 seeds 12000–12011, 144 trajectories; two wording families averaged within seed. |
| Figs. S10–S11; Tables S11–S13 | Same archive, calibration_bridge_B2_20260911: B2, 12 seeds 13000–13011, 143 conditions per seed, 1716 trajectories. All sign/zero controls and negative individual contrasts retained. |
| Table S14 | `revision/results/learning_analysis.*`, `learning_paired_duration_changes.json`; four re-executed learning rules and all three policies. |
| Table S15 | `revision/results/relaxation_analysis.json`; 48 common exits, 192 return-rate futures plus 48 normal futures. |
| Table S16 | `revision/results/confirmation_analysis.json`, restricted_mean_remaining; all original saved confirmation cohorts, sum over survival ticks 0–399. |
| Table S17 | `confirmation_analysis.*`; 6000 seed-block resamples [20260912,1], retaining all conditions, policies, methods and ticks within block. |
| Fig. S12; Table S18 | `results/stress_*_pred.npz` and `*_future.npz`; new seeds 16000–16011, baseline/strong opinion/strong trust, u=0.1; predictions saved before each future. |
| Tables S19–S20 | `results/numerics_*`, `quadrature_*`, `future_*`, `precision_analysis.*`; high-degree/heterogeneous seed 10000 and strong-opinion seed 16000, two policies each. P={64,128,256}, particle reps 1–4, conditional-future reps 0–31. |

## New raw records and sample units

696 newly executed network paths: 48 normal baseline re-executions, 192 common-exit return-rate branches, 192 learning histories, 72 stress futures and 192 conditional future repetitions. These are not 696 independent graph samples. New graph seeds are only 16000–16011 in the stress experiment; older graph seeds are intentionally reused for paired diagnostics.

500 distinct new scientific prediction paths: 48 baseline, 288 stress (four methods), 144 particle-resolution comparisons and 20 additional quadrature-order predictions. Quadrature files also contain 12 reused order-12 arrays for direct comparison. Original 144×4 predictions and the compatibility replays are not counted as new independent tests.

For original suites not enumerated numerically here, `docs/experiment_inventory.json` supplies exact condition/seed inventories read from raw metadata. The original design manifests retain full per-record configurations. No hashes or checksums are used.
