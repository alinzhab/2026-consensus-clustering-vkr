# Experiments Folder Guide

Scripts are grouped by purpose:

## Dataset generation

- `generate_design_datasets.py` — small designed suite.
- `generate_massive_synthetic_suite.py` — large stratified suite + manifest.

## Benchmarks and comparisons

- `run_single_vs_consensus_benchmark.py` — main dataset-level benchmark.
- `run_designed_qd_experiment.py` — QD-focused designed experiment.
- `run_qd_selection_analysis.py` — QD random vs qd analysis.
- `run_sdgca_modified_ablation.py` — A/B/C... ablation for SDGCA modified.
- `run_default_vs_tuned.py` — tuned vs generic diagnostic (not main claim).

## Model/profile selection and final analysis

- `select_consensus_hyperparameters.py` — choose global profile on validation.
- `analyze_single_vs_consensus.py` — final test analysis/statistics.

## Utilities

- `merge_designed_experiment_results.py`
- `plot_results.py`
- `run_statistical_analysis.py`
- `run_runs_stability.py`
- `run_complexity_benchmark.py`
