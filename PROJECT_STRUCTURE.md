# Project Structure (Practical, Non-Breaking)

This file explains how to keep the repository easy to navigate **without**
breaking current Flask/API paths and experiment scripts.

## Core folders

- `consensus_lab/` — algorithms, metrics, generators, diagnostics.
- `experiments/` — reproducible research scripts.
- `datasets/` — input datasets (including `massive_synthetic/`).
- `results/` — outputs consumed by web UI and reports.
- `templates/` — Flask HTML pages.
- `docs/` — thesis and protocol documentation.

## Rule of thumb

1. Keep **active canonical files** in `results/` root (used by app/API), e.g.:
   - `single_vs_consensus_benchmark.tsv`
   - `single_vs_consensus_summary.md`
   - `single_vs_consensus_by_type.tsv`
   - `single_vs_consensus_stat_tests.tsv`
   - `failure_cases.tsv`
   - `selected_consensus_profile.json`
   - `designed_qd_experiment_latest.tsv`
   - `sdgca_modified_ablation_latest.tsv`
2. Move timestamped historical artifacts into `results/archive/*`.
3. Keep plots in `results/plots/`.

## Suggested archive layout

- `results/archive/runs_json/` — timestamped run JSON files.
- `results/archive/qd_timestamped/` — timestamped designed QD files.
- `results/archive/ablation_timestamped/` — timestamped ablation files.

This keeps root clean while preserving reproducibility.
