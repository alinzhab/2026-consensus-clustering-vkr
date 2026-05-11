from __future__ import annotations
import argparse
import csv
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSENSUS_LAB = PROJECT_ROOT / 'consensus_lab'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'
sys.path.insert(0, str(CONSENSUS_LAB))
from ensemble_selection import DATASET_TYPE_LABELS, get_dataset_type
from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from sdgca import run_sdgca
from sdgca_modified import run_sdgca_modified
ALGORITHMS = {'hierarchical_baseline': run_hierarchical_consensus, 'hierarchical_weighted': run_weighted_hierarchical_consensus, 'sdgca': run_sdgca, 'sdgca_modified': run_sdgca_modified}
REAL_DATASETS = [('Ecoli', '.mat'), ('GLIOMA', '.mat'), ('Lung', '.mat'), ('Aggregation', '.mat')]
DESIGN_DATASETS = [('design_compact_easy_5k', '.npz'), ('design_compact_easy_8k', '.npz'), ('design_overlap_moderate', '.npz'), ('design_overlap_oblong', '.npz'), ('design_imbalanced_6x', '.npz'), ('design_imbalanced_oblong', '.npz'), ('design_highdim_20d', '.npz'), ('design_highdim_40d', '.npz'), ('design_elongated_2d', '.npz'), ('design_elongated_density', '.npz'), ('design_density_varied_low_noise', '.npz'), ('design_density_varied_noisy', '.npz'), ('design_mixed_complex_6d', '.npz'), ('design_mixed_complex_branchy', '.npz')]
SMOKE_DATASETS = [('design_mini_compact', '.npz'), ('design_mini_overlap', '.npz'), ('Ecoli', '.mat')]
MINI_DATASETS = [('design_mini_compact', '.npz'), ('design_mini_overlap', '.npz'), ('design_mini_imbalanced', '.npz'), ('design_mini_highdim', '.npz'), ('design_mini_elongated', '.npz'), ('design_mini_density_varied', '.npz'), ('design_mini_mixed_complex', '.npz'), ('Ecoli', '.mat'), ('GLIOMA', '.mat'), ('Lung', '.mat')]
TSV_COLUMNS = ['experiment', 'created_at', 'dataset', 'dataset_type', 'dataset_type_label', 'algorithm', 'selection_strategy', 'qd_alpha', 'linkage', 'm', 'runs', 'seed', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean', 'f_std', 'runtime_sec']

def find_dataset(name: str, suffix: str) -> Path | None:
    for folder in (DATASETS_DIR, DATASETS_DIR / 'uploaded'):
        path = folder / f'{name}{suffix}'
        if path.exists():
            return path
    return None

def run_one(dataset_path: Path, dataset_name: str, algorithm: str, selection_strategy: str, qd_alpha: float, linkage: str, m: int, runs: int, seed: int, created_at: str) -> dict | None:
    run_fn = ALGORITHMS[algorithm]
    kwargs = {'dataset_path': dataset_path, 'data_name': dataset_name, 'seed': seed, 'm': m, 'cnt_times': runs, 'method': linkage, 'selection_strategy': selection_strategy, 'qd_alpha': qd_alpha}
    if algorithm == 'sdgca_modified':
        kwargs['diffusion_time'] = 1.0
    start = time.perf_counter()
    try:
        result = run_fn(**kwargs)
    except Exception as exc:
        print(f'[SKIP] {dataset_name}/{algorithm}/{selection_strategy}/{linkage}/seed={seed}: {exc}', flush=True)
        return None
    runtime_sec = time.perf_counter() - start
    dtype = get_dataset_type(dataset_name)
    return {'experiment': 'designed_qd', 'created_at': created_at, 'dataset': dataset_name, 'dataset_type': dtype, 'dataset_type_label': DATASET_TYPE_LABELS.get(dtype, dtype), 'algorithm': algorithm, 'selection_strategy': selection_strategy, 'qd_alpha': qd_alpha if selection_strategy == 'qd' else '', 'linkage': linkage, 'm': m, 'runs': runs, 'seed': seed, 'nmi_mean': round(float(result['nmi_mean']), 6), 'nmi_std': round(float(result['nmi_std']), 6), 'ari_mean': round(float(result['ari_mean']), 6), 'ari_std': round(float(result['ari_std']), 6), 'f_mean': round(float(result['f_mean']), 6), 'f_std': round(float(result['f_std']), 6), 'runtime_sec': round(runtime_sec, 3)}

def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def write_report(path: Path, rows: list[dict], args, tsv_path: Path) -> None:
    pairs: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row['dataset'], row['algorithm'], row['linkage'], int(row['seed']))
        pairs[key][row['selection_strategy']] = float(row['nmi_mean'])
    delta_by_algorithm: dict[str, list[float]] = defaultdict(list)
    delta_by_type: dict[tuple[str, str], list[float]] = defaultdict(list)
    wins_losses: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for (dataset, algorithm, linkage, seed), strategies in pairs.items():
        if 'random' not in strategies or 'qd' not in strategies:
            continue
        delta = strategies['qd'] - strategies['random']
        delta_by_algorithm[algorithm].append(delta)
        delta_by_type[get_dataset_type(dataset), algorithm].append(delta)
        if delta > 0.002:
            wins_losses[algorithm][0] += 1
        elif delta < -0.002:
            wins_losses[algorithm][1] += 1
        else:
            wins_losses[algorithm][2] += 1
    metric_by_type_algo: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row['selection_strategy'] == 'random':
            metric_by_type_algo[row['dataset_type'], row['algorithm']].append(float(row['nmi_mean']))
    lines = ['# Designed QD-selection experiment', '', f"Generated: {datetime.now().isoformat(timespec='seconds')}", f'TSV: `{tsv_path.name}`', '', '## Experimental logic', '', 'The experiment tests a concrete question for the thesis: whether the way we choose the base clusterings changes the quality of hierarchical consensus clustering.', '', '- `random`: the baseline protocol, selecting `m` base clusterings randomly from the pool.', '- `qd`: quality-diversity selection, choosing a deterministic subensemble that balances consensus with diversity.', '', 'The analysis is stratified by dataset type. This is important because the thesis is not trying to claim that one algorithm is universally best; it studies when each algorithm works better or worse.', '', '## Parameters', '', f'- m: `{args.m}`', f'- runs inside each algorithm call: `{args.runs}`', f'- seeds: `{args.seeds}`', f'- linkage methods: `{args.methods}`', f'- qd_alpha: `{args.qd_alpha}`', f'- algorithms: `{args.algos}`', f'- rows: `{len(rows)}`', '', '## Dataset groups and hypotheses', '', '| Dataset type | Why this type is included | Expected stress for algorithms |', '|---|---|---|', '| compact | sanity check for well separated clusters | all algorithms should be stable |', '| overlapping | ambiguous borders | consensus may smooth noise, but may also merge classes |', '| imbalanced | small clusters can be swallowed | weighted methods and SDGCA may help |', '| high_dimensional | distance concentration and feature noise | base clustering quality becomes unstable |', '| elongated | non-spherical geometry | linkage choice becomes important |', '| density_varied | uneven local density | co-association can overfit dense regions |', '| mixed_complex | several difficulties at once | QD may help by removing weak subensemble members |', '| real_bio | real benchmark structure | external sanity check beyond synthetic data |', '', '## Delta NMI: QD - random by algorithm', '', '| Algorithm | Mean delta NMI | QD wins | QD losses | Tie |', '|---|---:|---:|---:|---:|']
    for algorithm in sorted(delta_by_algorithm):
        wins, losses, ties = wins_losses[algorithm]
        lines.append(f'| {algorithm} | {mean(delta_by_algorithm[algorithm]):+.4f} | {wins} | {losses} | {ties} |')
    lines += ['', '## Delta NMI by dataset type', '', '| Dataset type | Algorithm | Mean delta NMI | Comparisons |', '|---|---|---:|---:|']
    for (dtype, algorithm), deltas in sorted(delta_by_type.items()):
        lines.append(f'| {dtype} | {algorithm} | {mean(deltas):+.4f} | {len(deltas)} |')
    lines += ['', '## Baseline random NMI by dataset type', '', '| Dataset type | Algorithm | Mean random NMI | Rows |', '|---|---|---:|---:|']
    for (dtype, algorithm), values in sorted(metric_by_type_algo.items()):
        lines.append(f'| {dtype} | {algorithm} | {mean(values):.4f} | {len(values)} |')
    lines += ['', '## Correct interpretation', '', 'Allowed conclusions:', '', '- Compare QD vs random within the same dataset, algorithm, linkage, seed, and `m`.', '- Compare behavior across dataset types rather than only one global average.', '- Discuss runtime as part of the testing system, not only quality metrics.', '', 'Not allowed without additional runs:', '', '- Claiming that QD-selection is universally better.', '- Claiming statistical significance if the number of independent seeds/datasets is too small.', '- Claiming that a result transfers to very large datasets without separate scalability tests.', '', '## Thesis contribution wording', '', 'This experiment strengthens the thesis because it turns the system from a simple runner into an analysis platform: the same algorithms are compared under controlled dataset types and under two subensemble selection protocols.']
    path.write_text('\n'.join(lines), encoding='utf-8')

def choose_datasets(profile: str) -> list[tuple[str, str]]:
    if profile == 'smoke':
        return SMOKE_DATASETS
    if profile == 'sdgca':
        return MINI_DATASETS
    if profile == 'core':
        return REAL_DATASETS + DESIGN_DATASETS
    return REAL_DATASETS + DESIGN_DATASETS

def main() -> None:
    parser = argparse.ArgumentParser(description='Run the designed QD-selection thesis experiment.')
    parser.add_argument('--profile', choices=['smoke', 'core', 'sdgca', 'full'], default='core')
    parser.add_argument('--m', type=int, default=12)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seeds', nargs='+', type=int, default=[19])
    parser.add_argument('--methods', nargs='+', default=['average'], choices=['average', 'complete', 'single', 'ward'])
    parser.add_argument('--algos', nargs='+', default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument('--qd-alpha', dest='qd_alpha', type=float, default=0.5)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    created_at = datetime.now().isoformat(timespec='seconds')
    tsv_path = RESULTS_DIR / f'designed_qd_experiment_{timestamp}.tsv'
    report_path = RESULTS_DIR / f'designed_qd_experiment_{timestamp}.md'
    latest_tsv = RESULTS_DIR / 'designed_qd_experiment_latest.tsv'
    latest_report = RESULTS_DIR / 'designed_qd_experiment_latest.md'
    datasets = choose_datasets(args.profile)
    total = len(datasets) * len(args.algos) * 2 * len(args.methods) * len(args.seeds)
    print('=' * 72)
    print('Designed QD-selection experiment')
    print(f'profile={args.profile} datasets={len(datasets)} algos={len(args.algos)} methods={args.methods}')
    print(f'm={args.m} runs={args.runs} seeds={args.seeds} qd_alpha={args.qd_alpha}')
    print(f'planned rows={total}')
    print('=' * 72)
    rows: list[dict] = []
    done = 0
    for dataset_name, suffix in datasets:
        dataset_path = find_dataset(dataset_name, suffix)
        if dataset_path is None:
            print(f'[MISS] {dataset_name}{suffix}')
            continue
        print(f'\n[DATASET] {dataset_name} type={get_dataset_type(dataset_name)}')
        for algorithm in args.algos:
            for linkage in args.methods:
                for seed in args.seeds:
                    for selection_strategy in ('random', 'qd'):
                        done += 1
                        print(f'  [{done}/{total}] {algorithm}/{selection_strategy}/{linkage}/seed={seed}', end=' ... ', flush=True)
                        row = run_one(dataset_path=dataset_path, dataset_name=dataset_name, algorithm=algorithm, selection_strategy=selection_strategy, qd_alpha=args.qd_alpha, linkage=linkage, m=args.m, runs=args.runs, seed=seed, created_at=created_at)
                        if row is None:
                            print('skip', flush=True)
                            continue
                        rows.append(row)
                        write_rows(tsv_path, rows)
                        shutil.copyfile(tsv_path, latest_tsv)
                        print(f"NMI={row['nmi_mean']:.4f}+/-{row['nmi_std']:.4f} time={row['runtime_sec']:.2f}s", flush=True)
    if not rows:
        print('[WARN] no rows generated')
        return
    write_rows(tsv_path, rows)
    write_report(report_path, rows, args, tsv_path)
    shutil.copyfile(tsv_path, latest_tsv)
    shutil.copyfile(report_path, latest_report)
    print('\n[OK] TSV:', tsv_path)
    print('[OK] report:', report_path)
    print('[OK] latest TSV:', latest_tsv)
    print('[OK] latest report:', latest_report)
    print('[OK] rows:', len(rows))
if __name__ == '__main__':
    main()
