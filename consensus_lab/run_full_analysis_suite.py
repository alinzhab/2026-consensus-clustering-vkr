from __future__ import annotations
import argparse
import csv
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from sdgca import run_sdgca
from sdgca_modified import resolve_params, run_sdgca_modified
from io_utils import append_tsv_row
ALGORITHMS = ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified']
METHODS = ['average', 'complete', 'single', 'ward']
HEADER = ['dataset', 'algorithm', 'method', 'seed', 'm', 'runs', 'n_objects', 'n_members', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean', 'f_std', 'seconds', 'status', 'error']

def discover_datasets(root: Path, prefix: str) -> list[Path]:
    paths = sorted(root.glob(f'{prefix}*.npz')) + sorted(root.glob(f'{prefix}*.mat'))
    return [path for path in paths if 'smoke' not in path.stem.lower()]

def dataset_shape(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == '.mat':
        data = loadmat(path)
        members = np.asarray(data['members'])
    else:
        data = np.load(path, allow_pickle=True)
        members = np.asarray(data['members'])
    return (int(members.shape[0]), int(members.shape[1]))

def load_completed(path: Path) -> set[tuple[str, str, str, str, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if row.get('status') == 'ok':
                completed.add((row['dataset'], row['algorithm'], row['method'], row['m'], row['runs']))
    return completed

def run_one(path: Path, algorithm: str, method: str, seed: int, m: int, runs: int, sharpen: float) -> dict:
    started = time.time()
    dataset = path.stem
    n_objects, n_members = dataset_shape(path)
    effective_m = min(m, n_members)
    row = {'dataset': dataset, 'algorithm': algorithm, 'method': method, 'seed': seed, 'm': effective_m, 'runs': runs, 'n_objects': n_objects, 'n_members': n_members, 'nmi_mean': '', 'nmi_std': '', 'ari_mean': '', 'ari_std': '', 'f_mean': '', 'f_std': '', 'seconds': '', 'status': 'ok', 'error': ''}
    try:
        if algorithm == 'hierarchical_baseline':
            result = run_hierarchical_consensus(path, dataset, seed=seed, m=effective_m, cnt_times=runs, method=method)
        elif algorithm == 'hierarchical_weighted':
            result = run_weighted_hierarchical_consensus(path, dataset, seed=seed, m=effective_m, cnt_times=runs, method=method, sharpen=sharpen)
        elif algorithm == 'sdgca':
            result = run_sdgca(path, dataset, seed=seed, m=effective_m, cnt_times=runs, method=method)
        elif algorithm == 'sdgca_modified':
            params = resolve_params(dataset, None, None, None, None)
            result = run_sdgca_modified(path, dataset, seed=seed, m=effective_m, cnt_times=runs, nwca_para=params['lambda_'], eta=params['eta'], theta=params['theta'], method=method, diffusion_time=params['diffusion_time'])
        else:
            raise ValueError(f'Unknown algorithm: {algorithm}')
        row.update({'nmi_mean': f"{result['nmi_mean']:.6f}", 'nmi_std': f"{result['nmi_std']:.6f}", 'ari_mean': f"{result['ari_mean']:.6f}", 'ari_std': f"{result['ari_std']:.6f}", 'f_mean': f"{result['f_mean']:.6f}", 'f_std': f"{result['f_std']:.6f}"})
    except Exception as exc:
        row['status'] = 'error'
        row['error'] = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
    row['seconds'] = f'{time.time() - started:.2f}'
    return row

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--output', default=Path(__file__).resolve().parents[1] / 'results' / 'analysis_full_suite.tsv')
    parser.add_argument('--prefix', default='analysis_')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--sharpen', type=float, default=1.5)
    parser.add_argument('--algorithms', nargs='+', default=ALGORITHMS, choices=ALGORITHMS)
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    datasets = discover_datasets(root, args.prefix)
    completed = load_completed(output)
    if not datasets:
        raise SystemExit(f"No datasets found by pattern {root / (args.prefix + '*')}. Run generate_analysis_datasets.py first.")
    print(f'Datasets: {len(datasets)}')
    print(f'Output: {output}')
    for path in datasets:
        n_objects, n_members = dataset_shape(path)
        print(f'- {path.name}: objects={n_objects}, members={n_members}')
    for path in datasets:
        for method in args.methods:
            for algorithm in args.algorithms:
                effective_m = min(args.m, dataset_shape(path)[1])
                key = (path.stem, algorithm, method, str(effective_m), str(args.runs))
                if key in completed:
                    print(f'SKIP {path.stem:<36} {algorithm:<24} {method}')
                    continue
                row = run_one(path, algorithm, method, args.seed, args.m, args.runs, args.sharpen)
                append_tsv_row(output, row, HEADER)
                if row['status'] == 'ok':
                    completed.add(key)
                print(f"{path.stem:<36} {algorithm:<24} {method:<8} NMI={row['nmi_mean'] or 'ERR'} ARI={row['ari_mean'] or 'ERR'} F={row['f_mean'] or 'ERR'} {row['seconds']}s")
if __name__ == '__main__':
    main()
