from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / 'consensus_lab'
sys.path.insert(0, str(LAB_DIR))
from densired_style_generator import generate_densired_style_dataset, save_dataset as save_densired_dataset
from repliclust_style_generator import generate_archetype_dataset, save_dataset as save_repliclust_dataset
from simple_dataset_generator import generate_simple_gaussian_dataset, save_dataset as save_simple_dataset
DATASET_TYPES = ['compact', 'overlapping', 'imbalanced', 'high_dimensional', 'elongated', 'density_varied', 'mixed_complex']
MANIFEST_COLUMNS = ['dataset_id', 'path', 'generator', 'dataset_type', 'difficulty_level', 'n_samples', 'n_clusters', 'dim', 'seed', 'split', 'fold_id', 'size_profile', 'allow_sdgca', 'base_clusterings', 'base_k_min', 'base_k_max', 'params_json']

def _size_profile(index: int, rng: np.random.Generator, smoke: bool) -> tuple[str, int]:
    if smoke:
        return ('small', int(rng.integers(300, 551)))
    bucket = index % 10
    if bucket < 5:
        return ('small', int(rng.integers(300, 601)))
    if bucket < 8:
        return ('medium', int(rng.integers(800, 1501)))
    return ('large', int(rng.integers(2000, 5001)))

def _difficulty(index: int, smoke: bool) -> str:
    if smoke:
        return ['easy', 'medium', 'hard'][index % 3]
    return ['easy', 'medium', 'hard', 'stress'][index % 4]

def _common_base_params(n_clusters: int, size_profile: str, smoke: bool) -> dict[str, Any]:
    base_clusterings = 24 if smoke else 30 if size_profile != 'large' else 18
    return {'base_clusterings': base_clusterings, 'base_k_min': max(2, n_clusters - 2), 'base_k_max': n_clusters + 2, 'base_strategy': 'mixed' if size_profile != 'large' else 'kmeans'}

def _make_config(dataset_type: str, dataset_id: str, index: int, seed: int, rng: np.random.Generator, smoke: bool) -> tuple[str, dict[str, Any], dict[str, Any]]:
    size_profile, n_samples = _size_profile(index, rng, smoke)
    difficulty = _difficulty(index, smoke)
    n_clusters = int(rng.integers(3, 8 if smoke else 11))
    dim = 2
    generator = 'simple'
    params: dict[str, Any]
    if dataset_type == 'compact':
        dim = int(rng.choice([2, 3, 5]))
        params = {'name': dataset_id, 'n_samples': n_samples, 'n_clusters': n_clusters, 'dim': dim, 'cluster_std': float(rng.uniform(0.18, 0.35)), 'separation': float(rng.uniform(5.5, 8.0)), 'imbalance_ratio': float(rng.uniform(1.0, 1.4)), 'seed': seed, **_common_base_params(n_clusters, size_profile, smoke)}
    elif dataset_type == 'overlapping':
        dim = int(rng.choice([2, 3, 5, 10]))
        params = {'name': dataset_id, 'n_samples': n_samples, 'n_clusters': n_clusters, 'dim': dim, 'cluster_std': float(rng.uniform(0.65, 1.15)), 'separation': float(rng.uniform(1.8, 3.0)), 'imbalance_ratio': float(rng.uniform(1.0, 1.8)), 'seed': seed, **_common_base_params(n_clusters, size_profile, smoke)}
    elif dataset_type == 'imbalanced':
        dim = int(rng.choice([2, 5, 10]))
        params = {'name': dataset_id, 'n_samples': n_samples, 'n_clusters': n_clusters, 'dim': dim, 'cluster_std': float(rng.uniform(0.35, 0.75)), 'separation': float(rng.uniform(3.2, 5.5)), 'imbalance_ratio': float(rng.uniform(3.0, 8.0)), 'seed': seed, **_common_base_params(n_clusters, size_profile, smoke)}
    elif dataset_type == 'high_dimensional':
        dim = int(rng.choice([10, 20, 50, 100]))
        params = {'name': dataset_id, 'n_samples': n_samples, 'n_clusters': n_clusters, 'dim': dim, 'cluster_std': float(rng.uniform(0.45, 0.95)), 'separation': float(rng.uniform(2.4, 4.8)), 'imbalance_ratio': float(rng.uniform(1.0, 2.5)), 'seed': seed, **_common_base_params(n_clusters, size_profile, smoke)}
    elif dataset_type == 'elongated':
        generator = 'repliclust'
        dim = int(rng.choice([2, 3, 5, 10]))
        params = {'name': dataset_id, 'n_clusters': n_clusters, 'dim': dim, 'n_samples': n_samples, 'aspect_ref': float(rng.uniform(4.5, 7.0)), 'aspect_maxmin': float(rng.uniform(4.0, 8.0)), 'radius_ref': 1.0, 'radius_maxmin': float(rng.uniform(1.5, 3.0)), 'min_overlap': float(rng.uniform(0.02, 0.08)), 'max_overlap': float(rng.uniform(0.12, 0.28)), 'imbalance_ratio': float(rng.uniform(1.5, 3.5)), 'distributions': ['normal', 'student_t', 'lognormal'], 'distribution_proportions': [0.45, 0.35, 0.2], 'seed': seed, **_common_base_params(n_clusters, size_profile, smoke)}
    elif dataset_type == 'density_varied':
        generator = 'densired'
        dim = int(rng.choice([2, 3, 5]))
        factors = rng.uniform(0.35, 2.1, size=n_clusters).round(3).tolist()
        params = {'name': dataset_id, 'dim': dim, 'clunum': n_clusters, 'core_num': int(rng.integers(max(n_clusters * 8, 30), max(n_clusters * 22, 70))), 'data_num': n_samples, 'seed': seed, 'domain_size': 20.0, 'radius': float(rng.uniform(0.025, 0.055)), 'step': float(rng.uniform(0.04, 0.075)), 'noise_ratio': float(rng.uniform(0.04, 0.14)), 'density_factors': factors, 'momentum': float(rng.uniform(0.0, 0.45)), 'branch': float(rng.uniform(0.0, 0.08)), 'star': float(rng.uniform(0.0, 0.2)), 'distribution': str(rng.choice(['uniform', 'gaussian'])), **_common_base_params(n_clusters, size_profile, smoke)}
    elif dataset_type == 'mixed_complex':
        generator = 'repliclust'
        dim = int(rng.choice([10, 20, 50]))
        params = {'name': dataset_id, 'n_clusters': n_clusters, 'dim': dim, 'n_samples': n_samples, 'aspect_ref': float(rng.uniform(4.5, 7.0)), 'aspect_maxmin': float(rng.uniform(4.0, 8.0)), 'radius_ref': 1.0, 'radius_maxmin': float(rng.uniform(2.0, 4.0)), 'min_overlap': float(rng.uniform(0.04, 0.12)), 'max_overlap': float(rng.uniform(0.18, 0.35)), 'imbalance_ratio': float(rng.uniform(3.0, 7.0)), 'distributions': ['normal', 'student_t', 'exponential', 'lognormal'], 'distribution_proportions': [0.3, 0.3, 0.2, 0.2], 'seed': seed, **_common_base_params(n_clusters, size_profile, smoke)}
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
    metadata = {'generator': generator, 'dataset_type': dataset_type, 'difficulty_level': difficulty, 'size_profile': size_profile, 'n_samples': n_samples, 'n_clusters': n_clusters, 'dim': dim}
    return (generator, params, metadata)

def _assign_splits(rows: list[dict[str, Any]], seed: int, folds: int) -> None:
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_type.setdefault(row['dataset_type'], []).append(idx)
    for indices in by_type.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n >= 3:
            train_count = max(1, int(np.floor(0.6 * n)))
            validation_count = max(1, int(np.floor(0.2 * n)))
            if train_count + validation_count >= n:
                train_count = max(1, n - 2)
                validation_count = 1
        elif n == 2:
            train_count = 1
            validation_count = 1
        else:
            train_count = 1
            validation_count = 0
        train_end = train_count
        val_end = train_count + validation_count
        for order, row_idx in enumerate(shuffled):
            if order < train_end:
                split = 'train'
            elif order < val_end:
                split = 'validation'
            else:
                split = 'test'
            rows[row_idx]['split'] = split
            rows[row_idx]['fold_id'] = int(order % folds)

def generate_suite(count: int, output_dir: Path, seed: int, folds: int, force: bool) -> list[dict[str, Any]]:
    smoke = count <= 50
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        dataset_type = DATASET_TYPES[index % len(DATASET_TYPES)]
        per_type_index = index // len(DATASET_TYPES)
        dataset_seed = int(seed + 1009 * index + 17)
        dataset_id = f'massive_{dataset_type}_{per_type_index:03d}'
        path = output_dir / f'{dataset_id}.npz'
        generator, params, meta = _make_config(dataset_type, dataset_id, index, dataset_seed, rng, smoke)
        if force or not path.exists():
            if generator == 'simple':
                x, gt, members, saved_meta = generate_simple_gaussian_dataset(**params)
                save_simple_dataset(path, x, gt, members, {**saved_meta, **meta})
            elif generator == 'densired':
                x, gt, members, saved_meta = generate_densired_style_dataset(**params)
                save_densired_dataset(path, x, gt, members, {**saved_meta, **meta})
            elif generator == 'repliclust':
                x, gt, members, saved_meta = generate_archetype_dataset(**params)
                save_repliclust_dataset(path, x, gt, members, {**saved_meta, **meta})
            else:
                raise ValueError(generator)
        allow_sdgca = meta['size_profile'] == 'small' and int(meta['n_samples']) <= 650
        rows.append({'dataset_id': dataset_id, 'path': str(path), 'generator': generator, 'dataset_type': dataset_type, 'difficulty_level': meta['difficulty_level'], 'n_samples': int(meta['n_samples']), 'n_clusters': int(meta['n_clusters']), 'dim': int(meta['dim']), 'seed': dataset_seed, 'split': '', 'fold_id': '', 'size_profile': meta['size_profile'], 'allow_sdgca': str(bool(allow_sdgca)).lower(), 'base_clusterings': int(params['base_clusterings']), 'base_k_min': int(params['base_k_min']), 'base_k_max': int(params['base_k_max']), 'params_json': json.dumps(params, ensure_ascii=False, sort_keys=True)})
    _assign_splits(rows, seed=seed + 404, folds=folds)
    manifest_path = output_dir / 'manifest.tsv'
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=21)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'datasets' / 'massive_synthetic')
    parser.add_argument('--seed', type=int, default=20260508)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.count < len(DATASET_TYPES):
        raise ValueError(f'count must be at least {len(DATASET_TYPES)} to cover all dataset types')
    rows = generate_suite(args.count, args.output_dir, args.seed, args.folds, args.force)
    by_split: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        by_split[row['split']] = by_split.get(row['split'], 0) + 1
        by_type[row['dataset_type']] = by_type.get(row['dataset_type'], 0) + 1
    print(f'Generated/registered {len(rows)} datasets')
    print(f"Manifest: {args.output_dir / 'manifest.tsv'}")
    print(f'By split: {by_split}')
    print(f'By type: {by_type}')
if __name__ == '__main__':
    main()
