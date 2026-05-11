import csv
from pathlib import Path
import numpy as np
from base_clusterings import build_base_clusterings
from hierarchical_consensus import build_coassociation_matrix, get_cls_result
from metrics import compute_ari, compute_nmi
PREPROCESSING_MODES = ('none', 'minmax', 'zscore')
RESULTS_PATH = Path(__file__).resolve().parent.parent / 'results' / 'preprocessing_comparison.tsv'

def _make_gaussian_dataset(name, centers, stds, counts, rng, outliers=0, extra_scale=None):
    x_parts = []
    gt_parts = []
    for idx, (center, std, count) in enumerate(zip(centers, stds, counts), start=1):
        points = rng.normal(loc=center, scale=std, size=(count, len(center)))
        x_parts.append(points)
        gt_parts.append(np.full(count, idx, dtype=np.int64))
    x = np.vstack(x_parts)
    gt = np.concatenate(gt_parts)
    if outliers > 0:
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        span = maxs - mins
        noise = rng.uniform(mins - 3.0 * span, maxs + 3.0 * span, size=(outliers, x.shape[1]))
        nearest = np.argmin(np.linalg.norm(noise[:, None, :] - x[None, :, :], axis=2), axis=1)
        x = np.vstack([x, noise])
        gt = np.concatenate([gt, gt[nearest]])
    if extra_scale is not None:
        x = x * np.asarray(extra_scale, dtype=np.float64)
    order = rng.permutation(x.shape[0])
    return {'name': name, 'X': x[order], 'gt': gt[order], 'n_clusters': len(counts)}

def make_datasets(seed=2026):
    rng = np.random.default_rng(seed)
    return [_make_gaussian_dataset(name='balanced_same_scale', centers=np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]]), stds=np.array([[0.7, 0.7], [0.7, 0.7], [0.7, 0.7]]), counts=[180, 180, 180], rng=rng), _make_gaussian_dataset(name='feature_scale_imbalance', centers=np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]]), stds=np.array([[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]]), counts=[180, 180, 180], rng=rng, extra_scale=[1.0, 150.0]), _make_gaussian_dataset(name='with_outliers', centers=np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]]), stds=np.array([[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]]), counts=[170, 170, 170], rng=rng, outliers=30), _make_gaussian_dataset(name='anisotropic_imbalanced', centers=np.array([[-5.0, -1.0], [0.0, 3.5], [5.0, -1.0]]), stds=np.array([[1.8, 0.25], [0.35, 1.5], [1.1, 0.45]]), counts=[260, 160, 90], rng=rng)]

def evaluate_members(members, gt):
    base_nmi = [compute_nmi(members[:, idx], gt) for idx in range(members.shape[1])]
    base_ari = [compute_ari(members[:, idx], gt) for idx in range(members.shape[1])]
    consensus = build_coassociation_matrix(members)
    labels = get_cls_result(consensus, cls_num=np.unique(gt).size, method='average')
    return {'base_nmi_mean': float(np.mean(base_nmi)), 'base_ari_mean': float(np.mean(base_ari)), 'consensus_nmi': float(compute_nmi(labels, gt)), 'consensus_ari': float(compute_ari(labels, gt))}

def run_comparison():
    rows = []
    for dataset in make_datasets():
        for mode in PREPROCESSING_MODES:
            members = build_base_clusterings(dataset['X'], n_clusterings=24, k_min=max(2, dataset['n_clusters'] - 1), k_max=dataset['n_clusters'] + 1, rng=19, strategy='mixed', preprocessing=mode, feature_subsample=True, noise_scale=0.01)
            metrics = evaluate_members(members, dataset['gt'])
            rows.append({'dataset': dataset['name'], 'preprocessing': mode, **metrics})
    return rows

def write_rows(rows, path=RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        fieldnames = ['dataset', 'preprocessing', 'base_nmi_mean', 'base_ari_mean', 'consensus_nmi', 'consensus_ari']
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

def print_summary(rows):
    print('dataset\tpreprocessing\tbase_nmi\tbase_ari\tconsensus_nmi\tconsensus_ari')
    for row in rows:
        print(f"{row['dataset']}\t{row['preprocessing']}\t{row['base_nmi_mean']:.4f}\t{row['base_ari_mean']:.4f}\t{row['consensus_nmi']:.4f}\t{row['consensus_ari']:.4f}")
    print('\nMean by preprocessing:')
    for mode in PREPROCESSING_MODES:
        selected = [row for row in rows if row['preprocessing'] == mode]
        mean_consensus_nmi = np.mean([row['consensus_nmi'] for row in selected])
        mean_consensus_ari = np.mean([row['consensus_ari'] for row in selected])
        mean_base_nmi = np.mean([row['base_nmi_mean'] for row in selected])
        print(f'{mode}: base_nmi={mean_base_nmi:.4f}, consensus_nmi={mean_consensus_nmi:.4f}, consensus_ari={mean_consensus_ari:.4f}')
if __name__ == '__main__':
    rows = run_comparison()
    write_rows(rows)
    print_summary(rows)
    print(f'\nSaved: {RESULTS_PATH}')
