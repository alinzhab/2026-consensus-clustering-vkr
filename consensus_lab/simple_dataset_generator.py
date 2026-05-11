import json
from pathlib import Path
import numpy as np
from base_clusterings import build_base_clusterings

def _sample_cluster_sizes(n_samples, n_clusters, imbalance_ratio, rng):
    if imbalance_ratio <= 1:
        weights = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    else:
        weights = rng.lognormal(mean=0.0, sigma=np.log(imbalance_ratio) / 2.0, size=n_clusters)
        weights = weights / weights.sum()
    counts = np.floor(weights * n_samples).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() < n_samples:
        counts[np.argmax(weights - counts / max(n_samples, 1))] += 1
    while counts.sum() > n_samples:
        idx = int(np.argmax(counts))
        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break
    return counts

def normalize_features(x):
    mins = np.min(x, axis=0, keepdims=True)
    spans = np.max(x, axis=0, keepdims=True) - mins
    spans[spans == 0] = 1.0
    return (x - mins) / spans

def generate_simple_gaussian_dataset(name, n_samples, n_clusters, dim, cluster_std, separation, imbalance_ratio, seed, base_clusterings=30, base_k_min=2, base_k_max=8, base_strategy='mixed'):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * separation
    counts = _sample_cluster_sizes(n_samples, n_clusters, imbalance_ratio, rng)
    x_parts = []
    gt_parts = []
    for cluster_idx, count in enumerate(counts):
        points = centers[cluster_idx] + rng.normal(scale=cluster_std, size=(count, dim))
        x_parts.append(points)
        gt_parts.append(np.full(count, cluster_idx + 1, dtype=np.int64))
    x = normalize_features(np.vstack(x_parts))
    gt = np.concatenate(gt_parts)
    permutation = rng.permutation(x.shape[0])
    x = x[permutation]
    gt = gt[permutation]
    members, base_info = build_base_clusterings(x, n_clusterings=base_clusterings, k_min=base_k_min, k_max=base_k_max, rng=rng, strategy=base_strategy, return_info=True)
    meta = {'name': name, 'generator': 'simple_gaussian', 'n_samples': n_samples, 'n_clusters': n_clusters, 'dim': dim, 'cluster_std': cluster_std, 'separation': separation, 'imbalance_ratio': imbalance_ratio, 'seed': seed, 'base_clusterings': base_clusterings, 'base_k_min': base_k_min, 'base_k_max': base_k_max, 'base_strategy': base_strategy, 'base_info': base_info}
    return (x, gt, members, meta)

def save_dataset(path, x, gt, members, meta):
    path = Path(path)
    np.savez(path, X=x, gt=gt, members=members, meta=json.dumps(meta, ensure_ascii=False))
