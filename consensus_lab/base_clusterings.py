from __future__ import annotations
from typing import Any, Sequence, Tuple
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
__all__ = ['build_base_clusterings']
_RngLike = 'np.random.Generator | int | None'

def _as_rng(rng_or_seed: Any) -> np.random.Generator:
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    return np.random.default_rng(rng_or_seed)

def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64) + 1

def _standardize_features(x: np.ndarray) -> Tuple[np.ndarray, dict[str, Any]]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    constant_features = (std == 0).reshape(-1)
    std[std == 0] = 1.0
    return ((x - mean) / std, {'method': 'zscore', 'constant_feature_count': int(np.sum(constant_features))})

def _minmax_scale_features(x: np.ndarray) -> Tuple[np.ndarray, dict[str, Any]]:
    mins = np.min(x, axis=0, keepdims=True)
    spans = np.max(x, axis=0, keepdims=True) - mins
    constant_features = (spans == 0).reshape(-1)
    spans[spans == 0] = 1.0
    return ((x - mins) / spans, {'method': 'minmax', 'constant_feature_count': int(np.sum(constant_features))})

def _preprocess_features(x: np.ndarray, preprocessing: str) -> Tuple[np.ndarray, dict[str, Any]]:
    if preprocessing == 'zscore':
        return _standardize_features(x)
    if preprocessing == 'minmax':
        return _minmax_scale_features(x)
    if preprocessing == 'none':
        return (x.copy(), {'method': 'none', 'constant_feature_count': int(np.sum(np.std(x, axis=0) == 0))})
    raise ValueError('preprocessing must be one of: zscore, minmax, none')

def _choose_feature_subset(x: np.ndarray, rng: np.random.Generator, min_fraction: float=0.5) -> Tuple[np.ndarray, list[int]]:
    n_features = x.shape[1]
    if n_features <= 2:
        return (x, list(range(n_features)))
    min_features = max(2, int(np.ceil(n_features * min_fraction)))
    feature_count = int(rng.integers(min_features, n_features + 1))
    feature_idx = np.sort(rng.choice(n_features, size=feature_count, replace=False))
    return (x[:, feature_idx], feature_idx.astype(int).tolist())

def _kmeans_labels(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    _, labels = kmeans2(x, k, minit='points', iter=50, seed=rng)
    return _normalize_labels(labels)

def _hierarchical_labels(x: np.ndarray, k: int, method: str) -> np.ndarray:
    tree = linkage(x, method=method, metric='euclidean')
    return fcluster(tree, t=k, criterion='maxclust').astype(np.int64)

def build_base_clusterings(x: np.ndarray, n_clusterings: int=30, k_min: int=2, k_max: int=8, rng: Any=None, strategy: str='mixed', feature_subsample: bool=True, noise_scale: float=0.01, max_hierarchical_objects: int=2500, standardize: bool=True, preprocessing: str='zscore', return_info: bool=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('X must be a 2D feature matrix')
    if x.shape[0] < 2:
        raise ValueError('X must contain at least two objects')
    if x.shape[1] < 1:
        raise ValueError('X must contain at least one feature')
    if not np.all(np.isfinite(x)):
        raise ValueError('X must contain only finite numeric values')
    if n_clusterings < 1:
        raise ValueError('n_clusterings must be positive')
    if strategy not in {'mixed', 'kmeans', 'hierarchical'}:
        raise ValueError('strategy must be one of: mixed, kmeans, hierarchical')
    if standardize is False:
        preprocessing = 'none'
    rng = _as_rng(rng)
    k_min = max(2, int(k_min))
    k_max = max(k_min, int(k_max))
    k_max = min(k_max, x.shape[0])
    x, preprocessing_info = _preprocess_features(x, preprocessing)
    preprocessing_info['standardize'] = preprocessing == 'zscore'
    members = np.zeros((x.shape[0], int(n_clusterings)), dtype=np.int64)
    hierarchical_methods: Sequence[str] = ('average', 'complete', 'single', 'ward')
    clusterings_info: list[dict[str, Any]] = []
    for j in range(int(n_clusterings)):
        if feature_subsample:
            x_work, feature_idx = _choose_feature_subset(x, rng)
        else:
            x_work = x.copy()
            feature_idx = list(range(x.shape[1]))
        if noise_scale > 0:
            x_work = x_work + rng.normal(scale=noise_scale * (1 + j % 3), size=x_work.shape)
        k = int(rng.integers(k_min, k_max + 1))
        use_hierarchical = strategy == 'mixed' and j % 2 == 1 and (x.shape[0] <= max_hierarchical_objects)
        if strategy == 'hierarchical' and x.shape[0] <= max_hierarchical_objects:
            use_hierarchical = True
        if use_hierarchical:
            method = hierarchical_methods[j % len(hierarchical_methods)]
            members[:, j] = _hierarchical_labels(x_work, k, method)
            clustering_info: dict[str, Any] = {'index': j, 'algorithm': 'hierarchical', 'linkage': method, 'k': k}
        else:
            members[:, j] = _kmeans_labels(x_work, k, rng)
            clustering_info = {'index': j, 'algorithm': 'kmeans', 'k': k}
            if strategy == 'hierarchical' and x.shape[0] > max_hierarchical_objects:
                clustering_info['fallback_reason'] = 'too_many_objects_for_hierarchical'
        clustering_info.update({'feature_count': len(feature_idx), 'feature_indices': feature_idx, 'noise_scale': float(noise_scale * (1 + j % 3)) if noise_scale > 0 else 0.0})
        clusterings_info.append(clustering_info)
    if return_info:
        return (members, {'strategy': strategy, 'n_clusterings': int(n_clusterings), 'k_min': int(k_min), 'k_max': int(k_max), 'feature_subsample': bool(feature_subsample), 'preprocessing': preprocessing_info, 'max_hierarchical_objects': int(max_hierarchical_objects), 'clusterings': clusterings_info})
    return members
