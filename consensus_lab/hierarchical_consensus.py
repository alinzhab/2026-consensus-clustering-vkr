from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.io import loadmat
from scipy.spatial.distance import squareform
from metrics import compute_ari, compute_nmi, compute_pairwise_f_score
SUPPORTED_LINKAGE_METHODS: frozenset[str] = frozenset({'average', 'complete', 'single', 'ward'})
__all__ = ['SUPPORTED_LINKAGE_METHODS', 'build_coassociation_matrix', 'get_cls_result', 'load_dataset', 'load_dataset_full', 'run_hierarchical_consensus', 'validate_gt', 'validate_members', 'validate_method']

def load_dataset(dataset_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(dataset_path)
    suffix = dataset_path.suffix.lower()
    if suffix == '.mat':
        data = loadmat(dataset_path)
        members = np.asarray(data['members'], dtype=np.int64)
        gt = np.asarray(data['gt']).reshape(-1).astype(np.int64)
    elif suffix == '.npz':
        data = np.load(dataset_path)
        members = np.asarray(data['members'], dtype=np.int64)
        gt = np.asarray(data['gt']).reshape(-1).astype(np.int64)
    else:
        raise ValueError(f'Unsupported dataset format: {suffix}')
    if gt.size > 0 and gt.min() == 0:
        warnings.warn('gt labels start at 0; shifting to 1-based.', RuntimeWarning, stacklevel=2)
        gt = gt + 1
    if members.shape[0] != gt.shape[0]:
        raise ValueError('members row count must match gt length')
    return (members, gt)

def load_dataset_full(dataset_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dataset_path = Path(dataset_path)
    suffix = dataset_path.suffix.lower()
    if suffix == '.mat':
        data = loadmat(dataset_path)
        members = np.asarray(data['members'], dtype=np.int64)
        gt = np.asarray(data['gt']).reshape(-1).astype(np.int64)
        x: np.ndarray | None = None
        for key in ('X', 'x', 'data', 'fea', 'features'):
            if key in data:
                x = np.asarray(data[key], dtype=np.float64)
                break
    elif suffix == '.npz':
        data = np.load(dataset_path, allow_pickle=True)
        members = np.asarray(data['members'], dtype=np.int64)
        gt = np.asarray(data['gt']).reshape(-1).astype(np.int64)
        x = np.asarray(data['X'], dtype=np.float64) if 'X' in data.files else None
    else:
        raise ValueError(f'Unsupported dataset format: {suffix}')
    if gt.size > 0 and gt.min() == 0:
        warnings.warn('gt labels start at 0; shifting to 1-based.', RuntimeWarning, stacklevel=2)
        gt = gt + 1
    return (members, gt, x)

def validate_members(members: np.ndarray, m: int | None=None) -> np.ndarray:
    members = np.asarray(members)
    if members.ndim != 2:
        raise ValueError('members must be a 2D matrix')
    if members.shape[0] < 2:
        raise ValueError('dataset must contain at least two objects')
    if members.shape[1] < 1:
        raise ValueError('dataset must contain at least one base clustering')
    if not np.all(np.isfinite(members)):
        raise ValueError('members must contain only finite numeric values')
    if m is not None and (m < 1 or m > members.shape[1]):
        raise ValueError('m must be between 1 and the number of base clusterings')
    members = members.astype(np.int64)
    # Normalize every column to consecutive 1-based labels. build_coassociation_matrix
    # and SDGCA index by `labels - 1`, so 0-based or sparse label sets (e.g. {0,2,5})
    # silently produced wrong consensus matrices.
    for j in range(members.shape[1]):
        _, inverse = np.unique(members[:, j], return_inverse=True)
        members[:, j] = inverse.astype(np.int64) + 1
    return members

def validate_gt(gt: np.ndarray, n_objects: int | None=None) -> np.ndarray:
    gt = np.asarray(gt).reshape(-1)
    if gt.size < 2:
        raise ValueError('gt must contain at least two labels')
    if n_objects is not None and gt.size != n_objects:
        raise ValueError('gt length must match the number of objects')
    if not np.all(np.isfinite(gt)):
        raise ValueError('gt must contain only finite numeric values')
    gt = gt.astype(np.int64)
    if np.unique(gt).size < 2:
        raise ValueError('gt must contain at least two classes')
    return gt

def validate_method(method: str) -> str:
    if method not in SUPPORTED_LINKAGE_METHODS:
        allowed = ', '.join(sorted(SUPPORTED_LINKAGE_METHODS))
        raise ValueError(f'method must be one of: {allowed}')
    return method

def build_coassociation_matrix(base_cls: np.ndarray) -> np.ndarray:
    n, m = base_cls.shape
    consensus = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        labels = base_cls[:, j]
        k = int(labels.max())
        indicator = np.zeros((n, k), dtype=np.float64)
        indicator[np.arange(n), labels - 1] = 1.0
        consensus += indicator @ indicator.T
    consensus /= m
    consensus = (consensus + consensus.T) / 2.0
    consensus = np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus

def get_cls_result(consensus_matrix: np.ndarray, cls_num: int, method: str='average') -> np.ndarray:
    method = validate_method(method)
    if method == 'ward':
        warnings.warn('Ward linkage is designed for Euclidean feature vectors; here it is applied to 1 - co-association distances.', RuntimeWarning, stacklevel=2)
    consensus_matrix = np.clip(consensus_matrix, 0.0, 1.0)
    consensus_matrix = np.maximum(consensus_matrix, consensus_matrix.T)
    matrix = consensus_matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    similarity = squareform(matrix, checks=False)
    distance = 1.0 - similarity
    tree = linkage(distance, method=method)
    return fcluster(tree, t=cls_num, criterion='maxclust').astype(np.int64)

def run_hierarchical_consensus(dataset_path: str | Path, data_name: str | None=None, seed: int=19, m: int=40, cnt_times: int=20, method: str='average', selection_strategy: str='random', qd_alpha: float=0.5) -> dict:
    from consensus_runner import run_consensus_loop

    def _build(base_cls, _gt, _m):
        return build_coassociation_matrix(base_cls)
    return run_consensus_loop(dataset_path, _build, data_name=data_name, seed=seed, m=m, cnt_times=cnt_times, method=method, selection_strategy=selection_strategy, qd_alpha=qd_alpha, clamp_m_name='hierarchical_baseline')

def main() -> None:
    parser = argparse.ArgumentParser(description='Базовая иерархическая консенсус-кластеризация.')
    parser.add_argument('--dataset', default='Ecoli')
    parser.add_argument('--root', default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--method', default='average', choices=sorted(SUPPORTED_LINKAGE_METHODS))
    args = parser.parse_args()
    dataset_path = Path(args.root) / f'{args.dataset}.mat'
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f'{args.dataset}.npz'
    result = run_hierarchical_consensus(dataset_path=dataset_path, data_name=args.dataset, seed=args.seed, m=args.m, cnt_times=args.runs, method=args.method)
    print('           mean    variance')
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")
if __name__ == '__main__':
    main()
