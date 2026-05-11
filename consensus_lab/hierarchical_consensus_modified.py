from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from hierarchical_consensus import SUPPORTED_LINKAGE_METHODS, validate_members
from ensemble_selection import partition_agreement
__all__ = ['build_partition_matrix', 'build_weighted_consensus_matrix', 'compute_base_clustering_weights', 'partition_agreement', 'run_weighted_hierarchical_consensus']

def build_partition_matrix(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    return (labels[:, None] == labels[None, :]).astype(np.float64)

def compute_base_clustering_weights(base_cls: np.ndarray) -> np.ndarray:
    base_cls = validate_members(base_cls)
    m = base_cls.shape[1]
    agreement = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            score = partition_agreement(base_cls[:, i], base_cls[:, j])
            agreement[i, j] = score
            agreement[j, i] = score
    weights = np.mean(agreement, axis=1)
    weights_sum = float(np.sum(weights))
    if weights_sum == 0.0:
        return np.full(m, 1.0 / m, dtype=np.float64)
    return weights / weights_sum

def build_weighted_consensus_matrix(base_cls: np.ndarray, sharpen: float=1.0) -> Tuple[np.ndarray, np.ndarray]:
    base_cls = validate_members(base_cls)
    if sharpen <= 0:
        raise ValueError('sharpen must be positive')
    n, m = base_cls.shape
    weights = compute_base_clustering_weights(base_cls)
    if sharpen != 1.0:
        weights = weights ** sharpen
        weights_sum = float(np.sum(weights))
        weights = np.full(m, 1.0 / m, dtype=np.float64) if weights_sum == 0.0 else weights / weights_sum
    consensus = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        consensus += weights[j] * build_partition_matrix(base_cls[:, j])
    consensus = (consensus + consensus.T) / 2.0
    consensus = np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(consensus, 1.0)
    return (consensus, weights)

def run_weighted_hierarchical_consensus(dataset_path: str | Path, data_name: str | None=None, seed: int=19, m: int=40, cnt_times: int=20, method: str='average', sharpen: float=1.0, selection_strategy: str='random', qd_alpha: float=0.5) -> dict:
    from consensus_runner import run_consensus_loop
    if sharpen <= 0:
        raise ValueError('sharpen must be positive')
    weight_bank: list[np.ndarray] = []

    def _build(base_cls, _gt, _m):
        consensus, weights = build_weighted_consensus_matrix(base_cls, sharpen=sharpen)
        weight_bank.append(weights)
        return consensus
    result = run_consensus_loop(dataset_path, _build, data_name=data_name, seed=seed, m=m, cnt_times=cnt_times, method=method, selection_strategy=selection_strategy, qd_alpha=qd_alpha)
    result['sharpen'] = float(sharpen)
    result['avg_weights'] = np.mean(np.vstack(weight_bank), axis=0).astype(float).tolist()
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='Взвешенная иерархическая консенсус-кластеризация.')
    parser.add_argument('--dataset', default='Ecoli')
    parser.add_argument('--root', default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--method', default='average', choices=sorted(SUPPORTED_LINKAGE_METHODS))
    parser.add_argument('--sharpen', type=float, default=1.0)
    args = parser.parse_args()
    dataset_path = Path(args.root) / f'{args.dataset}.mat'
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f'{args.dataset}.npz'
    result = run_weighted_hierarchical_consensus(dataset_path=dataset_path, data_name=args.dataset, seed=args.seed, m=args.m, cnt_times=args.runs, method=args.method, sharpen=args.sharpen)
    print('           mean    variance')
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")
if __name__ == '__main__':
    main()
