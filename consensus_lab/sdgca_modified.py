from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.linalg import eigh, expm
from hierarchical_consensus import SUPPORTED_LINKAGE_METHODS, load_dataset_full
from sdgca import _clamp_m, compute_norm_k, compute_nwca, compute_s, compute_w, get_all_segs, optimize_sdgca, simxjac
__all__ = ['DEFAULT_PARAMS', 'build_fuzzy_membership_matrix', 'compute_adaptive_diffusion_time', 'compute_d_diffusion', 'compute_fuzzy_entropy_weights', 'compute_modified_neci', 'compute_partition_agreements', 'graph_diffusion_of_cluster', 'resolve_params', 'run_sdgca_modified']
DEFAULT_PARAMS: dict[str, dict[str, float]] = {'Ecoli': {'lambda_': 0.09, 'eta': 0.65, 'theta': 0.75, 'diffusion_time': 1.0}, 'GLIOMA': {'lambda_': 0.02, 'eta': 0.8, 'theta': 0.6, 'diffusion_time': 1.0}, 'Aggregation': {'lambda_': 0.08, 'eta': 0.65, 'theta': 0.7, 'diffusion_time': 0.8}, 'Lung': {'lambda_': 0.75, 'eta': 0.75, 'theta': 0.6, 'diffusion_time': 1.2}, 'BBC': {'lambda_': 0.06, 'eta': 1.01, 'theta': 1.01, 'diffusion_time': 1.1}, 'orlraws10P': {'lambda_': 0.95, 'eta': 1.01, 'theta': 1.01, 'diffusion_time': 1.0}, 'densired_compact_hard': {'lambda_': 0.08, 'eta': 0.72, 'theta': 0.78, 'diffusion_time': 0.9}, 'densired_stretched_hard': {'lambda_': 0.06, 'eta': 0.82, 'theta': 0.88, 'diffusion_time': 1.3}, 'densired_mix_hard': {'lambda_': 0.07, 'eta': 0.75, 'theta': 0.82, 'diffusion_time': 1.1}}
_GENERIC_DEFAULTS: dict[str, float] = {'lambda_': 0.09, 'eta': 0.75, 'theta': 0.65, 'diffusion_time': 1.0}

def resolve_params(dataset_name: str, lambda_override: float | None, eta_override: float | None, theta_override: float | None, diffusion_override: float | None=None) -> dict[str, float]:
    params = DEFAULT_PARAMS.get(dataset_name, _GENERIC_DEFAULTS).copy()
    if lambda_override is not None:
        params['lambda_'] = lambda_override
    if eta_override is not None:
        params['eta'] = eta_override
    if theta_override is not None:
        params['theta'] = theta_override
    if diffusion_override is not None:
        params['diffusion_time'] = diffusion_override
    return params

def build_fuzzy_membership_matrix(x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    centroids = np.asarray([np.mean(x[labels == label], axis=0) for label in unique_labels], dtype=np.float64)
    x_sq = np.sum(x ** 2, axis=1, keepdims=True)
    c_sq = np.sum(centroids ** 2, axis=1)
    cross = x @ centroids.T
    dist_sq = np.maximum(x_sq - 2.0 * cross + c_sq, 0.0)
    distances = np.maximum(np.sqrt(dist_sq), 1e-12)
    inverse_dist = 1.0 / distances
    memberships = inverse_dist / np.sum(inverse_dist, axis=1, keepdims=True)
    hard_positions = np.searchsorted(unique_labels, labels)
    memberships[np.arange(x.shape[0]), hard_positions] += 1e-06
    memberships = memberships / np.sum(memberships, axis=1, keepdims=True)
    return (memberships, unique_labels)

def compute_fuzzy_entropy_weights(base_cls: np.ndarray, x: np.ndarray | None, para_theta: float) -> np.ndarray:
    n_cls_orig = base_cls.max(axis=0).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(n_cls_orig)[:-1]))
    total_clusters = int(np.sum(n_cls_orig))
    fuzzy_weights = np.ones(total_clusters, dtype=np.float64)
    if x is None:
        return fuzzy_weights
    m = base_cls.shape[1]
    for j in range(m):
        labels = base_cls[:, j]
        memberships, unique_labels = build_fuzzy_membership_matrix(x, labels)
        denom = np.log2(max(unique_labels.size, 2))
        for label in unique_labels:
            mask = labels == label
            cluster_memberships = memberships[mask]
            entropy = -np.sum(cluster_memberships * np.log2(np.maximum(cluster_memberships, 1e-12)), axis=1)
            entropy = float(np.mean(entropy)) / denom
            global_idx = offsets[j] + int(label) - 1
            fuzzy_weights[global_idx] = float(np.exp(-entropy / max(para_theta, 1e-12)))
    return fuzzy_weights

def compute_partition_agreements(base_cls: np.ndarray) -> np.ndarray:
    """Pairwise NMI (arithmetic-normalised) averaged per partition.

    The previous implementation called sklearn's normalized_mutual_info_score
    O(m^2) times — every call re-built a contingency table from raw labels
    and re-computed entropies. For m=40 this dominated SDGCA-modified runtime.
    Here we cache (label inverse, marginal entropy) once per partition and
    compute every pair's mutual information from a single np.add.at call.
    """
    n, m = base_cls.shape
    if m == 1:
        return np.ones(1, dtype=np.float64)
    inverses: list[np.ndarray] = []
    sizes: list[int] = []
    entropies: list[float] = []
    for j in range(m):
        _, inv = np.unique(base_cls[:, j], return_inverse=True)
        k = int(inv.max()) + 1
        counts = np.bincount(inv, minlength=k).astype(np.float64)
        probs = counts[counts > 0] / n
        h = float(-(probs * np.log(probs)).sum())
        inverses.append(inv.astype(np.int64))
        sizes.append(k)
        entropies.append(h)
    pairwise = np.zeros((m, m), dtype=np.float64)
    for j in range(m):
        inv_j = inverses[j]
        kj = sizes[j]
        hj = entropies[j]
        for k in range(j + 1, m):
            inv_k = inverses[k]
            kk = sizes[k]
            hk = entropies[k]
            denom = (hj + hk) / 2.0
            if denom <= 0.0:
                v = 0.0
            else:
                cont = np.zeros((kj, kk), dtype=np.float64)
                np.add.at(cont, (inv_j, inv_k), 1)
                cont /= n
                row = cont.sum(axis=1)
                col = cont.sum(axis=0)
                nz = cont > 0
                if not nz.any():
                    v = 0.0
                else:
                    outer = np.outer(row, col)
                    mi = float((cont[nz] * (np.log(cont[nz]) - np.log(outer[nz]))).sum())
                    v = max(0.0, min(1.0, mi / denom))
            pairwise[j, k] = v
            pairwise[k, j] = v
    agreements = pairwise.sum(axis=1) / max(m - 1, 1)
    max_ag = float(agreements.max())
    if max_ag > 1e-12:
        agreements = agreements / max_ag
    return np.clip(agreements, 1e-06, 1.0)

def compute_modified_neci(base_cls: np.ndarray, bcs: np.ndarray, base_cls_segs: np.ndarray, x: np.ndarray | None, para_theta: float) -> np.ndarray:
    fuzzy_weights = compute_fuzzy_entropy_weights(base_cls, x, para_theta)
    k_per_partition = base_cls.max(axis=0).astype(np.int64)
    partition_agreements = compute_partition_agreements(base_cls)
    partition_cluster_weights = np.repeat(partition_agreements, k_per_partition)
    weights = fuzzy_weights * partition_cluster_weights
    max_weight = float(np.max(weights))
    if max_weight > 0:
        weights = weights / max_weight
    return np.clip(weights, 1e-06, 1.0)

def graph_diffusion_of_cluster(w: np.ndarray, diffusion_time: float=1.0) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    degree = np.sum(w, axis=1)
    inv_sqrt = np.zeros_like(degree)
    valid = degree > 1e-12
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    normalized = inv_sqrt[:, None] * w * inv_sqrt[None, :]
    laplacian = np.eye(w.shape[0], dtype=np.float64) - normalized
    heat = expm(-diffusion_time * laplacian)
    heat = (heat + heat.T) / 2.0
    diag = np.sqrt(np.maximum(np.diag(heat), 1e-12))
    heat = heat / np.outer(diag, diag)
    heat = np.clip(heat, 0.0, 1.0)
    np.fill_diagonal(heat, 1.0)
    return heat

def compute_adaptive_diffusion_time(sim_matrix: np.ndarray, clip_range: tuple[float, float]=(0.1, 5.0), fallback: float=1.0) -> float:
    w = np.asarray(sim_matrix, dtype=np.float64)
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    degree = w.sum(axis=1)
    if float(degree.max()) < 1e-12:
        return fallback
    inv_sqrt = np.zeros_like(degree)
    valid = degree > 1e-12
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    normalized = inv_sqrt[:, None] * w * inv_sqrt[None, :]
    laplacian = np.eye(w.shape[0], dtype=np.float64) - normalized
    try:
        n_lap = laplacian.shape[0]
        if n_lap < 2:
            return fallback
        eigenvalues = eigh(laplacian, eigvals_only=True, subset_by_index=[0, 1])
        lambda2 = float(eigenvalues[1]) if eigenvalues.size > 1 else 0.0
        if lambda2 < 1e-06:
            return fallback
        return float(np.clip(1.0 / lambda2, clip_range[0], clip_range[1]))
    except Exception:
        return fallback

def compute_d_diffusion(bcs: np.ndarray, base_cls_segs: np.ndarray, tau: float=0.8, diffusion_time: float=1.0, adaptive_tau: bool=True, tau_percentile: float=60.0) -> np.ndarray:
    n, m = bcs.shape
    d = np.zeros((n, n), dtype=np.float64)
    sim_of_cluster = simxjac(base_cls_segs)
    diffusion_similarity = graph_diffusion_of_cluster(sim_of_cluster, diffusion_time=diffusion_time)
    dis_of_cluster = 1.0 - diffusion_similarity
    for j in range(m):
        idx = bcs[:, j].astype(np.int64) - 1
        d = d + dis_of_cluster[np.ix_(idx, idx)]
    d = d / m
    if adaptive_tau:
        nonzero_vals = d[d > 1e-09]
        if nonzero_vals.size > 0:
            effective_tau = float(np.percentile(nonzero_vals, tau_percentile))
        else:
            effective_tau = tau
    else:
        effective_tau = tau
    d[d < effective_tau] = 0.0
    return d

def run_sdgca_modified(dataset_path: str | Path, data_name: str | None=None, seed: int=19, m: int=20, cnt_times: int=10, nwca_para: float=0.09, eta: float=0.75, theta: float=0.65, method: str='average', diffusion_time: float | None=1.0, adaptive_tau: bool=True, tau_percentile: float=60.0, selection_strategy: str='random', qd_alpha: float=0.5) -> dict:
    from consensus_runner import run_consensus_loop
    adaptive_mode = diffusion_time is None
    if not adaptive_mode and diffusion_time <= 0:
        raise ValueError('diffusion_time must be positive (or None for adaptive)')
    members_full, gt_full, x = load_dataset_full(dataset_path)
    base_x = None if x is None else np.asarray(x, dtype=np.float64)
    diffusion_times_used: list[float] = []

    def _build(base_cls, _gt, m_actual):
        bcs, base_cls_segs = get_all_segs(base_cls)
        ca = base_cls_segs.T @ base_cls_segs / m_actual
        neci = compute_modified_neci(base_cls, bcs, base_cls_segs, base_x, nwca_para)
        nwca = compute_nwca(base_cls_segs, neci, m_actual)
        if eta > 1:
            return nwca
        hc = ca.copy()
        hc[hc < eta] = 0.0
        l_matrix = np.diag(np.sum(hc, axis=1)) - hc
        mla = ca.copy()
        mla[mla < theta] = 0.0
        ml = compute_s(nwca, mla)
        if adaptive_mode:
            sim_of_cluster = simxjac(base_cls_segs)
            t = compute_adaptive_diffusion_time(sim_of_cluster)
        else:
            t = float(diffusion_time)
        diffusion_times_used.append(t)
        cl = compute_d_diffusion(bcs, base_cls_segs, diffusion_time=t, adaptive_tau=adaptive_tau, tau_percentile=tau_percentile)
        ml[cl > 0] = 0.0
        s, d = optimize_sdgca(l_matrix, ml, cl)
        return compute_w(s, d, nwca)
    result = run_consensus_loop(dataset_path, _build, data_name=data_name, seed=seed, m=m, cnt_times=cnt_times, method=method, selection_strategy=selection_strategy, qd_alpha=qd_alpha, clamp_m_name='sdgca_modified')
    if diffusion_times_used:
        result['diffusion_time_used'] = round(float(np.mean(diffusion_times_used)), 4)
        result['diffusion_time_per_run'] = [round(t, 4) for t in diffusion_times_used]
    else:
        result['diffusion_time_used'] = None if adaptive_mode else float(diffusion_time)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='SDGCA modified.')
    parser.add_argument('--dataset', default='Ecoli')
    parser.add_argument('--root', default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--lambda_', type=float, default=None)
    parser.add_argument('--eta', type=float, default=None)
    parser.add_argument('--theta', type=float, default=None)
    parser.add_argument('--diffusion_time', type=float, default=None)
    parser.add_argument('--adaptive', action='store_true', help='Использовать adaptive diffusion time (игнорирует --diffusion_time)')
    parser.add_argument('--method', default='average', choices=sorted(SUPPORTED_LINKAGE_METHODS))
    args = parser.parse_args()
    params = resolve_params(args.dataset, args.lambda_, args.eta, args.theta, args.diffusion_time)
    diffusion_time_arg: float | None = None if args.adaptive else params['diffusion_time']
    dataset_path = Path(args.root) / f'{args.dataset}.mat'
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f'{args.dataset}.npz'
    result = run_sdgca_modified(dataset_path=dataset_path, data_name=args.dataset, seed=args.seed, m=args.m, cnt_times=args.runs, nwca_para=params['lambda_'], eta=params['eta'], theta=params['theta'], method=args.method, diffusion_time=diffusion_time_arg)
    mode = 'adaptive' if args.adaptive else f"fixed t={params['diffusion_time']:.2f}"
    print(f'Mode: {mode}')
    if result.get('diffusion_time_used') is not None:
        print(f"Diffusion time used: {result['diffusion_time_used']}")
    print('           mean    variance')
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")
if __name__ == '__main__':
    main()
