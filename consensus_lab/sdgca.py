from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from hierarchical_consensus import SUPPORTED_LINKAGE_METHODS, get_cls_result, load_dataset, load_dataset_full
__all__ = ['compute_d', 'compute_neci', 'compute_norm_k', 'compute_nwca', 'compute_s', 'compute_w', 'get_all_cls_entropy', 'get_all_segs', 'load_dataset', 'load_dataset_full', 'optimize_sdgca', 'random_walk_of_cluster', 'run_sdgca', 'simxjac']

def get_all_segs(base_cls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, m = base_cls.shape
    base_cls = base_cls.copy()
    for j in range(m):
        _, inv = np.unique(base_cls[:, j], return_inverse=True)
        base_cls[:, j] = inv + 1
    n_cls_orig = base_cls.max(axis=0).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(n_cls_orig)[:-1]))
    bcs = base_cls + offsets
    n_cls = int(np.cumsum(n_cls_orig)[-1])
    base_cls_segs = np.zeros((n_cls, n), dtype=np.float64)
    rows = bcs.ravel(order='F') - 1
    cols = np.tile(np.arange(n), m)
    base_cls_segs[rows, cols] = 1.0
    return (bcs, base_cls_segs)

def compute_norm_k(bcs: np.ndarray) -> np.ndarray:
    clust = bcs.max(axis=0).astype(np.int64)
    k_per = np.diff(np.concatenate(([0], clust)))
    return np.repeat(k_per, k_per)

def get_one_cls_entropy(part_bcs: np.ndarray, norm_k_i: float) -> float:
    denom = np.log2(norm_k_i)
    if denom == 0:
        return 0.0
    entropy = 0.0
    for i in range(part_bcs.shape[1]):
        _, counts = np.unique(part_bcs[:, i], return_counts=True)
        if counts.size <= 1:
            continue
        probs = counts / counts.sum()
        entropy -= float(np.sum(probs * np.log2(probs)))
    return float(entropy / denom)

def get_all_cls_entropy(bcs: np.ndarray, base_cls_segs: np.ndarray) -> np.ndarray:
    base_cls_segs_t = base_cls_segs.T
    _, n_cls = base_cls_segs_t.shape
    entropies = np.zeros(n_cls, dtype=np.float64)
    norm_k = compute_norm_k(bcs)
    for i in range(n_cls):
        part_bcs = bcs[base_cls_segs_t[:, i] != 0, :]
        entropies[i] = get_one_cls_entropy(part_bcs, norm_k[i])
    return entropies

def compute_neci(bcs: np.ndarray, base_cls_segs: np.ndarray, para_theta: float) -> np.ndarray:
    m = bcs.shape[1]
    ets = get_all_cls_entropy(bcs, base_cls_segs)
    return np.exp(-ets / para_theta / m)

def compute_nwca(base_cls_segs: np.ndarray, neci: np.ndarray, m: int) -> np.ndarray:
    base_cls_segs_t = base_cls_segs.T
    nwca = base_cls_segs_t * neci @ base_cls_segs_t.T / m
    nwca = (nwca + nwca.T) / 2.0
    max_value = float(np.max(nwca))
    if max_value > 0:
        nwca = nwca / max_value
    np.fill_diagonal(nwca, 1.0)
    return nwca

def simxjac(a: np.ndarray, b: np.ndarray | None=None) -> np.ndarray:
    if b is None:
        b = a
    temp = a @ b.T
    asquared = np.sum(a ** 2, axis=1, keepdims=True)
    bsquared = np.sum(b ** 2, axis=1, keepdims=True).T
    denom = asquared + bsquared - temp
    out = np.zeros_like(temp, dtype=np.float64)
    np.divide(temp, denom, out=out, where=denom != 0)
    return out

def random_walk_of_cluster(w: np.ndarray, k: int=20, beta: float=1.0) -> np.ndarray:
    n = w.shape[0]
    w = w.copy()
    np.fill_diagonal(w, 0.0)
    col_sums = np.sum(w, axis=0)
    d = np.zeros((n, n), dtype=np.float64)
    valid = col_sums != 0
    d[valid, valid] = 1.0 / col_sums[valid]
    w_tilde = d @ w
    tmp_o = w_tilde.copy()
    o_tilde = w_tilde @ w_tilde.T
    for _ in range(k - 1):
        tmp_o = tmp_o @ w_tilde
        o_tilde = o_tilde + beta * (tmp_o @ tmp_o.T)
    diag_vals = np.diag(o_tilde)
    denom = np.sqrt(np.outer(diag_vals, diag_vals))
    r = np.zeros_like(o_tilde, dtype=np.float64)
    np.divide(o_tilde, denom, out=r, where=denom != 0)
    isolated_idx = np.where(np.sum(w_tilde, axis=1) < 1e-09)[0]
    if isolated_idx.size > 0:
        r[isolated_idx, :] = 0.0
        r[:, isolated_idx] = 0.0
    return r

def compute_d(bcs: np.ndarray, base_cls_segs: np.ndarray, tau: float=0.8) -> np.ndarray:
    n, m = bcs.shape
    d = np.zeros((n, n), dtype=np.float64)
    sim_of_cluster = simxjac(base_cls_segs)
    rw_of_cluster = random_walk_of_cluster(sim_of_cluster)
    np.fill_diagonal(rw_of_cluster, 1.0)
    dis_of_cluster = 1.0 - rw_of_cluster
    for j in range(m):
        idx = bcs[:, j].astype(np.int64) - 1
        d = d + dis_of_cluster[np.ix_(idx, idx)]
    d = d / m
    d[d < tau] = 0.0
    return d

def compute_s(nwca: np.ndarray, mla: np.ndarray) -> np.ndarray:
    # Mask of "must-link candidate" cells (mla>0). Outside the mask we
    # always return 0. Inside, we rescale nwca to [0.8, 1.0]. The previous
    # implementation relied on a floating-point equality check
    # (ml == 0.8) to re-zero masked cells, which is fragile and could
    # accidentally re-zero genuine values that landed exactly on 0.8.
    keep = mla > 0
    ml = np.zeros_like(nwca, dtype=np.float64)
    if not np.any(keep):
        return ml
    values = nwca[keep]
    lo = float(np.min(values))
    span = float(np.max(values) - lo)
    if span > 0:
        values = (values - lo) / span
    else:
        values = np.zeros_like(values)
    ml[keep] = values / 5.0 + 0.8
    return ml

def optimize_sdgca(l_matrix: np.ndarray, ml: np.ndarray, cl: np.ndarray, max_iter: int=120, tol: float=1e-3) -> Tuple[np.ndarray, np.ndarray]:
    n = l_matrix.shape[0]
    identity = np.eye(n, dtype=np.float64)
    l_sym = (l_matrix + l_matrix.T) / 2.0
    mu1 = 1.0
    mu2 = 1.0
    s = np.zeros((n, n), dtype=np.float64)
    d = np.zeros((n, n), dtype=np.float64)
    y1 = np.zeros((n, n), dtype=np.float64)
    y2 = np.zeros((n, n), dtype=np.float64)
    f1 = np.zeros((n, n), dtype=np.float64)
    f2 = np.zeros((n, n), dtype=np.float64)
    # mu1 and mu2 change every iteration, so the Cholesky factor must be
    # refreshed every iteration too. The previous code re-used a stale
    # factor for several iterations after mu was grown, which silently
    # solved the wrong linear system. We refactor every step.
    for it in range(max_iter):
        s_prev = s.copy()
        d_prev = d.copy()
        chol1 = cho_factor(2.0 * l_sym + 2.0 * mu1 * identity, lower=False, overwrite_a=False, check_finite=False)
        rhs_s = 2.0 * mu1 * f1 - d - y1
        s = cho_solve(chol1, rhs_s, overwrite_b=False, check_finite=False)
        s = (s + s.T) / 2.0
        f1 = y1 / (2.0 * mu1) + s
        f1[ml > 0] = 0.0
        np.clip(f1, 0.0, 1.0, out=f1)
        f1 = f1 + ml
        f1 = (f1 + f1.T) / 2.0
        chol2 = cho_factor(2.0 * l_sym + 2.0 * mu2 * identity, lower=False, overwrite_a=False, check_finite=False)
        rhs_d = 2.0 * mu2 * f2 - s - y2
        d = cho_solve(chol2, rhs_d, overwrite_b=False, check_finite=False)
        d = (d + d.T) / 2.0
        f2 = y2 / (2.0 * mu2) + d
        f2[cl > 0] = 0.0
        np.clip(f2, 0.0, 1.0, out=f2)
        f2 = f2 + cl
        f2 = (f2 + f2.T) / 2.0
        y1 = y1 + mu1 * (s - f1)
        y2 = y2 + mu2 * (d - f2)
        mu1 = min(mu1 * 1.1, 1e6)
        mu2 = min(mu2 * 1.1, 1e6)
        err = max(
            np.linalg.norm(s - s_prev, ord='fro'),
            np.linalg.norm(d - d_prev, ord='fro'),
            np.linalg.norm(s - f1, ord='fro'),
            np.linalg.norm(d - f2, ord='fro'),
        )
        if err < tol:
            break
    return (s, d)

def compute_w(s: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    s = (s + s.T) / 2.0
    d = np.clip(d, 0.0, 1.0)
    d = (d + d.T) / 2.0
    w1 = 1.0 - (1.0 - s + d) * (1.0 - w)
    w2 = (1.0 + s - d) * w
    flag = s - d
    w1[flag < 0] = 0.0
    w2[flag >= 0] = 0.0
    return w1 + w2

def _clamp_m(m: int, pool_size: int, algorithm_name: str) -> int:
    if m > pool_size:
        warnings.warn(f'{algorithm_name}: requested m={m} exceeds pool size {pool_size}; using m={pool_size}', RuntimeWarning, stacklevel=3)
        return pool_size
    return m

def run_sdgca(dataset_path: str | Path, data_name: str | None=None, seed: int=19, m: int=40, cnt_times: int=20, nwca_para: float=0.09, eta: float=0.75, theta: float=0.65, method: str='average', selection_strategy: str='random', qd_alpha: float=0.5) -> dict:
    from consensus_runner import run_consensus_loop

    def _build(base_cls, _gt, m_actual):
        bcs, base_cls_segs = get_all_segs(base_cls)
        ca = base_cls_segs.T @ base_cls_segs / m_actual
        nwca = compute_nwca(base_cls_segs, compute_neci(bcs, base_cls_segs, nwca_para), m_actual)
        if eta > 1:
            return nwca
        hc = ca.copy()
        hc[hc < eta] = 0.0
        l_matrix = np.diag(np.sum(hc, axis=1)) - hc
        mla = ca.copy()
        mla[mla < theta] = 0.0
        ml = compute_s(nwca, mla)
        cl = compute_d(bcs, base_cls_segs)
        ml[cl > 0] = 0.0
        s, d = optimize_sdgca(l_matrix, ml, cl)
        return compute_w(s, d, nwca)
    return run_consensus_loop(dataset_path, _build, data_name=data_name, seed=seed, m=m, cnt_times=cnt_times, method=method, selection_strategy=selection_strategy, qd_alpha=qd_alpha, clamp_m_name='sdgca')

def main() -> None:
    parser = argparse.ArgumentParser(description='SDGCA baseline.')
    parser.add_argument('--dataset', default='Ecoli')
    parser.add_argument('--root', default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--lambda_', type=float, default=0.09)
    parser.add_argument('--eta', type=float, default=0.75)
    parser.add_argument('--theta', type=float, default=0.65)
    parser.add_argument('--method', default='average', choices=sorted(SUPPORTED_LINKAGE_METHODS))
    args = parser.parse_args()
    dataset_path = Path(args.root) / f'{args.dataset}.mat'
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f'{args.dataset}.npz'
    result = run_sdgca(dataset_path=dataset_path, data_name=args.dataset, seed=args.seed, m=args.m, cnt_times=args.runs, nwca_para=args.lambda_, eta=args.eta, theta=args.theta, method=args.method)
    print('           mean    variance')
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")
if __name__ == '__main__':
    main()
