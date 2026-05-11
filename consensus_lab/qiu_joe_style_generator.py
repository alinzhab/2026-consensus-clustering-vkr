from __future__ import annotations
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def build_simplex_vertices(p: int) -> np.ndarray:
    vertices = np.zeros((p + 1, p))
    vertices[0, 0] = -1.0
    vertices[1, 0] = 1.0
    for k in range(2, p + 1):
        prev = vertices[:k]
        v_bar = prev.mean(axis=0)
        new_v = v_bar.copy()
        new_v[k:] = 0.0
        sq_sum = np.sum((prev - v_bar) ** 2)
        under = 4.0 - sq_sum / k
        if under < 0.0:
            under = 0.0
        new_v[k - 1] = np.sqrt(under)
        vertices[k] = new_v
    return vertices

def get_cluster_centers_from_simplex(K: int, p: int) -> np.ndarray:
    verts = build_simplex_vertices(p)
    centers = []
    if K <= p + 1:
        centers = [verts[i] for i in range(K)]
    else:
        centers = list(verts)
        shift_step = 0
        extra_needed = K - (p + 1)
        while extra_needed > 0:
            shift_step += 1
            shift = np.zeros(p)
            shift[0] = 2.0 * shift_step
            for i in range(1, p + 1):
                if extra_needed <= 0:
                    break
                centers.append(verts[i] + shift)
                extra_needed -= 1
    return np.array(centers[:K])

def separation_index(a: np.ndarray, mu1: np.ndarray, mu2: np.ndarray, sigma1: np.ndarray, sigma2: np.ndarray, alpha: float=0.05) -> float:
    z = norm.ppf(1.0 - alpha / 2.0)
    diff = mu2 - mu1
    aT_diff = float(a @ diff)
    if aT_diff < 0:
        a = -a
        aT_diff = -aT_diff
    s1 = float(np.sqrt(max(a @ sigma1 @ a, 0.0)))
    s2 = float(np.sqrt(max(a @ sigma2 @ a, 0.0)))
    denom = aT_diff + z * (s1 + s2)
    if denom <= 0.0:
        return -1.0
    numer = aT_diff - z * (s1 + s2)
    return numer / denom

def _g_objective(y: np.ndarray, Q1: np.ndarray, Q2: np.ndarray, V: np.ndarray, v11: float, v21: np.ndarray, V22: np.ndarray, c2: float) -> float:
    g1 = float(y @ y) + 1.0
    V22_y_v21 = V22 @ y + v21
    inner = float(V22_y_v21 @ V22_y_v21) + c2
    g2 = inner
    val = np.sqrt(max(g1, 0.0)) + np.sqrt(max(g2, 0.0))
    return val

def optimal_separation_and_direction(mu1: np.ndarray, mu2: np.ndarray, sigma1: np.ndarray, sigma2: np.ndarray, alpha: float=0.05) -> tuple[float, np.ndarray]:
    p = len(mu1)
    diff = mu2 - mu1
    if np.allclose(diff, 0.0):
        return (-1.0, diff / (np.linalg.norm(diff) + 1e-12))
    try:
        L1 = np.linalg.cholesky(sigma1 + 1e-09 * np.eye(p))
        Q1 = np.linalg.inv(L1).T
    except np.linalg.LinAlgError:
        Q1 = np.eye(p)
    Q1T_diff = Q1.T @ diff
    c1 = np.linalg.norm(Q1T_diff)
    if c1 < 1e-12:
        return (separation_index(diff, mu1, mu2, sigma1, sigma2, alpha), diff)
    e1 = np.zeros(p)
    e1[0] = 1.0
    u = Q1T_diff / c1
    v_house = u - e1
    if np.linalg.norm(v_house) < 1e-12:
        Q2 = np.eye(p)
    else:
        v_house = v_house / np.linalg.norm(v_house)
        Q2 = np.eye(p) - 2.0 * np.outer(v_house, v_house)
    V = Q2.T @ Q1.T @ sigma2 @ Q1 @ Q2
    v11 = float(V[0, 0])
    v21 = V[1:, 0].copy()
    V22 = V[1:, 1:].copy()
    c2 = v11 - float(v21 @ np.linalg.pinv(V22) @ v21) if p > 1 else v11
    y0 = (Q2.T @ Q1.T @ diff)[1:]
    if len(y0) == 0:
        a_star = diff / np.linalg.norm(diff)
        j_star = separation_index(a_star, mu1, mu2, sigma1, sigma2, alpha)
        return (j_star, a_star)
    try:
        res = minimize(_g_objective, y0, args=(Q1, Q2, V, v11, v21, V22, c2), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        y_opt = res.x
    except Exception:
        y_opt = y0
    y_full = np.concatenate([[1.0], y_opt])
    a_tilde = Q2 @ y_full
    a_star = Q1 @ a_tilde
    norm_a = np.linalg.norm(a_star)
    if norm_a < 1e-12:
        a_star = diff / np.linalg.norm(diff)
    else:
        a_star = a_star / norm_a
    j_star = separation_index(a_star, mu1, mu2, sigma1, sigma2, alpha)
    return (j_star, a_star)

def compute_separation_matrix(means: np.ndarray, covariances: list[np.ndarray], alpha: float=0.05) -> np.ndarray:
    K = len(means)
    J = np.full((K, K), -1.0)
    for i in range(K):
        for j in range(i + 1, K):
            j_star, _ = optimal_separation_and_direction(means[i], means[j], covariances[i], covariances[j], alpha)
            J[i, j] = j_star
            J[j, i] = j_star
    return J

def random_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.standard_normal((p, p))
    M = np.tril(M)
    np.fill_diagonal(M, np.abs(np.diag(M)) + 1e-06)
    Q, _ = np.linalg.qr(M)
    return Q

def random_covariance(p: int, lam_min: float=1.0, r_lambda: float=10.0, rng: np.random.Generator=None) -> np.ndarray:
    lam_max = lam_min * r_lambda
    eigvals = rng.uniform(lam_min, lam_max, size=p)
    eigvals = np.sort(eigvals)[::-1]
    Q = random_orthogonal(p, rng)
    cov = Q @ np.diag(eigvals) @ Q.T
    cov = (cov + cov.T) / 2.0
    return cov

def cluster_center_allocation(K: int, p: int, covariances: list[np.ndarray], J0: float, alpha: float=0.05, max_cov_iters: int=50) -> tuple[np.ndarray, list[np.ndarray]]:
    covariances = [cov.copy() for cov in covariances]
    base_centers = get_cluster_centers_from_simplex(K, p)

    def min_separation(scale: float) -> float:
        centers = base_centers * scale
        J_mat = compute_separation_matrix(centers, covariances, alpha)
        off = J_mat[~np.eye(K, dtype=bool)]
        return float(np.min(off))
    lo, hi = (0.01, 1.0)
    for _ in range(30):
        if min_separation(hi) >= J0:
            break
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if min_separation(mid) < J0:
            lo = mid
        else:
            hi = mid
    c1 = (lo + hi) / 2.0
    means = base_centers * c1
    for iteration in range(max_cov_iters):
        J_mat = compute_separation_matrix(means, covariances, alpha)
        J_k_min = np.array([np.min([J_mat[k, j] for j in range(K) if j != k]) for k in range(K)])
        k_star = int(np.argmax(J_k_min))
        j_k_star_min = J_k_min[k_star]
        if j_k_star_min <= J0 + 0.0001:
            break

        def sep_k_star(c2: float) -> float:
            cov_scaled = covariances[k_star] * c2
            return float(np.min([optimal_separation_and_direction(means[k_star], means[j], cov_scaled, covariances[j], alpha)[0] for j in range(K) if j != k_star]))
        lo2, hi2 = (1.0, 1.0)
        while sep_k_star(hi2) > J0:
            hi2 *= 2.0
            if hi2 > 1000000.0:
                break
        for _ in range(60):
            mid2 = (lo2 + hi2) / 2.0
            if sep_k_star(mid2) > J0:
                lo2 = mid2
            else:
                hi2 = mid2
        c2 = (lo2 + hi2) / 2.0
        covariances[k_star] = covariances[k_star] * c2
    return (means, covariances)

def generate_noisy_variables(X_signal: np.ndarray, means: np.ndarray, cluster_sizes: np.ndarray, p2: int, rng: np.random.Generator) -> np.ndarray:
    if p2 == 0:
        return np.empty((X_signal.shape[0], 0))
    N = X_signal.shape[0]
    K = len(means)
    pi_k = cluster_sizes / cluster_sizes.sum()
    mu_mix = np.sum(pi_k[:, None] * means, axis=0)
    var_signal = np.var(X_signal, axis=0)
    lam_min_sig = float(np.min(var_signal))
    lam_max_sig = float(np.max(var_signal))
    mu_noise_min = float(np.min(mu_mix))
    mu_noise_max = float(np.max(mu_mix))
    eigvals_noise = rng.uniform(lam_min_sig, lam_max_sig, size=p2)
    Q_noise = random_orthogonal(p2, rng)
    Sigma_noise = Q_noise @ np.diag(eigvals_noise) @ Q_noise.T
    Sigma_noise = (Sigma_noise + Sigma_noise.T) / 2.0
    mu0 = rng.uniform(mu_noise_min, mu_noise_max, size=p2)
    X_noise = rng.multivariate_normal(mu0, Sigma_noise, size=N)
    return X_noise

def generate_outliers(X: np.ndarray, n_outliers: int, rng: np.random.Generator) -> np.ndarray:
    if n_outliers == 0:
        return np.empty((0, X.shape[1]))
    mu_j = np.mean(X, axis=0)
    sigma_j = np.std(X, axis=0)
    low = mu_j - 4.0 * sigma_j
    high = mu_j + 4.0 * sigma_j
    outliers = rng.uniform(low, high, size=(n_outliers, X.shape[1]))
    return outliers

def apply_random_rotation(X: np.ndarray, means: np.ndarray, covariances: list[np.ndarray], p1: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    Q = random_orthogonal(p1, rng)
    X_rot = X.copy()
    X_rot[:, :p1] = X[:, :p1] @ Q.T
    rotated_means = (Q @ means.T).T
    rotated_covs = [Q @ cov @ Q.T for cov in covariances]
    return (X_rot, rotated_means, rotated_covs)

@dataclass
class ClusterGenResult:
    X: np.ndarray
    labels: np.ndarray
    means: np.ndarray
    covariances: list[np.ndarray]
    means_rotated: np.ndarray
    covariances_rotated: list
    J_population: np.ndarray
    J_sample: np.ndarray
    J_k_min_population: np.ndarray
    J_k_min_sample: np.ndarray
    cluster_sizes: np.ndarray
    rotation_matrix: Optional[np.ndarray] = None
    params: dict = field(default_factory=dict)

def generate_clusters(p1: int=4, K: int=3, J0: float=0.21, alpha: float=0.05, p2: int=0, n_outliers: int=0, lam_min: float=1.0, r_lambda: float=10.0, n_min: int=50, n_max: int=200, cluster_sizes: Optional[np.ndarray]=None, apply_rotation: bool=True, seed: Optional[int]=None) -> ClusterGenResult:
    if p1 < 1:
        raise ValueError('p1 must be positive')
    if K < 2:
        raise ValueError('K must be at least 2')
    if not 0.0 < J0 < 1.0:
        raise ValueError('J0 must be between 0 and 1')
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError('alpha must be between 0 and 1')
    if p2 < 0:
        raise ValueError('p2 must be non-negative')
    if n_outliers < 0:
        raise ValueError('n_outliers must be non-negative')
    if lam_min <= 0.0:
        raise ValueError('lam_min must be positive')
    if r_lambda < 1.0:
        raise ValueError('r_lambda must be at least 1')
    if cluster_sizes is not None:
        cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)
        if cluster_sizes.shape != (K,):
            raise ValueError('cluster_sizes must have shape (K,)')
        if np.any(cluster_sizes < 2):
            raise ValueError('each cluster must contain at least two objects')
    elif n_min < 2 or n_max < n_min:
        raise ValueError('n_min/n_max must satisfy 2 <= n_min <= n_max')
    rng = np.random.default_rng(seed)
    params = dict(p1=p1, K=K, J0=J0, alpha=alpha, p2=p2, n_outliers=n_outliers, lam_min=lam_min, r_lambda=r_lambda, n_min=n_min, n_max=n_max, cluster_sizes=None if cluster_sizes is None else cluster_sizes.astype(int).tolist(), apply_rotation=apply_rotation, seed=seed)
    covariances_init = [random_covariance(p1, lam_min, r_lambda, rng) for _ in range(K)]
    print(f'[Step 2] Размещение {K} кластеров в {p1}D с J0={J0}...')
    means, covariances = cluster_center_allocation(K=K, p=p1, covariances=covariances_init, J0=J0, alpha=alpha)
    print(f'[Step 2] Готово. Центры:\n{means.round(3)}')
    if cluster_sizes is None:
        cluster_sizes = rng.integers(n_min, n_max + 1, size=K)
    N_signal = int(cluster_sizes.sum())
    X_parts = []
    labels_parts = []
    for k in range(K):
        n_k = int(cluster_sizes[k])
        X_k = rng.multivariate_normal(means[k], covariances[k], size=n_k)
        X_parts.append(X_k)
        labels_parts.append(np.full(n_k, k + 1, dtype=np.int64))
    X_signal = np.vstack(X_parts)
    labels = np.concatenate(labels_parts)
    rotation_matrix = None
    means_rotated = means.copy()
    covariances_rotated = [c.copy() for c in covariances]
    if apply_rotation:
        X_signal, means_rotated, covariances_rotated = apply_random_rotation(X_signal, means, covariances, p1, rng)
        rotation_matrix = random_orthogonal(p1, rng)
    if p2 > 0:
        X_noise = generate_noisy_variables(X_signal, means, cluster_sizes, p2, rng)
        X_full = np.hstack([X_signal, X_noise])
    else:
        X_full = X_signal
    if n_outliers > 0:
        X_out = generate_outliers(X_full, n_outliers, rng)
        X_full = np.vstack([X_full, X_out])
        labels = np.concatenate([labels, np.zeros(n_outliers, dtype=np.int64)])
    print('[Step 8] Вычисление популяционной матрицы разделения...')
    J_pop = compute_separation_matrix(means, covariances, alpha)
    J_k_min_pop = np.array([np.min([J_pop[k, j] for j in range(K) if j != k]) for k in range(K)])
    print('[Step 9] Вычисление выборочной матрицы разделения...')
    sample_means = []
    sample_covs = []
    for k in range(K):
        mask = labels == k + 1
        X_k_data = X_full[mask, :p1]
        sample_means.append(np.mean(X_k_data, axis=0))
        sample_covs.append(np.cov(X_k_data, rowvar=False) if X_k_data.shape[0] > 1 else covariances[k])
    J_sam = compute_separation_matrix(np.array(sample_means), sample_covs, alpha)
    J_k_min_sam = np.array([np.min([J_sam[k, j] for j in range(K) if j != k]) for k in range(K)])
    print(f"\n{'=' * 55}")
    print(f'Заданный J0                    = {J0:.4f}')
    print(f'Популяционный J*_k_min (min)   = {np.min(J_k_min_pop):.4f}')
    print(f'Популяционный J*_k_min (mean)  = {np.mean(J_k_min_pop):.4f}')
    print(f'Выборочный    J*_k_min (min)   = {np.min(J_k_min_sam):.4f}')
    print(f'Выборочный    J*_k_min (mean)  = {np.mean(J_k_min_sam):.4f}')
    print(f'Форма данных                   = {X_full.shape}')
    print(f'Размеры кластеров              = {cluster_sizes.tolist()}')
    print(f"{'=' * 55}\n")
    return ClusterGenResult(X=X_full, labels=labels, means=means, covariances=covariances, means_rotated=means_rotated, covariances_rotated=covariances_rotated, J_population=J_pop, J_sample=J_sam, J_k_min_population=J_k_min_pop, J_k_min_sample=J_k_min_sam, cluster_sizes=cluster_sizes, rotation_matrix=rotation_matrix, params=params)

def _exact_cluster_sizes(total: int, n_clusters: int, imbalance_ratio: float, rng: np.random.Generator) -> np.ndarray:
    if total < 2 * n_clusters:
        raise ValueError('n_samples is too small: need at least two objects per cluster after outliers')
    if imbalance_ratio < 1.0:
        raise ValueError('imbalance_ratio must be at least 1')
    if np.isclose(imbalance_ratio, 1.0):
        weights = np.ones(n_clusters, dtype=np.float64)
    else:
        weights = np.geomspace(1.0, float(imbalance_ratio), n_clusters)
        rng.shuffle(weights)
    raw = weights / weights.sum() * total
    sizes = np.floor(raw).astype(np.int64)
    sizes[sizes < 2] = 2
    while sizes.sum() < total:
        residual = raw - sizes
        sizes[int(np.argmax(residual))] += 1
    while sizes.sum() > total:
        candidates = np.where(sizes > 2)[0]
        if candidates.size == 0:
            break
        idx = int(candidates[np.argmax(sizes[candidates])])
        sizes[idx] -= 1
    return sizes

def generate_qiu_joe_style_dataset(name='qiu_joe_style_dataset', n_samples=2000, n_clusters=6, dim=10, overlap_level='medium', separation=1.0, shape_ratio=6.0, volume_mean=1.0, imbalance_ratio=2.0, orientation='random', noise_ratio=0.0, seed=19, base_clusterings=30, base_k_min=2, base_k_max=8, base_strategy='mixed'):
    if overlap_level not in {'low', 'medium', 'high'}:
        raise ValueError('overlap_level must be one of: low, medium, high')
    if orientation not in {'random', 'axis_aligned'}:
        raise ValueError('orientation must be one of: random, axis_aligned')
    if n_clusters < 2:
        raise ValueError('n_clusters must be at least 2')
    if dim < 1:
        raise ValueError('dim must be positive')
    if n_samples < 2 * n_clusters:
        raise ValueError('n_samples must be at least 2 * n_clusters')
    if separation <= 0.0:
        raise ValueError('separation must be positive')
    if shape_ratio < 1.0:
        raise ValueError('shape_ratio must be at least 1')
    if volume_mean <= 0.0:
        raise ValueError('volume_mean must be positive')
    if not 0.0 <= noise_ratio < 0.5:
        raise ValueError('noise_ratio must be in [0, 0.5)')
    if base_clusterings < 1:
        raise ValueError('base_clusterings must be positive')
    rng = np.random.default_rng(seed)
    target_j0 = {'high': 0.01, 'medium': 0.21, 'low': 0.342}[overlap_level]
    target_j0 = float(np.clip(target_j0 * separation, 0.001, 0.95))
    n_outliers = int(round(int(n_samples) * float(noise_ratio)))
    n_signal = int(n_samples) - n_outliers
    sizes = _exact_cluster_sizes(n_signal, int(n_clusters), float(imbalance_ratio), rng)
    result = generate_clusters(p1=int(dim), K=int(n_clusters), J0=target_j0, alpha=0.05, p2=0, n_outliers=n_outliers, lam_min=float(volume_mean), r_lambda=float(shape_ratio), cluster_sizes=sizes, apply_rotation=orientation == 'random', seed=seed)
    x = np.asarray(result.X, dtype=np.float64)
    gt = np.asarray(result.labels, dtype=np.int64)
    if np.any(gt == 0):
        gt = gt.copy()
        gt[gt == 0] = int(n_clusters) + 1
    from base_clusterings import build_base_clusterings
    members, base_info = build_base_clusterings(x, n_clusterings=int(base_clusterings), k_min=int(base_k_min), k_max=int(base_k_max), rng=seed, strategy=base_strategy, return_info=True)
    meta = {'name': name, 'generator': 'qiu_joe_cluster_generation_style', 'reference': 'Qiu & Joe (2006), Generation of Random Clusters with Specified Degree of Separation', 'n_samples': int(n_samples), 'n_samples_actual': int(x.shape[0]), 'n_clusters': int(n_clusters), 'dim': int(dim), 'overlap_level': overlap_level, 'target_j0': target_j0, 'separation': float(separation), 'shape_ratio': float(shape_ratio), 'volume_mean': float(volume_mean), 'imbalance_ratio': float(imbalance_ratio), 'orientation': orientation, 'noise_ratio': float(noise_ratio), 'n_outliers': int(n_outliers), 'cluster_sizes': sizes.astype(int).tolist(), 'j_population_min': float(np.min(result.J_k_min_population)), 'j_population_mean': float(np.mean(result.J_k_min_population)), 'j_sample_min': float(np.min(result.J_k_min_sample)), 'j_sample_mean': float(np.mean(result.J_k_min_sample)), 'seed': int(seed), 'base_clusterings': int(base_clusterings), 'base_k_min': int(base_k_min), 'base_k_max': int(base_k_max), 'base_strategy': base_strategy, 'base_info': base_info}
    return (x, gt, members, meta)

def save_dataset(path, x, gt, members, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=np.asarray(x, dtype=np.float64), gt=np.asarray(gt, dtype=np.int64), members=np.asarray(members, dtype=np.int64), meta=json.dumps(meta, ensure_ascii=False))

def factorial_experiment(K_levels: list[int]=(3, 6, 9), J0_levels: list[float]=(0.01, 0.21, 0.342), p1_levels: list[int]=(4, 8, 20), p2_ratios: list[float]=(0.0, 0.5, 1.0), n_replicates: int=3, base_seed: int=42, **kwargs) -> list[dict]:
    results = []
    seed = base_seed
    total = len(K_levels) * len(J0_levels) * len(p1_levels) * len(p2_ratios) * n_replicates
    done = 0
    for K in K_levels:
        for J0 in J0_levels:
            for p1 in p1_levels:
                for p2_ratio in p2_ratios:
                    p2 = max(1, int(round(p2_ratio * p1))) if p2_ratio > 0 else 0
                    for rep in range(n_replicates):
                        done += 1
                        print(f'\n[{done}/{total}] K={K}, J0={J0}, p1={p1}, p2={p2}, rep={rep + 1}')
                        try:
                            result = generate_clusters(p1=p1, K=K, J0=J0, p2=p2, seed=seed, **kwargs)
                            results.append({'K': K, 'J0': J0, 'p1': p1, 'p2': p2, 'replicate': rep + 1, 'J_k_min_pop': result.J_k_min_population.tolist(), 'J_k_min_sam': result.J_k_min_sample.tolist(), 'cluster_sizes': result.cluster_sizes.tolist(), 'n_samples': result.X.shape[0]})
                        except Exception as e:
                            warnings.warn(f'Ошибка при K={K},J0={J0},p1={p1},p2={p2},rep={rep}: {e}')
                        seed += 1
    return results
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('=' * 55)
    print('Qiu & Joe (2006) — демонстрация алгоритма')
    print('=' * 55)
    print('\n--- Пример 1: 5 кластеров, 2D, J0=0.01 (close) ---')
    res1 = generate_clusters(p1=2, K=5, J0=0.01, alpha=0.05, p2=0, n_outliers=0, lam_min=1.0, r_lambda=10.0, n_min=100, n_max=200, apply_rotation=False, seed=42)
    print('\n--- Пример 2: 3 кластера, 4D, J0=0.342 (well-separated) ---')
    res2 = generate_clusters(p1=4, K=3, J0=0.342, alpha=0.05, p2=2, n_outliers=10, lam_min=1.0, r_lambda=10.0, n_min=50, n_max=150, apply_rotation=True, seed=7)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes:
        ax.set_facecolor('#0f0f1a')
    colors = ['#e63946', '#f4a261', '#2a9d8f', '#a8dadc', '#457b9d']
    ax = axes[0]
    for k in range(5):
        mask = res1.labels == k + 1
        ax.scatter(res1.X[mask, 0], res1.X[mask, 1], s=15, alpha=0.6, color=colors[k], label=f'Кластер {k + 1}')
    ax.set_title(f'J₀ = 0.01 (close)\nJ*_k_min ≈ {np.min(res1.J_k_min_population):.3f}', color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e', edgecolor='#333')
    ax = axes[1]
    res3 = generate_clusters(p1=2, K=3, J0=0.342, alpha=0.05, p2=0, n_outliers=0, n_min=80, n_max=150, apply_rotation=True, seed=99)
    for k in range(3):
        mask = res3.labels == k + 1
        ax.scatter(res3.X[mask, 0], res3.X[mask, 1], s=15, alpha=0.7, color=colors[k], label=f'Кластер {k + 1}')
    ax.set_title(f'J₀ = 0.342 (well-separated)\nJ*_k_min ≈ {np.min(res3.J_k_min_population):.3f}', color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e', edgecolor='#333')
    plt.suptitle('Qiu & Joe (2006) — генерация кластеров с заданным J₀', color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/qiu_joe_clusters.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print('\nГрафик сохранён: qiu_joe_clusters.png')
    print('\n' + '=' * 55)
    print('Верификация (аналог Table 2 из статьи):')
    print(f"{'J0':>8} {'mean':>8} {'std':>8} {'bias':>8} {'RMSE':>8}")
    for res, J0 in [(res1, 0.01), (res2, 0.342), (res3, 0.342)]:
        vals = res.J_k_min_sample
        m = np.mean(vals)
        s = np.std(vals)
        bias = m - J0
        rmse = np.sqrt(s ** 2 + bias ** 2)
        print(f'{J0:>8.3f} {m:>8.4f} {s:>8.4f} {bias:>8.4f} {rmse:>8.4f}')
    print('=' * 55)
