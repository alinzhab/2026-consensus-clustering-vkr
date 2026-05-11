from __future__ import annotations
import warnings
from typing import Any
import numpy as np
__all__ = ['compute_diagnostics', 'recommend_from_diagnostics']
_MAX_EXPENSIVE = 3000
_KNN_K = 5
_HOPKINS_M = 150

def _as_rng(seed: Any) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)

def _subsample_idx(n: int, rng: np.random.Generator, max_n: int=_MAX_EXPENSIVE) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)

def _clean_X(X_raw: np.ndarray) -> np.ndarray:
    X = X_raw.copy().astype(np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            good = col[~bad]
            fill = float(np.mean(good)) if len(good) else 0.0
            X[bad, j] = fill
    return X

def _r(v: float, decimals: int=6) -> float:
    return round(float(v), decimals)

def _basic(X_raw: np.ndarray) -> dict:
    n, d = X_raw.shape
    total = n * d
    n_nan = int(np.isnan(X_raw).sum())
    n_inf = int(np.isinf(X_raw).sum())
    X_f = np.where(np.isfinite(X_raw), X_raw, np.nan)
    variances = np.nanvar(X_f, axis=0)
    n_const = int((variances == 0).sum())
    return {'n_objects': n, 'n_features': d, 'dimensionality_ratio': _r(d / n), 'missing_ratio': _r(n_nan / total), 'infinite_ratio': _r(n_inf / total), 'constant_feature_ratio': _r(n_const / d), 'feature_variance_mean': _r(float(np.nanmean(variances))), 'feature_variance_std': _r(float(np.nanstd(variances))), 'feature_variance_min': _r(float(np.nanmin(variances))), 'feature_variance_max': _r(float(np.nanmax(variances)))}

def _pca(X_clean: np.ndarray) -> dict:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return {'pca_error': 'sklearn_not_installed'}
    n, d = X_clean.shape
    n_comp = min(n - 1, d)
    if n_comp < 2:
        return {'pca_error': 'too_few_dimensions'}
    try:
        pca = PCA(n_components=n_comp, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pca.fit(X_clean)
        ev = pca.explained_variance_ratio_
        cum = np.cumsum(ev)
        eig = pca.explained_variance_
        exp_2d = float(np.sum(ev[:2])) if len(ev) >= 2 else float(ev[0])
        exp_10d = float(np.sum(ev[:10])) if len(ev) >= 10 else float(np.sum(ev))
        eff_90 = min(int(np.searchsorted(cum, 0.9)) + 1, n_comp)
        eff_95 = min(int(np.searchsorted(cum, 0.95)) + 1, n_comp)
        sum_sq = float(np.sum(eig ** 2))
        pr = float(np.sum(eig) ** 2 / sum_sq) if sum_sq > 0 else float(n_comp)
        return {'explained_variance_2d': round(exp_2d, 4), 'explained_variance_10d': round(exp_10d, 4), 'effective_dimension_90': eff_90, 'effective_dimension_95': eff_95, 'participation_ratio': round(pr, 3)}
    except Exception as exc:
        return {'pca_error': str(exc)}

def _density(X_sub: np.ndarray) -> dict:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return {'density_error': 'sklearn_not_installed'}
    n = X_sub.shape[0]
    k = min(_KNN_K, n - 1)
    if k < 1:
        return {'density_error': 'too_few_objects'}
    try:
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
        nn.fit(X_sub)
        dists, _ = nn.kneighbors(X_sub)
        kd = dists[:, -1]
        mean_d = float(np.mean(kd))
        std_d = float(np.std(kd))
        cv = std_d / mean_d if mean_d > 1e-12 else 0.0
        thr95 = float(np.percentile(kd, 95))
        outlier_ratio = float(np.mean(kd > thr95))
        return {'knn_distance_mean': _r(mean_d), 'knn_distance_std': _r(std_d), 'density_variation': round(cv, 4), 'outlier_ratio': round(outlier_ratio, 4)}
    except Exception as exc:
        return {'density_error': str(exc)}

def _hopkins(X_sub: np.ndarray, rng: np.random.Generator) -> dict:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return {'hopkins': None, 'hopkins_error': 'sklearn_not_installed'}
    n, d = X_sub.shape
    m = min(_HOPKINS_M, n // 3)
    if m < 5:
        return {'hopkins': None, 'hopkins_error': 'too_few_objects'}
    try:
        sample_idx = rng.choice(n, size=m, replace=False)
        X_samp = X_sub[sample_idx]
        mins = X_sub.min(axis=0)
        spans = X_sub.max(axis=0) - mins
        spans[spans == 0] = 1.0
        X_rand = mins + rng.uniform(size=(m, d)) * spans
        nn2 = NearestNeighbors(n_neighbors=2, algorithm='auto')
        nn2.fit(X_sub)
        du, _ = nn2.kneighbors(X_samp)
        u = du[:, 1]
        nn1 = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn1.fit(X_sub)
        dw, _ = nn1.kneighbors(X_rand)
        w = dw[:, 0]
        sw = float(np.sum(w))
        su = float(np.sum(u))
        denom = sw + su
        if denom < 1e-12:
            return {'hopkins': None, 'hopkins_error': 'zero_distances'}
        return {'hopkins': round(sw / denom, 4)}
    except Exception as exc:
        return {'hopkins': None, 'hopkins_error': str(exc)}

def _class_stats(gt: np.ndarray) -> dict:
    classes, counts = np.unique(gt, return_counts=True)
    n = len(gt)
    k = len(classes)
    mn = int(counts.min())
    mx = int(counts.max())
    imb = round(mx / max(mn, 1), 3)
    avg = n / k
    thr = max(5, int(0.1 * avg))
    small_ratio = round(float((counts < thr).sum() / k), 3)
    probs = counts / counts.sum()
    raw_h = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-15))))
    return {'n_classes': k, 'class_size_min': mn, 'class_size_max': mx, 'imbalance_ratio': imb, 'small_class_ratio': small_ratio, 'class_entropy': round(raw_h, 4)}

def _separability(X_sub: np.ndarray, gt_sub: np.ndarray) -> dict:
    try:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    except ImportError:
        return {'separability_error': 'sklearn_not_installed'}
    k = len(np.unique(gt_sub))
    if k < 2 or X_sub.shape[0] <= k:
        return {'separability_error': 'insufficient_data'}
    res: dict = {}
    try:
        sil = float(silhouette_score(X_sub, gt_sub, sample_size=min(2000, X_sub.shape[0]), random_state=0))
        res['silhouette_score'] = round(sil, 4)
    except Exception:
        res['silhouette_score'] = None
    try:
        res['calinski_harabasz_score'] = round(float(calinski_harabasz_score(X_sub, gt_sub)), 2)
    except Exception:
        res['calinski_harabasz_score'] = None
    try:
        res['davies_bouldin_score'] = round(float(davies_bouldin_score(X_sub, gt_sub)), 4)
    except Exception:
        res['davies_bouldin_score'] = None
    return res

def _overlap(X_sub: np.ndarray, gt_sub: np.ndarray) -> dict:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return {'overlap_error': 'sklearn_not_installed'}
    n = X_sub.shape[0]
    k_uniq = len(np.unique(gt_sub))
    if k_uniq < 2:
        return {'overlap_error': 'single_class'}
    try:
        n_neighbors = min(n - 1, 60)
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1)
        nn.fit(X_sub)
        dists, indices = nn.kneighbors(X_sub)
        friend_d = np.full(n, np.inf)
        enemy_d = np.full(n, np.inf)
        for i in range(n):
            for rank in range(n_neighbors):
                j = indices[i, rank]
                d = dists[i, rank]
                if gt_sub[j] == gt_sub[i] and friend_d[i] == np.inf:
                    friend_d[i] = d
                elif gt_sub[j] != gt_sub[i] and enemy_d[i] == np.inf:
                    enemy_d[i] = d
                if friend_d[i] < np.inf and enemy_d[i] < np.inf:
                    break
        valid = (friend_d < np.inf) & (enemy_d < np.inf)
        if not valid.any():
            return {'overlap_error': 'no_valid_pairs'}
        f = friend_d[valid]
        e = enemy_d[valid]
        return {'nearest_friend_distance': _r(float(np.mean(f))), 'nearest_enemy_distance': _r(float(np.mean(e))), 'overlap_ratio': round(float(np.mean(e < f)), 4), 'margin_ratio': round(float(np.mean(e / np.maximum(f, 1e-12))), 4)}
    except Exception as exc:
        return {'overlap_error': str(exc)}

def _centroids(X_clean: np.ndarray, gt: np.ndarray) -> dict:
    classes = np.unique(gt)
    k = len(classes)
    if k < 2:
        return {'centroid_error': 'single_class'}
    try:
        C = np.vstack([X_clean[gt == c].mean(axis=0) for c in classes])
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        D = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(D, np.inf)
        sorted_D = np.sort(D, axis=1)
        nearest = sorted_D[:, 0]
        cd_min = float(nearest.min())
        cd_mean = float(np.mean(D[D < np.inf]))
        margin_mean = None
        if k >= 3:
            second = sorted_D[:, 1]
            margin_mean = round(float(np.mean(nearest / np.maximum(second, 1e-12))), 4)
        return {'centroid_distance_min': _r(cd_min), 'centroid_distance_mean': _r(cd_mean), 'centroid_margin_mean': margin_mean}
    except Exception as exc:
        return {'centroid_error': str(exc)}

def _shape(X_clean: np.ndarray, gt: np.ndarray) -> dict:
    classes = np.unique(gt)
    elongations: list[float] = []
    conditions: list[float] = []
    for c in classes:
        Xc = X_clean[gt == c]
        if Xc.shape[0] < 3 or Xc.shape[1] < 2:
            continue
        try:
            cov = np.cov(Xc.T)
            if cov.ndim < 2:
                continue
            eig = np.linalg.eigvalsh(cov)
            eig = np.sort(eig)[::-1]
            pos = eig[eig > 1e-10]
            if len(pos) < 2:
                continue
            elongations.append(float(pos[0] / pos[1]))
            conditions.append(float(pos[0] / pos[-1]))
        except Exception:
            continue
    if not elongations:
        return {'shape_error': 'insufficient_data'}
    return {'elongation_mean': round(float(np.mean(elongations)), 3), 'elongation_max': round(float(np.max(elongations)), 3), 'covariance_condition_mean': round(float(np.mean(conditions)), 3), 'covariance_condition_max': round(float(np.max(conditions)), 3)}

def compute_diagnostics(X: np.ndarray, gt: 'np.ndarray | None'=None, seed: int=0) -> dict:
    rng = _as_rng(seed)
    X_raw = np.asarray(X, dtype=np.float64)
    if X_raw.ndim != 2:
        raise ValueError('X must be 2-D')
    n, d = X_raw.shape
    result: dict = {}
    result.update(_basic(X_raw))
    X_clean = _clean_X(X_raw)
    if n > 5000:
        pca_idx = rng.choice(n, 5000, replace=False)
        X_pca = X_clean[pca_idx]
    else:
        X_pca = X_clean
    result.update(_pca(X_pca))
    dense_idx = _subsample_idx(n, rng)
    X_dense = X_clean[dense_idx]
    result.update(_density(X_dense))
    result.update(_hopkins(X_dense, rng))
    if gt is not None:
        gt_arr = np.asarray(gt).reshape(-1).astype(np.int64)
        if len(gt_arr) != n:
            raise ValueError(f'gt length {len(gt_arr)} != X rows {n}')
        result.update(_class_stats(gt_arr))
        sub_idx = _subsample_idx(n, rng)
        X_sub = X_clean[sub_idx]
        gt_sub = gt_arr[sub_idx]
        result.update(_separability(X_sub, gt_sub))
        result.update(_overlap(X_sub, gt_sub))
        result.update(_centroids(X_clean, gt_arr))
        result.update(_shape(X_clean, gt_arr))
    return result

def recommend_from_diagnostics(diag: dict) -> dict:
    n = diag.get('n_objects') or 100
    d = diag.get('n_features') or 2
    k = diag.get('n_classes')
    hopkins = diag.get('hopkins')
    overlap_ratio = diag.get('overlap_ratio')
    margin_ratio = diag.get('margin_ratio')
    silhouette = diag.get('silhouette_score')
    db_score = diag.get('davies_bouldin_score')
    ch_score = diag.get('calinski_harabasz_score')
    imbalance = diag.get('imbalance_ratio')
    density_var = diag.get('density_variation')
    eff_dim_90 = diag.get('effective_dimension_90')
    eff_dim_95 = diag.get('effective_dimension_95')
    part_ratio = diag.get('participation_ratio')
    exp_var_2d = diag.get('explained_variance_2d')
    elongation_max = diag.get('elongation_max')
    const_ratio = diag.get('constant_feature_ratio', 0.0)
    missing_ratio = diag.get('missing_ratio', 0.0)
    reasoning: list[str] = []
    if k is not None:
        k_min = max(2, k - 2)
        k_max = min(k + 4, 30)
        reasoning.append(f'k_min={k_min}, k_max={k_max}: число кластеров k={k} взято из gt; диапазон расширен на ±2–4 для устойчивости ансамбля.')
    elif eff_dim_90 is not None:
        k_min = 2
        k_max = min(max(5, eff_dim_90 // 2 + 2), 20)
        reasoning.append(f'k_min={k_min}, k_max={k_max}: gt отсутствует; k_max = eff_dim_90//2+2={k_max} (effective_dimension_90={eff_dim_90}).')
    else:
        k_min, k_max = (2, 10)
        reasoning.append('k_min=2, k_max=10: gt и PCA недоступны; используются значения по умолчанию.')
    if n < 300:
        m_base = 20
    elif n < 1000:
        m_base = 30
    elif n < 5000:
        m_base = 40
    else:
        m_base = 50
    m_extra = 0
    if hopkins is not None and hopkins < 0.6:
        m_extra += 10
        reasoning.append(f'm +10: Hopkins={hopkins:.3f} < 0.60 — слабая кластерная тенденция, нужно больше базовых кластеризаций.')
    if overlap_ratio is not None and overlap_ratio > 0.3:
        m_extra += 10
        reasoning.append(f'm +10: overlap_ratio={overlap_ratio:.3f} > 0.30 — высокое перекрытие классов.')
    if imbalance is not None and imbalance > 5.0:
        m_extra += 10
        reasoning.append(f'm +10: imbalance_ratio={imbalance:.2f} > 5 — сильный дисбаланс классов.')
    m = min(m_base + m_extra, 100)
    reasoning.append(f'm={m}: base={m_base} + поправки={m_extra}.')
    if n > 2500:
        strategy = 'kmeans'
        reasoning.append(f'strategy=kmeans: n={n} > 2500 — иерархические базовые кластеризации O(n²) слишком медленны.')
    else:
        strategy = 'mixed'
        reasoning.append(f'strategy=mixed: n={n} ≤ 2500 — сочетание k-means и иерархических.')
    preprocessing = 'zscore'
    if const_ratio > 0.1:
        reasoning.append(f'preprocessing=zscore: {const_ratio * 100:.1f}% константных признаков — zscore обнулит их безопасно.')
    elif missing_ratio > 0.05:
        reasoning.append('preprocessing=zscore: >5% пропущенных значений — zscore после импутации средним.')
    if elongation_max is not None and elongation_max > 8.0:
        method = 'single'
        reasoning.append(f'method=single: elongation_max={elongation_max:.1f} > 8 — вытянутые кластеры, single linkage лучше сохраняет цепочечную структуру.')
    elif density_var is not None and density_var > 0.6:
        method = 'average'
        reasoning.append(f'method=average: density_variation={density_var:.3f} > 0.6 — неоднородная плотность, average устойчив к выбросам.')
    else:
        method = 'average'
        reasoning.append('method=average: выбран как наиболее устойчивый по умолчанию.')
    if overlap_ratio is not None and overlap_ratio > 0.35:
        sharpen = 2.5
        reasoning.append(f'sharpen=2.5: overlap_ratio={overlap_ratio:.3f} > 0.35 — высокое перекрытие, нужен сильный контраст матрицы совместной встречаемости.')
    elif silhouette is not None and silhouette > 0.5:
        sharpen = 1.2
        reasoning.append(f'sharpen=1.2: silhouette={silhouette:.3f} > 0.50 — данные хорошо разделены, слабое заострение.')
    elif k is not None and k > 6:
        sharpen = 2.0
        reasoning.append(f'sharpen=2.0: n_classes={k} > 6 — много классов, нужен контраст.')
    else:
        sharpen = 1.5
        reasoning.append('sharpen=1.5: стандартное значение для умеренного заострения.')
    if n > 2500:
        diffusion_time = 5
        reasoning.append(f'diffusion_time=5: n={n} > 2500 — большой граф, нужно больше шагов диффузии.')
    elif overlap_ratio is not None and overlap_ratio > 0.3 or (density_var is not None and density_var > 0.6):
        diffusion_time = 4
        reasoning.append('diffusion_time=4: высокое перекрытие или неоднородная плотность — дополнительная диффузия сглаживает шум.')
    else:
        diffusion_time = 3
        reasoning.append('diffusion_time=3: стандартное значение.')
    if overlap_ratio is not None and overlap_ratio > 0.4:
        lambda_ = 0.05
        eta = 0.8
        reasoning.append(f'lambda_=0.05, eta=0.80: overlap_ratio={overlap_ratio:.3f} > 0.40 — консервативный режим: снижаем вес несходства, усиливаем сходство.')
    elif silhouette is not None and silhouette > 0.5:
        lambda_ = 0.12
        eta = 0.7
        reasoning.append(f'lambda_=0.12, eta=0.70: silhouette={silhouette:.3f} > 0.50 — хорошее разделение, можно взять более агрессивные параметры.')
    else:
        lambda_ = 0.09
        eta = 0.75
        reasoning.append('lambda_=0.09, eta=0.75: значения по умолчанию из оригинальной статьи SDGCA.')
    if imbalance is not None and imbalance > 5.0:
        theta = 0.55
        reasoning.append(f'theta=0.55: imbalance_ratio={imbalance:.2f} > 5 — смягчаем порог, чтобы малые классы не потерялись.')
    elif silhouette is not None and silhouette > 0.5:
        theta = 0.7
        reasoning.append(f'theta=0.70: silhouette={silhouette:.3f} > 0.50 — строгий порог при хорошем разделении.')
    else:
        theta = 0.65
        reasoning.append('theta=0.65: стандартное значение.')
    warnings_list: list[str] = []
    if (diag.get('dimensionality_ratio') or 0) > 0.5:
        warnings_list.append(f"Высокое соотношение d/n={diag['dimensionality_ratio']:.2f}: риск переобучения базовых кластеризаций; рекомендуется feature_subsample=True.")
    if (diag.get('constant_feature_ratio') or 0) > 0.2:
        warnings_list.append(f"Константных признаков: {diag['constant_feature_ratio'] * 100:.0f}% — они не несут информации и замедляют обучение.")
    if (diag.get('missing_ratio') or 0) > 0.1:
        warnings_list.append(f"Пропущенных значений: {diag['missing_ratio'] * 100:.1f}% — произведена замена средним, качество может снизиться.")
    if (diag.get('outlier_ratio') or 0) > 0.1:
        warnings_list.append(f"Выбросов (по 95-й перцентили kNN): {diag['outlier_ratio'] * 100:.1f}% — шум может ухудшить NMI/ARI, особенно для иерархических методов.")
    if exp_var_2d is not None and exp_var_2d < 0.5:
        warnings_list.append(f'2D-объяснённая дисперсия={exp_var_2d:.2f} < 0.50: визуализация в 2D будет искажённой; данные многомерны.')
    return {'m': m, 'k_min': k_min, 'k_max': k_max, 'strategy': strategy, 'preprocessing': preprocessing, 'warnings': warnings_list, 'reasoning': reasoning, 'per_algorithm': {'hierarchical_baseline': {'method': method}, 'hierarchical_weighted': {'method': method, 'sharpen': sharpen}, 'sdgca': {'method': method, 'lambda_': lambda_, 'eta': eta, 'theta': theta}, 'sdgca_modified': {'method': method, 'lambda_': lambda_, 'eta': eta, 'theta': theta, 'diffusion_time': diffusion_time}}}
