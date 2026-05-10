"""Smart hyperparameter tuning for SDGCA.

Параметры SDGCA и их роль
--------------------------
λ  (nwca_para) — масштаб энтропийных весов в NECI.
    Малое λ → кластеры с высокой энтропией сильно подавляются.
    Большое λ → все кластеры получают почти одинаковые веса.
    Диапазон из статьи: [0.02, 0.95]. Типичное значение: 0.05–0.1.

η  (eta) — порог must-link графа (граф сходства для лапласиана).
    Если η > 1 — шаг ADMM пропускается, возвращается просто NWCA.
    Диапазон из статьи: [0.65, 1.01]. Типичное значение: 0.7–0.95.

θ  (theta) — порог включения ребра в матрицу must-link ограничений.
    Определяет, какие пары точек считаются «должны быть вместе».
    Диапазон из статьи: [0.6, 1.01]. Типичное значение: 0.65–0.95.

Сетка из оригинальной статьи (по датасетам)
--------------------------------------------
Dataset         λ       η       θ
Ecoli           0.09    0.65    0.75
GLIOMA          0.02    0.80    0.60
Aggregation     0.08    0.65    0.70
MF              0.05    0.75    0.95
IS              0.03    0.90    0.95
MNIST           0.07    0.95    0.95
Texture         0.04    1.00    1.00  ← η,θ>1 → ADMM bypass
SPF             0.06    0.70    0.90
ODR             0.06    0.95    0.95
LS              0.18    0.70    0.60
ISOLET          0.06    0.95    0.95
USPS            0.06    0.95    0.95
orlraws10P      0.95    1.01    1.01  ← ADMM bypass
BBC             0.06    1.01    1.01  ← ADMM bypass
lung            0.75    0.75    0.60

Стратегия поиска
-----------------
1. Coarse prior (warm-start) — стартовые точки из таблицы выше,
   выбранные по схожести датасета (n, K, d).
2. Bayesian optimisation (TPE, Optuna) — 80 trials по внутренним метрикам.
3. Внутренние метрики (unsupervised, т.к. gt недоступен на практике):
   - Silhouette score   (выше = лучше)
   - Davies-Bouldin     (ниже = лучше)
   - Ensemble agreement — средний NMI(consensus_label, base_member_k)
     (выше = лучше: консенсус должен согласовываться с базовыми членами)
4. Composite objective = silhouette_normalized
                       + agreement_normalized
                       - db_normalized
5. После поиска — финальная верификация по NMI/ARI с gt (если есть).

Использование
--------------
    from sdgca_tuner import tune_sdgca

    best = tune_sdgca(
        dataset_path="datasets/real/dermatology.npz",
        n_trials=80,       # количество испытаний Bayesian
        m=20,              # размер ансамбля на один прогон
        seed=42,
        verbose=True,
    )
    print(best)
    # {'nwca_para': 0.07, 'eta': 0.82, 'theta': 0.73, 'score': 0.641, ...}
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")


# ── Сетка из оригинальной статьи ──────────────────────────────────────────────
# Используется для warm-start: стартуем с нескольких точек, ближайших к
# характеристикам текущего датасета.

PAPER_GRID: list[dict] = [
    {"name": "Ecoli",       "n": 336,  "K": 8,  "d": 7,    "lam": 0.09, "eta": 0.65, "theta": 0.75},
    {"name": "GLIOMA",      "n": 50,   "K": 4,  "d": 4434, "lam": 0.02, "eta": 0.80, "theta": 0.60},
    {"name": "Aggregation", "n": 788,  "K": 7,  "d": 2,    "lam": 0.08, "eta": 0.65, "theta": 0.70},
    {"name": "MF",          "n": 2000, "K": 10, "d": 649,  "lam": 0.05, "eta": 0.75, "theta": 0.95},
    {"name": "IS",          "n": 6435, "K": 7,  "d": 36,   "lam": 0.03, "eta": 0.90, "theta": 0.95},
    {"name": "MNIST",       "n": 2000, "K": 10, "d": 784,  "lam": 0.07, "eta": 0.95, "theta": 0.95},
    {"name": "Texture",     "n": 5500, "K": 11, "d": 40,   "lam": 0.04, "eta": 1.01, "theta": 1.01},
    {"name": "SPF",         "n": 4435, "K": 12, "d": 30,   "lam": 0.06, "eta": 0.70, "theta": 0.90},
    {"name": "ODR",         "n": 4000, "K": 10, "d": 27,   "lam": 0.06, "eta": 0.95, "theta": 0.95},
    {"name": "LS",          "n": 6435, "K": 6,  "d": 36,   "lam": 0.18, "eta": 0.70, "theta": 0.60},
    {"name": "ISOLET",      "n": 7797, "K": 26, "d": 617,  "lam": 0.06, "eta": 0.95, "theta": 0.95},
    {"name": "USPS",        "n": 9298, "K": 10, "d": 256,  "lam": 0.06, "eta": 0.95, "theta": 0.95},
    {"name": "orlraws10P",  "n": 100,  "K": 10, "d": 10304,"lam": 0.95, "eta": 1.01, "theta": 1.01},
    {"name": "BBC",         "n": 2225, "K": 5,  "d": 9635, "lam": 0.06, "eta": 1.01, "theta": 1.01},
    {"name": "lung",        "n": 203,  "K": 5,  "d": 3312, "lam": 0.75, "eta": 0.75, "theta": 0.60},
]

# Полная сетка для grid-search (если нужен детерминированный перебор)
FULL_GRID = {
    "nwca_para": [0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.18,
                  0.30, 0.50, 0.75, 0.95],
    "eta":       [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
    "theta":     [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
}
# Итого: 13 × 9 × 9 = 1053 комбинации — для полного перебора много,
# Bayesian поиск перекрывает это за 60–100 испытаний.


# ── helpers ────────────────────────────────────────────────────────────────────

def _load(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Загрузить (members, gt, X) из .npz или .mat."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hierarchical_consensus import load_dataset_full
    members, gt, X = load_dataset_full(path)
    return members, gt, X


def _warm_start_points(n: int, K: int, d: int, top_k: int = 5) -> list[dict]:
    """Возвращает top_k точек из PAPER_GRID, ближайших по (n, K, d)."""
    def dist(row: dict) -> float:
        dn = abs(np.log1p(row["n"]) - np.log1p(n))
        dk = abs(row["K"] - K) / max(K, 1)
        dd = abs(np.log1p(row["d"]) - np.log1p(d))
        return dn + 2 * dk + 0.5 * dd   # K-близость важнее всего
    ranked = sorted(PAPER_GRID, key=dist)
    return [{"nwca_para": r["lam"], "eta": r["eta"], "theta": r["theta"]}
            for r in ranked[:top_k]]


def _compute_metrics(
    X: np.ndarray | None,
    labels: np.ndarray,
    base_cls: np.ndarray,
) -> dict[str, float]:
    """Вычислить внутренние метрики для одного прогона."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from metrics import compute_nmi

    n_unique = len(np.unique(labels))
    result: dict[str, float] = {}

    # 1. Silhouette и Davies-Bouldin требуют X
    if X is not None and n_unique > 1 and n_unique < len(labels):
        try:
            sil = float(silhouette_score(X, labels, sample_size=min(2000, len(labels))))
        except Exception:
            sil = -1.0
        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:
            db = 9.9
        result["silhouette"] = sil
        result["davies_bouldin"] = db
    else:
        result["silhouette"] = 0.0
        result["davies_bouldin"] = 5.0

    # 2. Ensemble agreement: mean NMI(consensus, base_member_j)
    nmis = []
    for j in range(base_cls.shape[1]):
        col = base_cls[:, j].astype(np.int64)
        if len(np.unique(col)) < 2:
            continue
        try:
            nmis.append(compute_nmi(labels, col))
        except Exception:
            pass
    result["ensemble_agreement"] = float(np.mean(nmis)) if nmis else 0.0

    return result


def _run_one(
    members: np.ndarray,
    gt: np.ndarray,
    X: np.ndarray | None,
    nwca_para: float,
    eta: float,
    theta: float,
    m: int,
    seed: int,
    method: str,
) -> dict[str, float]:
    """Один прогон SDGCA с заданными параметрами. Возвращает метрики."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sdgca import (get_all_segs, compute_nwca, compute_neci,
                       compute_d, compute_s, compute_w, optimize_sdgca)
    from hierarchical_consensus import get_cls_result
    from metrics import compute_nmi, compute_ari

    rng = np.random.default_rng(seed)
    pool_size = members.shape[1]
    m_actual = min(m, pool_size)
    idx = rng.choice(pool_size, m_actual, replace=False)
    base_cls = members[:, idx]

    bcs, base_cls_segs = get_all_segs(base_cls)
    ca = base_cls_segs.T @ base_cls_segs / m_actual
    nwca = compute_nwca(base_cls_segs, compute_neci(bcs, base_cls_segs, nwca_para), m_actual)

    if eta > 1.0:
        consensus_matrix = nwca
    else:
        hc = ca.copy(); hc[hc < eta] = 0.0
        l_matrix = np.diag(np.sum(hc, axis=1)) - hc
        mla = ca.copy(); mla[mla < theta] = 0.0
        ml = compute_s(nwca, mla)
        cl = compute_d(bcs, base_cls_segs)
        ml[cl > 0] = 0.0
        s, d = optimize_sdgca(l_matrix, ml, cl)
        consensus_matrix = compute_w(s, d, nwca)

    k = int(np.unique(gt).size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = get_cls_result(consensus_matrix, k, method=method)

    int_metrics = _compute_metrics(X, labels, base_cls)
    nmi = float(compute_nmi(labels, gt))
    ari = float(compute_ari(labels, gt))
    return {**int_metrics, "nmi": nmi, "ari": ari, "labels": labels}


def _objective(
    members: np.ndarray,
    gt: np.ndarray,
    X: np.ndarray | None,
    nwca_para: float,
    eta: float,
    theta: float,
    m: int,
    seed: int,
    method: str,
    n_avg: int,
) -> float:
    """Целевая функция для Optuna (unsupervised — без gt).

    Composite score = 0.4 * silhouette_norm
                    + 0.4 * agreement
                    - 0.2 * db_norm

    silhouette_norm ∈ [0,1]: (sil + 1) / 2
    db_norm ∈ [0,1]:  1 / (1 + db)
    agreement ∈ [0,1]: уже нормировано (это NMI)
    """
    scores: list[float] = []
    for run in range(n_avg):
        try:
            r = _run_one(members, gt, X, nwca_para, eta, theta,
                         m, seed + run * 97, method)
            sil_n = (r["silhouette"] + 1) / 2          # [-1,1] → [0,1]
            db_n  = 1.0 / (1.0 + r["davies_bouldin"])  # [0,∞) → (0,1]
            agr   = r["ensemble_agreement"]             # [0,1]
            composite = 0.40 * sil_n + 0.40 * agr + 0.20 * db_n
            scores.append(composite)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


# ── Main API ───────────────────────────────────────────────────────────────────

def tune_sdgca(
    dataset_path: str | Path,
    n_trials: int = 80,
    m: int = 20,
    seed: int = 42,
    method: str = "ward",
    n_avg: int = 2,
    verbose: bool = True,
    use_gt_for_validation: bool = True,
) -> dict[str, Any]:
    """Подобрать λ, η, θ для SDGCA на заданном датасете.

    Args:
        dataset_path: путь к .npz или .mat файлу.
        n_trials:     число испытаний Bayesian поиска.
        m:            размер ансамбля за одно испытание.
        seed:         random seed.
        method:       иерархическая агрегация ('ward', 'average', ...).
        n_avg:        число прогонов за одно испытание (усреднение шума).
        verbose:      печатать прогресс.
        use_gt_for_validation: если True — после поиска дополнительно
                               вычислить NMI/ARI по ground truth.

    Returns:
        dict с ключами: nwca_para, eta, theta, score (composite),
        + nmi, ari (если use_gt_for_validation=True).
    """
    import optuna
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )

    members, gt, X = _load(dataset_path)
    n, pool = members.shape
    K = int(np.unique(gt).size)
    d = int(X.shape[1]) if X is not None else 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"SDGCA Hyperparameter Tuning")
        print(f"Dataset : {Path(dataset_path).stem}")
        print(f"n={n}  K={K}  d={d}  pool={pool}")
        print(f"Trials  : {n_trials}  (Bayesian TPE + warm-start)")
        print(f"{'='*60}")
        print()
        print("Parameter grid:")
        print("  lam (nwca_para) in [0.02, 0.95]  -- entropy weight scale")
        print("  eta             in [0.60, 1.05]  -- must-link graph threshold")
        print("  theta           in [0.60, 1.05]  -- must-link constraint threshold")
        print("  (eta > 1 or theta > 1 -> ADMM bypassed, pure NWCA)")
        print()

    # Warm-start из статьи
    warm_pts = _warm_start_points(n, K, d, top_k=5)
    if verbose:
        print(f"Warm-start points (nearest paper datasets):")
        for i, pt in enumerate(warm_pts):
            print(f"  [{i+1}] lam={pt['nwca_para']:.3f}  eta={pt['eta']:.3f}  theta={pt['theta']:.3f}")
        print()

    def optuna_objective(trial: optuna.Trial) -> float:
        lam   = trial.suggest_float("nwca_para", 0.02, 0.95, log=True)
        eta   = trial.suggest_float("eta",       0.60, 1.05)
        theta = trial.suggest_float("theta",     0.60, 1.05)
        return _objective(members, gt, X, lam, eta, theta,
                          m, seed, method, n_avg)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=max(10, len(warm_pts)),
        multivariate=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    # Добавляем warm-start точки
    for pt in warm_pts:
        study.enqueue_trial(pt)

    # Несколько anchor-точек из крайних значений
    anchors = [
        {"nwca_para": 0.06, "eta": 0.75, "theta": 0.75},  # "универсальный"
        {"nwca_para": 0.06, "eta": 1.01, "theta": 1.01},  # ADMM bypass
        {"nwca_para": 0.15, "eta": 0.70, "theta": 0.65},  # высокая λ
    ]
    for a in anchors:
        study.enqueue_trial(a)

    study.optimize(optuna_objective, n_trials=n_trials,
                   show_progress_bar=verbose)

    best = study.best_params
    best_score = study.best_value

    if verbose:
        print()
        print("Best parameters found:")
        print(f"  lam (nwca_para) = {best['nwca_para']:.4f}")
        print(f"  eta             = {best['eta']:.4f}")
        print(f"  theta           = {best['theta']:.4f}")
        print(f"  Composite score = {best_score:.4f}")

    result: dict[str, Any] = {
        "nwca_para": best["nwca_para"],
        "eta":       best["eta"],
        "theta":     best["theta"],
        "score":     best_score,
        "n_trials":  n_trials,
        "dataset":   Path(dataset_path).stem,
    }

    # Финальная верификация по NMI/ARI (с gt)
    if use_gt_for_validation:
        if verbose:
            print("\nFinal validation (averaged over 5 runs):")
        nmi_list, ari_list = [], []
        for run in range(5):
            try:
                r = _run_one(members, gt, X,
                             best["nwca_para"], best["eta"], best["theta"],
                             m, seed + run * 13, method)
                nmi_list.append(r["nmi"])
                ari_list.append(r["ari"])
            except Exception:
                pass
        if nmi_list:
            result["nmi_mean"] = float(np.mean(nmi_list))
            result["nmi_std"]  = float(np.std(nmi_list))
            result["ari_mean"] = float(np.mean(ari_list))
            result["ari_std"]  = float(np.std(ari_list))
            if verbose:
                print(f"  NMI = {result['nmi_mean']:.4f} +/- {result['nmi_std']:.4f}")
                print(f"  ARI = {result['ari_mean']:.4f} +/- {result['ari_std']:.4f}")

    return result


def tune_and_compare(
    dataset_path: str | Path,
    n_trials: int = 80,
    m: int = 20,
    seed: int = 42,
    method: str = "ward",
    verbose: bool = True,
) -> dict[str, Any]:
    """Тюнинг + сравнение с дефолтными параметрами.

    Показывает: стало ли лучше после подбора параметров?
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sdgca import run_sdgca
    from metrics import compute_nmi, compute_ari

    print("Running SDGCA with DEFAULT parameters (lam=0.09, eta=0.75, theta=0.65)...")
    default = run_sdgca(str(dataset_path), m=m, cnt_times=5, method=method, seed=seed)
    print(f"  Default NMI = {default['nmi_mean']:.4f} +/- {default['nmi_std']:.4f}")
    print()

    best = tune_sdgca(dataset_path, n_trials=n_trials, m=m, seed=seed,
                      method=method, verbose=verbose)

    print()
    print("="*60)
    print("COMPARISON: Default vs Tuned")
    print("="*60)
    if "nmi_mean" in best:
        delta_nmi = best["nmi_mean"] - default["nmi_mean"]
        delta_ari = best["ari_mean"] - default["ari_mean"]
        print(f"{'Metric':<10} {'Default':>10} {'Tuned':>10} {'Delta':>10}")
        print("-"*44)
        print(f"{'NMI':<10} {default['nmi_mean']:>10.4f} {best['nmi_mean']:>10.4f} {delta_nmi:>+10.4f}")
        print(f"{'ARI':<10} {default['ari_mean']:>10.4f} {best['ari_mean']:>10.4f} {delta_ari:>+10.4f}")
        verdict = "IMPROVED" if delta_nmi > 0.005 else ("SAME" if abs(delta_nmi) <= 0.005 else "WORSE")
        print(f"\nVerdict: {verdict}")

    best["default_nmi"] = default["nmi_mean"]
    best["default_ari"] = default["ari_mean"]
    return best


def grid_search_sdgca(
    dataset_path: str | Path,
    m: int = 20,
    seed: int = 42,
    method: str = "ward",
    coarse: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Детерминированный grid-search по сетке из статьи.

    Параметры сетки
    ----------------
    Grubый режим (coarse=True):  5 × 5 × 5 = 125 комбинаций
    Полный режим (coarse=False): 13 × 9 × 9 = 1053 комбинации

    Grubая сетка (coarse)
    ----------------------
    λ:  0.03, 0.06, 0.09, 0.18, 0.75
    η:  0.65, 0.75, 0.85, 0.95, 1.01
    θ:  0.60, 0.70, 0.80, 0.90, 1.01
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    members, gt, X = _load(dataset_path)

    if coarse:
        lam_grid   = [0.03, 0.06, 0.09, 0.18, 0.75]
        eta_grid   = [0.65, 0.75, 0.85, 0.95, 1.01]
        theta_grid = [0.60, 0.70, 0.80, 0.90, 1.01]
    else:
        lam_grid   = FULL_GRID["nwca_para"]
        eta_grid   = FULL_GRID["eta"]
        theta_grid = FULL_GRID["theta"]

    total = len(lam_grid) * len(eta_grid) * len(theta_grid)
    if verbose:
        print(f"Grid search: {len(lam_grid)} lam x {len(eta_grid)} eta x {len(theta_grid)} theta = {total} combos")

    best_score = -np.inf
    best_params: dict = {}
    all_results: list[dict] = []
    done = 0

    for lam in lam_grid:
        for eta in eta_grid:
            for theta in theta_grid:
                score = _objective(members, gt, X, lam, eta, theta,
                                   m, seed, method, n_avg=1)
                rec = {"nwca_para": lam, "eta": eta, "theta": theta, "score": score}
                all_results.append(rec)
                if score > best_score:
                    best_score = score
                    best_params = dict(rec)
                done += 1
                if verbose and done % 25 == 0:
                    print(f"  [{done}/{total}] best so far: "
                          f"lam={best_params['nwca_para']:.3f} "
                          f"eta={best_params['eta']:.3f} "
                          f"theta={best_params['theta']:.3f} "
                          f"score={best_score:.4f}")

    if verbose:
        print(f"\nBest: lam={best_params['nwca_para']:.3f} "
              f"eta={best_params['eta']:.3f} "
              f"theta={best_params['theta']:.3f} "
              f"score={best_score:.4f}")

    return {"best": best_params, "all": all_results}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys
    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(
        description="Smart hyperparameter tuning for SDGCA"
    )
    parser.add_argument("dataset", help="Path to .npz or .mat dataset")
    parser.add_argument("--trials",  type=int,   default=80,      help="Bayesian trials (default 80)")
    parser.add_argument("--m",       type=int,   default=20,      help="Ensemble size (default 20)")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--method",  default="ward")
    parser.add_argument("--mode",    choices=["bayesian","grid","compare"],
                        default="bayesian",
                        help="bayesian=TPE (default), grid=det. grid, compare=default vs tuned")
    parser.add_argument("--coarse",  action="store_true",
                        help="Coarse grid (5×5×5) instead of full 13×9×9")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.mode == "bayesian":
        result = tune_sdgca(
            args.dataset, n_trials=args.trials, m=args.m,
            seed=args.seed, method=args.method, verbose=verbose,
        )
        print("\nResult JSON:")
        print(json.dumps({k: v for k, v in result.items() if k != "labels"},
                         indent=2))

    elif args.mode == "grid":
        result = grid_search_sdgca(
            args.dataset, m=args.m, seed=args.seed,
            method=args.method, coarse=args.coarse, verbose=verbose,
        )
        print("\nBest JSON:")
        print(json.dumps(result["best"], indent=2))

    elif args.mode == "compare":
        result = tune_and_compare(
            args.dataset, n_trials=args.trials, m=args.m,
            seed=args.seed, method=args.method, verbose=verbose,
        )
        print("\nResult JSON:")
        print(json.dumps({k: v for k, v in result.items() if k != "labels"},
                         indent=2))
