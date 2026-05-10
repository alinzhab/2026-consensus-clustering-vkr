"""Smart dataset-aware parameter selection and algorithm recommendation.

Pipeline:
  1. compute_diagnostics(X, gt)   -- ~30 structural metrics
  2. smart_recommend(diag)        -- nearest-neighbor PAPER_GRID lookup
                                     + rule-based algorithm ranking
  3. build_grok_prompt(diag, rec) -- structured prompt for Grok LLM
  4. Grok returns natural-language explanation shown in the UI

PAPER_GRID nearest-neighbor logic
----------------------------------
Distance between current dataset (n, K, d) and each reference dataset:
  dist = |log(n_ref) - log(n)| + 2*|K_ref - K|/K + 0.5*|log(d_ref) - log(d)|

K-distance is weighted 2x because the number of clusters is the most
important factor for choosing SDGCA parameters.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# PAPER_GRID: known-good (lambda, eta, theta) from original SDGCA paper
# ---------------------------------------------------------------------------

PAPER_GRID: list[dict] = [
    {"name": "Ecoli",       "n": 336,  "K": 8,  "d": 7,     "lam": 0.09, "eta": 0.65, "theta": 0.75},
    {"name": "GLIOMA",      "n": 50,   "K": 4,  "d": 4434,  "lam": 0.02, "eta": 0.80, "theta": 0.60},
    {"name": "Aggregation", "n": 788,  "K": 7,  "d": 2,     "lam": 0.08, "eta": 0.65, "theta": 0.70},
    {"name": "MF",          "n": 2000, "K": 10, "d": 649,   "lam": 0.05, "eta": 0.75, "theta": 0.95},
    {"name": "IS",          "n": 6435, "K": 7,  "d": 36,    "lam": 0.03, "eta": 0.90, "theta": 0.95},
    {"name": "MNIST",       "n": 2000, "K": 10, "d": 784,   "lam": 0.07, "eta": 0.95, "theta": 0.95},
    {"name": "Texture",     "n": 5500, "K": 11, "d": 40,    "lam": 0.04, "eta": 1.01, "theta": 1.01},
    {"name": "SPF",         "n": 4435, "K": 12, "d": 30,    "lam": 0.06, "eta": 0.70, "theta": 0.90},
    {"name": "ODR",         "n": 4000, "K": 10, "d": 27,    "lam": 0.06, "eta": 0.95, "theta": 0.95},
    {"name": "LS",          "n": 6435, "K": 6,  "d": 36,    "lam": 0.18, "eta": 0.70, "theta": 0.60},
    {"name": "ISOLET",      "n": 7797, "K": 26, "d": 617,   "lam": 0.06, "eta": 0.95, "theta": 0.95},
    {"name": "USPS",        "n": 9298, "K": 10, "d": 256,   "lam": 0.06, "eta": 0.95, "theta": 0.95},
    {"name": "orlraws10P",  "n": 100,  "K": 10, "d": 10304, "lam": 0.95, "eta": 1.01, "theta": 1.01},
    {"name": "BBC",         "n": 2225, "K": 5,  "d": 9635,  "lam": 0.06, "eta": 1.01, "theta": 1.01},
    {"name": "lung",        "n": 203,  "K": 5,  "d": 3312,  "lam": 0.75, "eta": 0.75, "theta": 0.60},
]


def _paper_grid_distance(ref: dict, n: int, K: int, d: int) -> float:
    """Log-scale distance between reference dataset and query (n, K, d)."""
    dn = abs(math.log1p(ref["n"]) - math.log1p(n))
    dk = abs(ref["K"] - K) / max(K, 1)
    dd = abs(math.log1p(ref["d"]) - math.log1p(d))
    return dn + 2.0 * dk + 0.5 * dd


def nearest_paper_params(n: int, K: int, d: int, top_k: int = 3) -> list[dict]:
    """Return top_k nearest PAPER_GRID entries (lam, eta, theta) sorted by distance."""
    ranked = sorted(PAPER_GRID, key=lambda r: _paper_grid_distance(r, n, K, d))
    return [
        {
            "reference_dataset": r["name"],
            "ref_n": r["n"], "ref_K": r["K"], "ref_d": r["d"],
            "distance": round(_paper_grid_distance(r, n, K, d), 4),
            "lam": r["lam"], "eta": r["eta"], "theta": r["theta"],
        }
        for r in ranked[:top_k]
    ]


def _blend_params(matches: list[dict]) -> tuple[float, float, float]:
    """Weighted average of top-3 paper params (closer = higher weight)."""
    if not matches:
        return 0.09, 0.75, 0.65
    eps = 1e-6
    weights = [1.0 / (m["distance"] + eps) for m in matches]
    total = sum(weights)
    lam   = sum(w * m["lam"]   for w, m in zip(weights, matches)) / total
    eta   = sum(w * m["eta"]   for w, m in zip(weights, matches)) / total
    theta = sum(w * m["theta"] for w, m in zip(weights, matches)) / total
    return round(lam, 4), round(eta, 4), round(theta, 4)


# ---------------------------------------------------------------------------
# Algorithm ranking
# ---------------------------------------------------------------------------

_ALGO_NAMES = {
    "hierarchical_baseline": "Иерархический (базовый)",
    "hierarchical_weighted": "Иерархический взвешенный",
    "sdgca":                 "SDGCA",
    "sdgca_modified":        "SDGCA модифицированный",
}


def _rank_algorithms(diag: dict) -> list[dict]:
    """Score each algorithm on this dataset profile and return ranked list."""
    n             = diag.get("n_objects") or 100
    d             = diag.get("n_features") or 2
    K             = diag.get("n_classes") or 3
    overlap       = diag.get("overlap_ratio")
    silhouette    = diag.get("silhouette_score")
    dim_ratio     = diag.get("dimensionality_ratio") or (d / max(n, 1))
    imbalance     = diag.get("imbalance_ratio") or 1.0
    elongation    = diag.get("elongation_max")
    density_var   = diag.get("density_variation")
    hopkins       = diag.get("hopkins")

    scores: dict[str, float] = {a: 0.0 for a in _ALGO_NAMES}
    reasons: dict[str, list[str]] = {a: [] for a in _ALGO_NAMES}

    # --- sdgca_modified is the strongest by default ---
    scores["sdgca_modified"] += 1.0
    reasons["sdgca_modified"].append("Наилучший по умолчанию — учитывает диффузию связей")

    # High-dimensional data (d >> n): SDGCA handles it via NWCA weighting
    if dim_ratio is not None and dim_ratio > 0.5:
        scores["sdgca"]          += 2.0
        scores["sdgca_modified"] += 2.0
        reasons["sdgca"].append(f"d/n={dim_ratio:.2f} > 0.5: SDGCA хорошо работает при высоких d")
        reasons["sdgca_modified"].append(f"d/n={dim_ratio:.2f} > 0.5: SDGCA+диффузия лучше при high-d")

    # Very high overlap: hierarchical_weighted with strong sharpen
    if overlap is not None and overlap > 0.40:
        scores["hierarchical_weighted"] += 2.5
        scores["sdgca_modified"]        += 1.5
        reasons["hierarchical_weighted"].append(
            f"overlap_ratio={overlap:.2f} > 0.40: взвешенный иерархический с усиленным sharpen"
        )
        reasons["sdgca_modified"].append(
            f"overlap_ratio={overlap:.2f} > 0.40: диффузия помогает при сильном перекрытии"
        )

    # Well-separated data: baseline or weighted works fine
    if silhouette is not None and silhouette > 0.55:
        scores["hierarchical_baseline"] += 1.5
        scores["hierarchical_weighted"] += 1.0
        reasons["hierarchical_baseline"].append(
            f"silhouette={silhouette:.2f} > 0.55: данные хорошо разделены, базовый иерархический достаточен"
        )

    # Many clusters: SDGCA weights help
    if K > 7:
        scores["sdgca"]          += 1.5
        scores["sdgca_modified"] += 1.5
        reasons["sdgca"].append(f"K={K} > 7: SDGCA лучше управляет многими кластерами через веса NWCA")
        reasons["sdgca_modified"].append(f"K={K} > 7: SDGCA+диффузия стабильнее при K > 7")

    # Elongated clusters: hierarchical with single linkage
    if elongation is not None and elongation > 8.0:
        scores["hierarchical_baseline"] += 2.0
        scores["hierarchical_weighted"] += 1.0
        reasons["hierarchical_baseline"].append(
            f"elongation_max={elongation:.1f} > 8: вытянутые кластеры — иерархический (single) лучше"
        )

    # Strong class imbalance
    if imbalance is not None and imbalance > 5.0:
        scores["sdgca_modified"] += 1.0
        reasons["sdgca_modified"].append(
            f"imbalance_ratio={imbalance:.1f} > 5: диффузия сглаживает дисбаланс классов"
        )

    # Weak clustering tendency: need more diversity → weighted
    if hopkins is not None and hopkins < 0.55:
        scores["hierarchical_weighted"] += 1.5
        reasons["hierarchical_weighted"].append(
            f"Hopkins={hopkins:.2f} < 0.55: слабая кластерная тенденция — нужен взвешенный ансамбль"
        )

    ranked = sorted(_ALGO_NAMES.keys(), key=lambda a: scores[a], reverse=True)
    return [
        {
            "algorithm":   a,
            "label":       _ALGO_NAMES[a],
            "score":       round(scores[a], 2),
            "reasons":     reasons[a],
        }
        for a in ranked
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def smart_recommend(diag: dict) -> dict:
    """Given compute_diagnostics() output, return full smart recommendation.

    Returns
    -------
    dict with keys:
        sdgca_params        : {"lam": ..., "eta": ..., "theta": ...}
        paper_matches       : list of 3 nearest reference datasets
        algorithm_ranking   : list of 4 algorithms sorted by fit score
        best_algorithm      : str (algorithm key)
        best_algorithm_label: str (human-readable)
        reasoning           : list of str (why these choices)
    """
    n = int(diag.get("n_objects") or diag.get("n_samples") or 100)
    d = int(diag.get("n_features") or 2)
    K = int(diag.get("n_classes") or 3)

    # --- SDGCA params from PAPER_GRID nearest-neighbor ---
    matches = nearest_paper_params(n, K, d, top_k=3)
    lam, eta, theta = _blend_params(matches)

    # Override with diagnostics rules if they give a strong signal
    overlap   = diag.get("overlap_ratio")
    silhouette = diag.get("silhouette_score")
    imbalance = diag.get("imbalance_ratio")
    dim_ratio = diag.get("dimensionality_ratio") or (d / max(n, 1))

    override_reasons: list[str] = []

    # High-d genomic data: lambda needs to be higher
    if dim_ratio > 2.0 and lam < 0.15:
        lam = max(lam, 0.30)
        override_reasons.append(
            f"lambda увеличен до {lam:.2f}: d/n={dim_ratio:.2f} > 2 (genomic high-d)"
        )

    # Strong overlap: lower eta (tighter must-link graph)
    if overlap is not None and overlap > 0.40 and eta > 0.90:
        eta = 0.80
        override_reasons.append(
            f"eta снижен до {eta:.2f}: overlap_ratio={overlap:.2f} — нужен более строгий must-link граф"
        )

    # Good separation: can afford higher theta (stricter constraints)
    if silhouette is not None and silhouette > 0.55 and theta < 0.75:
        theta = 0.75
        override_reasons.append(
            f"theta повышен до {theta:.2f}: silhouette={silhouette:.2f} — данные хорошо разделены"
        )

    # Imbalanced: lower theta to include minority class pairs
    if imbalance is not None and imbalance > 5.0 and theta > 0.65:
        theta = min(theta, 0.60)
        override_reasons.append(
            f"theta снижен до {theta:.2f}: imbalance_ratio={imbalance:.1f} — защита малых классов"
        )

    sdgca_params = {
        "lam":   round(lam, 4),
        "eta":   round(eta, 4),
        "theta": round(theta, 4),
    }

    # --- Algorithm ranking ---
    ranking = _rank_algorithms(diag)
    best_algo = ranking[0]["algorithm"]

    # --- Reasoning summary ---
    top_match = matches[0]
    reasoning = [
        f"Ближайший эталон из статьи: {top_match['reference_dataset']} "
        f"(n={top_match['ref_n']}, K={top_match['ref_K']}, d={top_match['ref_d']}, "
        f"расстояние={top_match['distance']:.3f})",
        f"SDGCA параметры: lambda={lam:.4f}, eta={eta:.4f}, theta={theta:.4f} "
        f"(взвешенное среднее по 3 ближайшим эталонам)",
    ] + override_reasons + [
        f"Рекомендуемый алгоритм: {_ALGO_NAMES[best_algo]} "
        f"(оценка={ranking[0]['score']:.2f})",
    ]

    return {
        "sdgca_params":         sdgca_params,
        "paper_matches":        matches,
        "algorithm_ranking":    ranking,
        "best_algorithm":       best_algo,
        "best_algorithm_label": _ALGO_NAMES[best_algo],
        "reasoning":            reasoning,
        "dataset_profile": {
            "n": n, "K": K, "d": d,
            "overlap_ratio":    overlap,
            "silhouette_score": silhouette,
            "imbalance_ratio":  imbalance,
            "dim_ratio":        round(dim_ratio, 4),
        },
    }


# ---------------------------------------------------------------------------
# Grok prompt builder
# ---------------------------------------------------------------------------

def build_grok_prompt(diag: dict, rec: dict) -> str:
    """Build a concise structured prompt for Grok to explain recommendations.

    The prompt is in Russian (matching the web UI language) and contains:
    - Key dataset metrics (n, d, K, silhouette, overlap, Hopkins, ...)
    - SDGCA parameter recommendation with source (PAPER_GRID + overrides)
    - Algorithm ranking
    - Request for natural-language explanation in 3 sections
    """
    n         = diag.get("n_objects") or diag.get("n_samples") or "?"
    d         = diag.get("n_features") or "?"
    K         = diag.get("n_classes") or "неизвестно"
    sil       = diag.get("silhouette_score")
    db        = diag.get("davies_bouldin_score")
    ch        = diag.get("calinski_harabasz_score")
    overlap   = diag.get("overlap_ratio")
    margin    = diag.get("margin_ratio")
    imbalance = diag.get("imbalance_ratio")
    hopkins   = diag.get("hopkins")
    eff_dim   = diag.get("effective_dimension_90")
    exp_var2d = diag.get("explained_variance_2d")
    out_ratio = diag.get("outlier_ratio")
    missing   = diag.get("missing_ratio")
    const_r   = diag.get("constant_feature_ratio")
    elong     = diag.get("elongation_max")
    dim_ratio = diag.get("dimensionality_ratio") or (
        round(int(d) / max(int(n), 1), 4) if str(d).isdigit() else "?"
    )

    def _f(v, digits=3):
        if v is None:
            return "н/д"
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return str(v)

    sp = rec["sdgca_params"]
    best_label = rec["best_algorithm_label"]
    top_match = rec["paper_matches"][0] if rec["paper_matches"] else {}

    ranking_lines = "\n".join(
        f"  {i+1}. {r['label']} (score={r['score']:.2f}): {'; '.join(r['reasons'][:2]) or 'по умолчанию'}"
        for i, r in enumerate(rec["algorithm_ranking"])
    )

    matches_lines = "\n".join(
        f"  - {m['reference_dataset']}: n={m['ref_n']}, K={m['ref_K']}, d={m['ref_d']} "
        f"-> lambda={m['lam']}, eta={m['eta']}, theta={m['theta']} (dist={m['distance']:.3f})"
        for m in rec["paper_matches"]
    )

    warnings_from_diag: list[str] = []
    if (diag.get("missing_ratio") or 0) > 0.05:
        warnings_from_diag.append(f"Пропущенные значения: {_f(missing, 1)}%")
    if (diag.get("constant_feature_ratio") or 0) > 0.1:
        warnings_from_diag.append(f"Константные признаки: {_f(const_r, 1)}%")
    if (diag.get("outlier_ratio") or 0) > 0.10:
        warnings_from_diag.append(f"Выбросы: {_f(out_ratio, 1)}%")

    warnings_str = "\n".join(f"  ! {w}" for w in warnings_from_diag) if warnings_from_diag else "  Нет критичных проблем"

    prompt = f"""Ты — эксперт по ансамблевой кластеризации биомедицинских данных.
Тебе нужно объяснить пользователю результаты анализа его датасета и дать конкретные рекомендации.
Отвечай на РУССКОМ языке. Будь точным и лаконичным. Не используй markdown-заголовки, пиши plain-текст.

=== МЕТРИКИ ДАТАСЕТА ===
Объектов (n):           {n}
Признаков (d):          {d}
Классов (K):            {K}
d/n (dim_ratio):        {_f(dim_ratio)}
Silhouette score:       {_f(sil)}
Davies-Bouldin:         {_f(db)}
Calinski-Harabasz:      {_f(ch, 1)}
Overlap ratio:          {_f(overlap)}
Margin ratio:           {_f(margin)}
Imbalance ratio:        {_f(imbalance)}
Hopkins tendency:       {_f(hopkins)}
Effective dim (90%):    {_f(eff_dim, 0)}
Explained var 2D:       {_f(exp_var2d)}
Elongation max:         {_f(elong)}

=== РЕКОМЕНДОВАННЫЕ ПАРАМЕТРЫ SDGCA ===
lambda (nwca_para):     {sp['lam']}
eta (must-link порог):  {sp['eta']}
theta (constraint порог): {sp['theta']}

Источник: взвешенное среднее по 3 ближайшим датасетам из оригинальной статьи SDGCA:
{matches_lines}

=== РАНЖИРОВАНИЕ АЛГОРИТМОВ ===
{ranking_lines}

ЛУЧШИЙ АЛГОРИТМ: {best_label}

=== ПРЕДУПРЕЖДЕНИЯ ===
{warnings_str}

=== ЗАДАНИЕ ===
Напиши ответ из ТРЁХ чётко разделённых абзацев:

АБЗАЦ 1 — ХАРАКТЕРИСТИКА ДАТАСЕТА (3-4 предложения):
Опиши, что из себя представляет этот датасет с точки зрения кластеризации.
Насколько хорошо разделены классы? Есть ли проблемы (перекрытие, выбросы, дисбаланс, высокая размерность)?

АБЗАЦ 2 — ПОЧЕМУ ИМЕННО ЭТИ ПАРАМЕТРЫ (3-4 предложения):
Объясни, почему для SDGCA выбраны lambda={sp['lam']}, eta={sp['eta']}, theta={sp['theta']}.
Откуда взяты значения (ближайший эталон: {top_match.get('reference_dataset', 'н/д')}).
Что каждый параметр делает на практике для этого конкретного датасета.

АБЗАЦ 3 — РЕКОМЕНДАЦИЯ ПО АЛГОРИТМУ (2-3 предложения):
Почему лучший алгоритм — {best_label}?
Дай практический совет: какой алгоритм запустить первым, на что обратить внимание в результатах.
"""
    return prompt.strip()
