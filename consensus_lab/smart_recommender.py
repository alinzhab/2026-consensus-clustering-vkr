from __future__ import annotations
import math
from typing import Any
PAPER_GRID: list[dict] = [{'name': 'Ecoli', 'n': 336, 'K': 8, 'd': 7, 'lam': 0.09, 'eta': 0.65, 'theta': 0.75}, {'name': 'GLIOMA', 'n': 50, 'K': 4, 'd': 4434, 'lam': 0.02, 'eta': 0.8, 'theta': 0.6}, {'name': 'Aggregation', 'n': 788, 'K': 7, 'd': 2, 'lam': 0.08, 'eta': 0.65, 'theta': 0.7}, {'name': 'MF', 'n': 2000, 'K': 10, 'd': 649, 'lam': 0.05, 'eta': 0.75, 'theta': 0.95}, {'name': 'IS', 'n': 6435, 'K': 7, 'd': 36, 'lam': 0.03, 'eta': 0.9, 'theta': 0.95}, {'name': 'MNIST', 'n': 2000, 'K': 10, 'd': 784, 'lam': 0.07, 'eta': 0.95, 'theta': 0.95}, {'name': 'Texture', 'n': 5500, 'K': 11, 'd': 40, 'lam': 0.04, 'eta': 1.01, 'theta': 1.01}, {'name': 'SPF', 'n': 4435, 'K': 12, 'd': 30, 'lam': 0.06, 'eta': 0.7, 'theta': 0.9}, {'name': 'ODR', 'n': 4000, 'K': 10, 'd': 27, 'lam': 0.06, 'eta': 0.95, 'theta': 0.95}, {'name': 'LS', 'n': 6435, 'K': 6, 'd': 36, 'lam': 0.18, 'eta': 0.7, 'theta': 0.6}, {'name': 'ISOLET', 'n': 7797, 'K': 26, 'd': 617, 'lam': 0.06, 'eta': 0.95, 'theta': 0.95}, {'name': 'USPS', 'n': 9298, 'K': 10, 'd': 256, 'lam': 0.06, 'eta': 0.95, 'theta': 0.95}, {'name': 'orlraws10P', 'n': 100, 'K': 10, 'd': 10304, 'lam': 0.95, 'eta': 1.01, 'theta': 1.01}, {'name': 'BBC', 'n': 2225, 'K': 5, 'd': 9635, 'lam': 0.06, 'eta': 1.01, 'theta': 1.01}, {'name': 'lung', 'n': 203, 'K': 5, 'd': 3312, 'lam': 0.75, 'eta': 0.75, 'theta': 0.6}]

def _paper_grid_distance(ref: dict, n: int, K: int, d: int) -> float:
    dn = abs(math.log1p(ref['n']) - math.log1p(n))
    dk = abs(ref['K'] - K) / max(K, 1)
    dd = abs(math.log1p(ref['d']) - math.log1p(d))
    return dn + 2.0 * dk + 0.5 * dd

def nearest_paper_params(n: int, K: int, d: int, top_k: int=3) -> list[dict]:
    ranked = sorted(PAPER_GRID, key=lambda r: _paper_grid_distance(r, n, K, d))
    return [{'reference_dataset': r['name'], 'ref_n': r['n'], 'ref_K': r['K'], 'ref_d': r['d'], 'distance': round(_paper_grid_distance(r, n, K, d), 4), 'lam': r['lam'], 'eta': r['eta'], 'theta': r['theta']} for r in ranked[:top_k]]

def _blend_params(matches: list[dict]) -> tuple[float, float, float]:
    if not matches:
        return (0.09, 0.75, 0.65)
    eps = 1e-06
    weights = [1.0 / (m['distance'] + eps) for m in matches]
    total = sum(weights)
    lam = sum((w * m['lam'] for w, m in zip(weights, matches))) / total
    eta = sum((w * m['eta'] for w, m in zip(weights, matches))) / total
    theta = sum((w * m['theta'] for w, m in zip(weights, matches))) / total
    return (round(lam, 4), round(eta, 4), round(theta, 4))
_ALGO_NAMES = {'hierarchical_baseline': 'Иерархический (базовый)', 'hierarchical_weighted': 'Иерархический взвешенный', 'sdgca': 'SDGCA', 'sdgca_modified': 'SDGCA модифицированный'}

def _rank_algorithms(diag: dict) -> list[dict]:
    n = diag.get('n_objects') or 100
    d = diag.get('n_features') or 2
    K = diag.get('n_classes') or 3
    overlap = diag.get('overlap_ratio')
    silhouette = diag.get('silhouette_score')
    dim_ratio = diag.get('dimensionality_ratio') or d / max(n, 1)
    imbalance = diag.get('imbalance_ratio') or 1.0
    elongation = diag.get('elongation_max')
    density_var = diag.get('density_variation')
    hopkins = diag.get('hopkins')
    scores: dict[str, float] = {a: 0.0 for a in _ALGO_NAMES}
    reasons: dict[str, list[str]] = {a: [] for a in _ALGO_NAMES}
    scores['sdgca_modified'] += 1.0
    reasons['sdgca_modified'].append('Наилучший по умолчанию — учитывает диффузию связей')
    if dim_ratio is not None and dim_ratio > 0.5:
        scores['sdgca'] += 2.0
        scores['sdgca_modified'] += 2.0
        reasons['sdgca'].append(f'd/n={dim_ratio:.2f} > 0.5: SDGCA хорошо работает при высоких d')
        reasons['sdgca_modified'].append(f'd/n={dim_ratio:.2f} > 0.5: SDGCA+диффузия лучше при high-d')
    if overlap is not None and overlap > 0.4:
        scores['hierarchical_weighted'] += 2.5
        scores['sdgca_modified'] += 1.5
        reasons['hierarchical_weighted'].append(f'overlap_ratio={overlap:.2f} > 0.40: взвешенный иерархический с усиленным sharpen')
        reasons['sdgca_modified'].append(f'overlap_ratio={overlap:.2f} > 0.40: диффузия помогает при сильном перекрытии')
    if silhouette is not None and silhouette > 0.55:
        scores['hierarchical_baseline'] += 1.5
        scores['hierarchical_weighted'] += 1.0
        reasons['hierarchical_baseline'].append(f'silhouette={silhouette:.2f} > 0.55: данные хорошо разделены, базовый иерархический достаточен')
    if K > 7:
        scores['sdgca'] += 1.5
        scores['sdgca_modified'] += 1.5
        reasons['sdgca'].append(f'K={K} > 7: SDGCA лучше управляет многими кластерами через веса NWCA')
        reasons['sdgca_modified'].append(f'K={K} > 7: SDGCA+диффузия стабильнее при K > 7')
    if elongation is not None and elongation > 8.0:
        scores['hierarchical_baseline'] += 2.0
        scores['hierarchical_weighted'] += 1.0
        reasons['hierarchical_baseline'].append(f'elongation_max={elongation:.1f} > 8: вытянутые кластеры — иерархический (single) лучше')
    if imbalance is not None and imbalance > 5.0:
        scores['sdgca_modified'] += 1.0
        reasons['sdgca_modified'].append(f'imbalance_ratio={imbalance:.1f} > 5: диффузия сглаживает дисбаланс классов')
    if hopkins is not None and hopkins < 0.55:
        scores['hierarchical_weighted'] += 1.5
        reasons['hierarchical_weighted'].append(f'Hopkins={hopkins:.2f} < 0.55: слабая кластерная тенденция — нужен взвешенный ансамбль')
    ranked = sorted(_ALGO_NAMES.keys(), key=lambda a: scores[a], reverse=True)
    return [{'algorithm': a, 'label': _ALGO_NAMES[a], 'score': round(scores[a], 2), 'reasons': reasons[a]} for a in ranked]

def smart_recommend(diag: dict) -> dict:
    n = int(diag.get('n_objects') or diag.get('n_samples') or 100)
    d = int(diag.get('n_features') or 2)
    K = int(diag.get('n_classes') or 3)
    matches = nearest_paper_params(n, K, d, top_k=3)
    lam, eta, theta = _blend_params(matches)
    overlap = diag.get('overlap_ratio')
    silhouette = diag.get('silhouette_score')
    imbalance = diag.get('imbalance_ratio')
    dim_ratio = diag.get('dimensionality_ratio') or d / max(n, 1)
    override_reasons: list[str] = []
    if dim_ratio > 2.0 and lam < 0.15:
        lam = max(lam, 0.3)
        override_reasons.append(f'lambda увеличен до {lam:.2f}: d/n={dim_ratio:.2f} > 2 (genomic high-d)')
    if overlap is not None and overlap > 0.4 and (eta > 0.9):
        eta = 0.8
        override_reasons.append(f'eta снижен до {eta:.2f}: overlap_ratio={overlap:.2f} — нужен более строгий must-link граф')
    if silhouette is not None and silhouette > 0.55 and (theta < 0.75):
        theta = 0.75
        override_reasons.append(f'theta повышен до {theta:.2f}: silhouette={silhouette:.2f} — данные хорошо разделены')
    if imbalance is not None and imbalance > 5.0 and (theta > 0.65):
        theta = min(theta, 0.6)
        override_reasons.append(f'theta снижен до {theta:.2f}: imbalance_ratio={imbalance:.1f} — защита малых классов')
    sdgca_params = {'lam': round(lam, 4), 'eta': round(eta, 4), 'theta': round(theta, 4)}
    ranking = _rank_algorithms(diag)
    best_algo = ranking[0]['algorithm']
    top_match = matches[0]
    reasoning = [f"Ближайший эталон из статьи: {top_match['reference_dataset']} (n={top_match['ref_n']}, K={top_match['ref_K']}, d={top_match['ref_d']}, расстояние={top_match['distance']:.3f})", f'SDGCA параметры: lambda={lam:.4f}, eta={eta:.4f}, theta={theta:.4f} (взвешенное среднее по 3 ближайшим эталонам)'] + override_reasons + [f"Рекомендуемый алгоритм: {_ALGO_NAMES[best_algo]} (оценка={ranking[0]['score']:.2f})"]
    return {'sdgca_params': sdgca_params, 'paper_matches': matches, 'algorithm_ranking': ranking, 'best_algorithm': best_algo, 'best_algorithm_label': _ALGO_NAMES[best_algo], 'reasoning': reasoning, 'dataset_profile': {'n': n, 'K': K, 'd': d, 'overlap_ratio': overlap, 'silhouette_score': silhouette, 'imbalance_ratio': imbalance, 'dim_ratio': round(dim_ratio, 4)}}

def build_grok_prompt(diag: dict, rec: dict) -> str:
    n = diag.get('n_objects') or diag.get('n_samples') or '?'
    d = diag.get('n_features') or '?'
    K = diag.get('n_classes') or 'неизвестно'
    sil = diag.get('silhouette_score')
    db = diag.get('davies_bouldin_score')
    ch = diag.get('calinski_harabasz_score')
    overlap = diag.get('overlap_ratio')
    margin = diag.get('margin_ratio')
    imbalance = diag.get('imbalance_ratio')
    hopkins = diag.get('hopkins')
    eff_dim = diag.get('effective_dimension_90')
    exp_var2d = diag.get('explained_variance_2d')
    out_ratio = diag.get('outlier_ratio')
    missing = diag.get('missing_ratio')
    const_r = diag.get('constant_feature_ratio')
    elong = diag.get('elongation_max')
    dim_ratio = diag.get('dimensionality_ratio') or (round(int(d) / max(int(n), 1), 4) if str(d).isdigit() else '?')

    def _f(v, digits=3):
        if v is None:
            return 'н/д'
        try:
            return f'{float(v):.{digits}f}'
        except Exception:
            return str(v)
    sp = rec['sdgca_params']
    best_label = rec['best_algorithm_label']
    top_match = rec['paper_matches'][0] if rec['paper_matches'] else {}
    ranking_lines = '\n'.join((f"  {i + 1}. {r['label']} (score={r['score']:.2f}): {'; '.join(r['reasons'][:2]) or 'по умолчанию'}" for i, r in enumerate(rec['algorithm_ranking'])))
    matches_lines = '\n'.join((f"  - {m['reference_dataset']}: n={m['ref_n']}, K={m['ref_K']}, d={m['ref_d']} -> lambda={m['lam']}, eta={m['eta']}, theta={m['theta']} (dist={m['distance']:.3f})" for m in rec['paper_matches']))
    warnings_from_diag: list[str] = []
    if (diag.get('missing_ratio') or 0) > 0.05:
        warnings_from_diag.append(f'Пропущенные значения: {_f(missing, 1)}%')
    if (diag.get('constant_feature_ratio') or 0) > 0.1:
        warnings_from_diag.append(f'Константные признаки: {_f(const_r, 1)}%')
    if (diag.get('outlier_ratio') or 0) > 0.1:
        warnings_from_diag.append(f'Выбросы: {_f(out_ratio, 1)}%')
    warnings_str = '\n'.join((f'  ! {w}' for w in warnings_from_diag)) if warnings_from_diag else '  Нет критичных проблем'
    prompt = f"Ты — эксперт по ансамблевой кластеризации биомедицинских данных.\nТебе нужно объяснить пользователю результаты анализа его датасета и дать конкретные рекомендации.\nОтвечай на РУССКОМ языке. Будь точным и лаконичным. Не используй markdown-заголовки, пиши plain-текст.\n\n=== МЕТРИКИ ДАТАСЕТА ===\nОбъектов (n):           {n}\nПризнаков (d):          {d}\nКлассов (K):            {K}\nd/n (dim_ratio):        {_f(dim_ratio)}\nSilhouette score:       {_f(sil)}\nDavies-Bouldin:         {_f(db)}\nCalinski-Harabasz:      {_f(ch, 1)}\nOverlap ratio:          {_f(overlap)}\nMargin ratio:           {_f(margin)}\nImbalance ratio:        {_f(imbalance)}\nHopkins tendency:       {_f(hopkins)}\nEffective dim (90%):    {_f(eff_dim, 0)}\nExplained var 2D:       {_f(exp_var2d)}\nElongation max:         {_f(elong)}\n\n=== РЕКОМЕНДОВАННЫЕ ПАРАМЕТРЫ SDGCA ===\nlambda (nwca_para):     {sp['lam']}\neta (must-link порог):  {sp['eta']}\ntheta (constraint порог): {sp['theta']}\n\nИсточник: взвешенное среднее по 3 ближайшим датасетам из оригинальной статьи SDGCA:\n{matches_lines}\n\n=== РАНЖИРОВАНИЕ АЛГОРИТМОВ ===\n{ranking_lines}\n\nЛУЧШИЙ АЛГОРИТМ: {best_label}\n\n=== ПРЕДУПРЕЖДЕНИЯ ===\n{warnings_str}\n\n=== ЗАДАНИЕ ===\nНапиши ответ из ТРЁХ чётко разделённых абзацев:\n\nАБЗАЦ 1 — ХАРАКТЕРИСТИКА ДАТАСЕТА (3-4 предложения):\nОпиши, что из себя представляет этот датасет с точки зрения кластеризации.\nНасколько хорошо разделены классы? Есть ли проблемы (перекрытие, выбросы, дисбаланс, высокая размерность)?\n\nАБЗАЦ 2 — ПОЧЕМУ ИМЕННО ЭТИ ПАРАМЕТРЫ (3-4 предложения):\nОбъясни, почему для SDGCA выбраны lambda={sp['lam']}, eta={sp['eta']}, theta={sp['theta']}.\nОткуда взяты значения (ближайший эталон: {top_match.get('reference_dataset', 'н/д')}).\nЧто каждый параметр делает на практике для этого конкретного датасета.\n\nАБЗАЦ 3 — РЕКОМЕНДАЦИЯ ПО АЛГОРИТМУ (2-3 предложения):\nПочему лучший алгоритм — {best_label}?\nДай практический совет: какой алгоритм запустить первым, на что обратить внимание в результатах.\n"
    return prompt.strip()
