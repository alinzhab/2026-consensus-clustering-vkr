"""QD-selection: Quality-Diversity субдискретизация ансамбля базовых кластеризаций.

Вклад работы: вместо случайного выбора m разбиений из пула предлагается жадный
алгоритм, балансирующий качество (согласованность разбиения с большинством) и
разнообразие (непохожесть на уже выбранные разбиения). Применяется ко всем
четырём алгоритмам консенсус кластеризации как независимый шаг.

Теоретическое обоснование:
    Если ансамбль содержит дублирующие или нестабильные разбиения, co-association
    матрица искажается: пары объектов переоцениваются (дубли вносят повторный вклад)
    или занижаются (случайные разбиения добавляют шум). QD-отбор устраняет дубли и
    сохраняет информативное разнообразие, что повышает качество итоговой консенсус-
    матрицы без изменения самого алгоритма консенсуса.

Ссылки:
    - Vega-Pons & Ruiz-Shulcloper (2011): обзор ансамблевой кластеризации.
    - ESDF (Berikov & Popova, 2015): отбор ансамбля по diversity/frequency.
    - Cluster ensemble selection survey (EJOR 2024).
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "DATASET_TYPES",
    "compute_partition_quality",
    "compute_pairwise_agreement",
    "get_dataset_type",
    "select_qd_subset",
]


# ---------------------------------------------------------------------------
# Словарь типов датасетов — для анализа по доменам в экспериментах.
# Ключ: имя датасета без расширения. Значение: тип.
# ---------------------------------------------------------------------------
DATASET_TYPES: dict[str, str] = {
    # Реальные биологические данные
    "Ecoli": "real_bio",
    "GLIOMA": "real_bio",
    "Lung": "real_bio",
    # Реальные текстовые / документальные данные
    "BBC": "real_text",
    # Реальные высокоразмерные данные
    "orlraws10P": "high_dimensional",
    # Синтетические компактные кластеры
    "Aggregation": "compact",
    "densired_compact_hard": "compact",
    "analysis_densired_compact": "compact",
    "analysis_simple_separated": "compact",
    "custom_densired_dataset": "compact",
    # Синтетические перекрывающиеся кластеры
    "analysis_simple_overlap": "overlapping",
    "analysis_repliclust_oblong": "overlapping",
    "repliclust_oblong_overlap": "overlapping",
    # Синтетические вытянутые кластеры
    "densired_stretched_hard": "elongated",
    "analysis_densired_stretched": "elongated",
    # Синтетические с дисбалансом размеров
    "analysis_imbalanced": "imbalanced",
    # Синтетические высокоразмерные
    "analysis_highdim": "high_dimensional",
    "repliclust_highdim_hard": "high_dimensional",
    # Синтетические сложные / неоднородные
    "densired_mix_hard": "mixed_complex",
    "analysis_repliclust_heterogeneous": "mixed_complex",
    "repliclust_heterogeneous_hard": "mixed_complex",
    # Designed thesis experiment datasets
    "design_compact_easy_5k": "compact",
    "design_compact_easy_8k": "compact",
    "design_overlap_moderate": "overlapping",
    "design_overlap_oblong": "overlapping",
    "design_imbalanced_6x": "imbalanced",
    "design_imbalanced_oblong": "imbalanced",
    "design_highdim_20d": "high_dimensional",
    "design_highdim_40d": "high_dimensional",
    "design_elongated_2d": "elongated",
    "design_elongated_density": "elongated",
    "design_density_varied_low_noise": "density_varied",
    "design_density_varied_noisy": "density_varied",
    "design_mixed_complex_6d": "mixed_complex",
    "design_mixed_complex_branchy": "mixed_complex",
    "design_mini_compact": "compact",
    "design_mini_overlap": "overlapping",
    "design_mini_imbalanced": "imbalanced",
    "design_mini_highdim": "high_dimensional",
    "design_mini_elongated": "elongated",
    "design_mini_density_varied": "density_varied",
    "design_mini_mixed_complex": "mixed_complex",
}

# Человекочитаемые метки типов для отчётов
DATASET_TYPE_LABELS: dict[str, str] = {
    "real_bio": "Реальные (биология)",
    "real_text": "Реальные (текст/документы)",
    "compact": "Синт. компактные",
    "overlapping": "Синт. перекрывающиеся",
    "elongated": "Синт. вытянутые",
    "imbalanced": "Синт. с дисбалансом",
    "high_dimensional": "Высокоразмерные",
    "mixed_complex": "Синт. сложные",
    "density_varied": "Синт. неоднородная плотность",
    "unknown": "Неизвестный тип",
}


def get_dataset_type(dataset_name: str) -> str:
    """Определить тип датасета по его имени.

    Поиск без учёта расширения (.mat, .npz, .csv). Если точное совпадение
    не найдено — пробует поиск по подстроке. При отсутствии совпадения
    возвращает 'unknown'.

    Args:
        dataset_name: имя файла датасета (с расширением или без).

    Returns:
        Код типа из DATASET_TYPES или 'unknown'.
    """
    name = str(dataset_name)
    for suffix in (".mat", ".npz", ".csv", ".tsv", ".txt", ".json"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    # Прямое совпадение (case-sensitive)
    if name in DATASET_TYPES:
        return DATASET_TYPES[name]
    # Поиск по подстроке: ключ содержится в имени или наоборот
    for key, dtype in DATASET_TYPES.items():
        if key in name or name in key:
            return dtype
    return "unknown"


# ---------------------------------------------------------------------------
# Внутренние вычислительные функции
# ---------------------------------------------------------------------------


def _partition_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Rand-like согласие [0, 1] между двумя векторами меток.

    Вычислено через таблицу совместной встречаемости: O(n) по памяти
    вместо O(n²). Независимая реализация без импортов из других модулей.

    Returns:
        1.0 при полном совпадении разбиений, ниже при расхождении.
    """
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    n = int(a.size)
    _, inv_a = np.unique(a, return_inverse=True)
    _, inv_b = np.unique(b, return_inverse=True)
    ka = int(inv_a.max()) + 1
    kb = int(inv_b.max()) + 1
    cont = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(cont, (inv_a, inv_b), 1)
    row_sq = float(np.sum(cont.sum(axis=1) ** 2))
    col_sq = float(np.sum(cont.sum(axis=0) ** 2))
    both_sq = float(np.sum(cont**2))
    return float((n**2 - row_sq - col_sq + 2.0 * both_sq) / n**2)


def compute_pairwise_agreement(members: np.ndarray) -> np.ndarray:
    """Попарная матрица согласия для всех M базовых кластеризаций.

    Args:
        members: (n, M) матрица меток базовых кластеризаций.

    Returns:
        (M, M) симметричная float64 матрица в [0, 1], диагональ = 1.

    Сложность: O(M² · n) по времени, O(M²) по памяти.
    """
    M = members.shape[1]
    agg = np.eye(M, dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            v = _partition_agreement(members[:, i], members[:, j])
            agg[i, j] = v
            agg[j, i] = v
    return agg


def compute_partition_quality(agreement_matrix: np.ndarray) -> np.ndarray:
    """Качество каждой базовой кластеризации из матрицы согласий.

    Качество i-й кластеризации = среднее согласие с остальными M-1
    кластеризациями (без учёта самосогласия по диагонали).

    Интерпретация: высокое качество → кластеризация согласована с
    большинством ансамбля; низкое → разбиение нестабильно или случайно.

    Args:
        agreement_matrix: (M, M) матрица из compute_pairwise_agreement.

    Returns:
        (M,) float64 вектор качества в [0, 1].
    """
    M = agreement_matrix.shape[0]
    if M == 1:
        return np.ones(1, dtype=np.float64)
    # Сумма по строке − 1 (диагональ) / (M−1)
    off_diag_sum = agreement_matrix.sum(axis=1) - 1.0
    return off_diag_sum / float(M - 1)


# ---------------------------------------------------------------------------
# Основная функция QD-selection
# ---------------------------------------------------------------------------


def select_qd_subset(
    members: np.ndarray,
    m: int,
    qd_alpha: float = 0.5,
    agreement_matrix: np.ndarray | None = None,
    quality_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Жадный QD-выбор m базовых кластеризаций из пула.

    Алгоритм:
        1. Вычислить попарную матрицу согласий (M × M).
        2. Качество кластеризации i = среднее согласие с остальными.
        3. Инициализация: выбрать кластеризацию с максимальным качеством.
        4. Жадный шаг: добавлять кластеризацию j, максимизирующую
               score(j) = alpha * quality(j) + (1-alpha) * diversity(j),
           где diversity(j) = среднее (1 - agreement[j, уже_выбранные]).
        5. Повторять до |selected| == m.

    Args:
        members: (n, M) матрица базовых кластеризаций (все доступные).
        m: число выбираемых кластеризаций, 1 ≤ m ≤ M.
        qd_alpha: вес качества относительно разнообразия, в [0, 1].
            0.0 — максимальное разнообразие (игнорировать качество),
            1.0 — максимальное качество (игнорировать разнообразие),
            0.5 — баланс качества и разнообразия (значение по умолчанию).
        agreement_matrix: предвычисленная (M, M) матрица; если None — будет
            вычислена через compute_pairwise_agreement. Передавать, если
            матрица уже посчитана (для избежания повторных вычислений).
        quality_scores: предвычисленные (M,) баллы качества; если None —
            вычислятся из agreement_matrix.

    Returns:
        (m,) int64 массив индексов выбранных столбцов members.
        Гарантированно уникальных, в порядке жадного добавления.

    Notes:
        При selection_strategy="qd" все cnt_times прогонов алгоритма
        консенсус кластеризации используют одно и то же подмножество →
        std метрик отражает детерминированность алгоритма, а не дисперсию
        субдискретизации. Это feature, а не bug: QD-selection детерминирован
        и стабилен в отличие от случайного выбора.
    """
    M = int(members.shape[1])
    m = min(int(m), M)

    if m == M:
        return np.arange(M, dtype=np.int64)

    if agreement_matrix is None:
        agreement_matrix = compute_pairwise_agreement(members)

    if quality_scores is None:
        quality_scores = compute_partition_quality(agreement_matrix)

    qd_alpha = float(np.clip(qd_alpha, 0.0, 1.0))

    # Инициализация: лучший по качеству
    selected: list[int] = [int(np.argmax(quality_scores))]
    remaining = list(set(range(M)) - {selected[0]})

    while len(selected) < m and remaining:
        sel_arr = np.array(selected, dtype=np.int64)
        # Векторизованный расчёт diversity для всех remaining
        rem_arr = np.array(remaining, dtype=np.int64)
        # (len(remaining), len(selected)) матрица диссимилярностей
        diss = 1.0 - agreement_matrix[np.ix_(rem_arr, sel_arr)]
        avg_div = diss.mean(axis=1)  # (len(remaining),)
        qual = quality_scores[rem_arr]  # (len(remaining),)
        scores = qd_alpha * qual + (1.0 - qd_alpha) * avg_div
        best_local = int(np.argmax(scores))
        best_global = remaining[best_local]
        selected.append(best_global)
        remaining.pop(best_local)

    return np.array(selected[:m], dtype=np.int64)
