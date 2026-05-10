"""Метрики качества кластеризации.

Модуль централизует все метрики, по которым в проекте сравниваются
результаты консенсус-алгоритмов. Это позволяет:

- использовать одни и те же реализации в `hierarchical_consensus`,
  `hierarchical_consensus_modified`, `sdgca`, `sdgca_modified` и
  отдельных экспериментах из `experiments/`;
- проверять метрики независимо от алгоритмов в `tests/test_metrics.py`;
- при необходимости заменить реализацию (например, перейти с
  `arithmetic` на `geometric` усреднение в NMI) в одном месте.

Метрики:
    contingency               матрица совместной встречаемости меток
    compute_nmi               NMI (sklearn, arithmetic average)
    compute_ari               ARI (sklearn)
    compute_pairwise_f_score  парный F-score (авторская реализация)

Замечание о парном F-score: scikit-learn такой метрики «из коробки» не
предоставляет, поэтому используется собственная реализация по
определению Larsen-Aone (1999): precision/recall на парах объектов
внутри одного предсказанного / истинного кластера.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


__all__ = [
    "contingency",
    "compute_nmi",
    "compute_ari",
    "compute_pairwise_f_score",
]


_LabelArray = Union[np.ndarray, Sequence[int]]


def contingency(labels_a: _LabelArray, labels_b: _LabelArray) -> np.ndarray:
    """Матрица совместной встречаемости двух разбиений.

    Args:
        labels_a: вектор меток первого разбиения, длина n.
        labels_b: вектор меток второго разбиения, длина n.

    Returns:
        Матрица `(k_a, k_b)` целых чисел, где элемент `[i, j]` —
        количество объектов, которые в первом разбиении попали в
        i-й уникальный класс, а во втором — в j-й уникальный класс.
        Метки внутри каждого разбиения произвольны (используется
        переразметка через `np.unique`).

    Raises:
        ValueError: если длины `labels_a` и `labels_b` различаются.
    """
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    if labels_a.size != labels_b.size:
        raise ValueError("label vectors must have the same length")

    _, inv_a = np.unique(labels_a, return_inverse=True)
    _, inv_b = np.unique(labels_b, return_inverse=True)
    cont = np.zeros((inv_a.max() + 1, inv_b.max() + 1), dtype=np.int64)
    np.add.at(cont, (inv_a, inv_b), 1)
    return cont


def compute_nmi(pred_labels: _LabelArray, true_labels: _LabelArray) -> float:
    """Normalized Mutual Information между предсказанием и эталоном.

    Используется реализация из `sklearn.metrics.normalized_mutual_info_score`
    с дефолтным `average_method='arithmetic'`. Значения в `[0, 1]`,
    1 — полное совпадение разбиений (с точностью до перенумерации меток).

    Args:
        pred_labels: предсказанные метки.
        true_labels: истинные метки (ground truth).

    Returns:
        NMI как Python `float`.
    """
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def compute_ari(pred_labels: _LabelArray, true_labels: _LabelArray) -> float:
    """Adjusted Rand Index между предсказанием и эталоном.

    Используется реализация из `sklearn.metrics.adjusted_rand_score`.
    ARI ≈ 0 для случайного разбиения, 1 — полное совпадение, может
    быть слегка отрицательным для разбиений хуже случайного.

    Args:
        pred_labels: предсказанные метки.
        true_labels: истинные метки (ground truth).

    Returns:
        ARI как Python `float`.
    """
    return float(adjusted_rand_score(true_labels, pred_labels))


def compute_pairwise_f_score(
    pred_labels: _LabelArray, true_labels: _LabelArray
) -> float:
    """Парный F-score (Larsen & Aone, 1999) между разбиениями.

    Считает на парах объектов: precision = доля «правильных» пар среди
    всех пар внутри предсказанных кластеров, recall = доля «правильных»
    пар среди всех пар внутри истинных кластеров. F-score = их
    гармоническое среднее.

    Соглашение для вырожденных случаев:
        - если все объекты в синглтонах в предсказании и истине —
          0 пар, метрика возвращает 1.0 (precision = recall = 1).
        - если precision + recall == 0, метрика возвращает 0.0.

    Args:
        pred_labels: предсказанные метки.
        true_labels: истинные метки (ground truth).

    Returns:
        F-score в `[0, 1]` как Python `float`.
    """
    cont = contingency(pred_labels, true_labels).astype(np.float64)
    row_sums = np.sum(cont, axis=1)
    col_sums = np.sum(cont, axis=0)

    intersection_pairs = float(np.sum(cont * (cont - 1.0) / 2.0))
    predicted_pairs = float(np.sum(row_sums * (row_sums - 1.0) / 2.0))
    true_pairs = float(np.sum(col_sums * (col_sums - 1.0) / 2.0))

    precision = 1.0 if predicted_pairs == 0.0 else intersection_pairs / predicted_pairs
    recall = 1.0 if true_pairs == 0.0 else intersection_pairs / true_pairs
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
