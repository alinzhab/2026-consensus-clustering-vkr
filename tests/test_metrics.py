"""Тесты функций качества кластеризации (metrics.py).

Покрывают граничные случаи, в которых легче всего ошибиться при ручной
реализации pairwise F-score (эту функцию проект пишет сам, а не берёт
из sklearn).
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    compute_ari,
    compute_nmi,
    compute_pairwise_f_score,
    contingency,
)


def test_perfect_match_metrics():
    labels = np.array([1, 1, 2, 2, 3, 3])
    assert compute_nmi(labels, labels) == pytest.approx(1.0)
    assert compute_ari(labels, labels) == pytest.approx(1.0)
    assert compute_pairwise_f_score(labels, labels) == pytest.approx(1.0)


def test_completely_random_clustering_close_to_zero():
    rng = np.random.default_rng(0)
    truth = np.array([1] * 100 + [2] * 100)
    pred = rng.integers(1, 3, size=200)
    assert compute_ari(pred, truth) < 0.2


def test_pairwise_f_score_handles_singletons():
    truth = np.array([1, 2, 3, 4])
    pred = np.array([1, 2, 3, 4])
    # Никаких пар внутри классов — соглашение: precision=recall=1, F=1.
    assert compute_pairwise_f_score(pred, truth) == pytest.approx(1.0)


def test_contingency_shape_and_sum():
    a = np.array([1, 1, 2, 2, 3])
    b = np.array([1, 2, 2, 1, 3])
    cont = contingency(a, b)
    assert cont.shape == (3, 3)
    assert cont.sum() == a.size


def test_label_relabeling_invariance():
    a = np.array([1, 1, 2, 2])
    b = np.array([7, 7, 9, 9])  # та же структура, другие метки
    assert compute_nmi(a, b) == pytest.approx(1.0)
    assert compute_ari(a, b) == pytest.approx(1.0)
    assert compute_pairwise_f_score(a, b) == pytest.approx(1.0)
