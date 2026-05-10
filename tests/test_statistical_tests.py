"""Тесты модуля statistical_tests."""

from __future__ import annotations

import numpy as np
import pytest

from statistical_tests import (
    bootstrap_mean_ci,
    friedman_test,
    nemenyi_post_hoc,
    wilcoxon_holm,
)


def _synthetic_scores(better_idx: int, n_datasets: int = 8, seed: int = 0):
    """Строим матрицу N×4, где алгоритм с индексом `better_idx` стабильно лучше."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.4, 0.7, size=(n_datasets, 4))
    base[:, better_idx] += 0.15
    return base


def test_friedman_detects_clear_winner():
    scores = _synthetic_scores(better_idx=0)
    res = friedman_test(scores, ["A", "B", "C", "D"])
    assert res.p_value < 0.05
    assert res.average_ranks.argmin() == 0  # ранг 1 — самый низкий — лучший


def test_friedman_no_difference_when_random():
    rng = np.random.default_rng(42)
    # Идентичные распределения → ожидаем p > 0.05 (на 8 датасетах).
    scores = rng.uniform(0.5, 0.6, size=(8, 4))
    res = friedman_test(scores, ["A", "B", "C", "D"])
    assert res.p_value > 0.05


def test_nemenyi_critical_distance_positive():
    scores = _synthetic_scores(better_idx=0, n_datasets=10)
    fr = friedman_test(scores, ["A", "B", "C", "D"])
    nm = nemenyi_post_hoc(fr, alpha=0.05)
    assert nm.critical_distance > 0
    # Симметрия и нули на диагонали.
    np.testing.assert_allclose(nm.rank_diff_matrix, nm.rank_diff_matrix.T)
    assert np.all(np.diag(nm.rank_diff_matrix) == 0)


def test_wilcoxon_holm_rejects_for_winner():
    scores = _synthetic_scores(better_idx=0, n_datasets=12)
    res = wilcoxon_holm(scores, ["A", "B", "C", "D"], baseline="A", alpha=0.05)
    # Все три сравнения с явным лидером должны отвергаться.
    rejected = sum(1 for _, _, _, r in res.comparisons if r)
    assert rejected == 3


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.7, scale=0.05, size=30)
    ci = bootstrap_mean_ci(samples, confidence=0.95, n_resamples=2000, rng=0)
    assert ci.lower <= ci.mean <= ci.upper
    assert 0.6 < ci.mean < 0.8


def test_friedman_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        friedman_test(np.array([[1.0, 2.0]]), ["A", "B"])  # 1 датасет
    with pytest.raises(ValueError):
        friedman_test(
            np.array([[np.nan, 0.5], [0.6, 0.7]]), ["A", "B"]
        )  # NaN
