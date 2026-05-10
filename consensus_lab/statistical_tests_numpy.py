"""Чисто-numpy версия модуля statistical_tests (без scipy.stats).

Используется в окружениях, где scipy недоступен (CI, sandbox). Реализует
тот же контракт, что `statistical_tests.py`, но через numpy + scipy-
эквивалентные функции, написанные руками. Точность F-распределения и
ranksums — приближённая, но достаточная для отчёта.

Если scipy установлен, используйте основной модуль `statistical_tests.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# --- dataclass-контракты, дублируем локально, чтобы модуль не зависел от scipy
@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    average_ranks: np.ndarray
    n_datasets: int
    n_algorithms: int
    algorithm_names: list[str]

    def reject_h0(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


@dataclass
class NemenyiResult:
    critical_distance: float
    rank_diff_matrix: np.ndarray
    significant_matrix: np.ndarray
    alpha: float
    algorithm_names: list[str]

    def significant_pairs(self) -> list[tuple[str, str, float]]:
        out: list[tuple[str, str, float]] = []
        n = len(self.algorithm_names)
        for i in range(n):
            for j in range(i + 1, n):
                if self.significant_matrix[i, j]:
                    out.append(
                        (
                            self.algorithm_names[i],
                            self.algorithm_names[j],
                            float(self.rank_diff_matrix[i, j]),
                        )
                    )
        return out


@dataclass
class WilcoxonHolmResult:
    baseline: str
    comparisons: list[tuple[str, float, float, bool]]
    alpha: float


@dataclass
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int


_NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
}


def nemenyi_post_hoc(friedman: FriedmanResult, alpha: float = 0.05) -> NemenyiResult:
    if alpha not in _NEMENYI_Q:
        raise ValueError("Поддерживаются alpha=0.05 и alpha=0.10")
    k = friedman.n_algorithms
    n = friedman.n_datasets
    if k not in _NEMENYI_Q[alpha]:
        raise ValueError(f"Таблица q не содержит k={k} (поддерживаются 2..10)")
    q = _NEMENYI_Q[alpha][k]
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n))
    diffs = np.abs(friedman.average_ranks[:, None] - friedman.average_ranks[None, :])
    significant = diffs > cd
    np.fill_diagonal(significant, False)
    return NemenyiResult(
        critical_distance=cd,
        rank_diff_matrix=diffs,
        significant_matrix=significant,
        alpha=alpha,
        algorithm_names=list(friedman.algorithm_names),
    )


def bootstrap_mean_ci(
    samples: Iterable[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> BootstrapCI:
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size < 2:
        raise ValueError("нужно минимум 2 значения для bootstrap")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence должно быть в (0, 1)")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n = arr.size
    means = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = gen.integers(0, n, size=n)
        means[b] = float(np.mean(arr[idx]))
    alpha = 1.0 - confidence
    return BootstrapCI(
        mean=float(np.mean(arr)),
        lower=float(np.quantile(means, alpha / 2.0)),
        upper=float(np.quantile(means, 1.0 - alpha / 2.0)),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def format_friedman_summary(result: FriedmanResult, alpha: float = 0.05) -> str:
    decision = "отвергается" if result.reject_h0(alpha) else "не отвергается"
    lines = [
        "## Friedman test",
        "",
        f"- Статистика F = {result.statistic:.4f}",
        f"- p-value = {result.p_value:.4g}",
        f"- N датасетов = {result.n_datasets}, k алгоритмов = {result.n_algorithms}",
        f"- На уровне alpha = {alpha} H0 (все алгоритмы равны) {decision}.",
        "",
        "### Средние ранги",
        "",
        "| Алгоритм | Средний ранг |",
        "|---|---:|",
    ]
    order = np.argsort(result.average_ranks)
    for j in order:
        lines.append(f"| {result.algorithm_names[j]} | {result.average_ranks[j]:.3f} |")
    return "\n".join(lines)


def format_nemenyi_summary(result: NemenyiResult) -> str:
    lines = [
        "## Nemenyi post-hoc",
        "",
        f"- Критическое расстояние CD = {result.critical_distance:.3f} (alpha = {result.alpha})",
        "",
    ]
    pairs = result.significant_pairs()
    if not pairs:
        lines.append("Ни одна пара алгоритмов не различается значимо.")
    else:
        lines.append("Значимо различающиеся пары:")
        lines.append("")
        for a, b, d in sorted(pairs, key=lambda t: -t[2]):
            lines.append(f"- {a} vs {b}: |Δrank| = {d:.3f}")
    return "\n".join(lines)


def format_wilcoxon_summary(result: WilcoxonHolmResult) -> str:
    lines = [
        f"## Wilcoxon vs {result.baseline} (Holm correction)",
        "",
        "| Сравнение | raw p | holm p | reject H0 |",
        "|---|---:|---:|:---:|",
    ]
    for name, raw_p, holm_p, reject in sorted(result.comparisons, key=lambda t: t[2]):
        mark = "да" if reject else "нет"
        lines.append(f"| {result.baseline} vs {name} | {raw_p:.4g} | {holm_p:.4g} | {mark} |")
    return "\n".join(lines)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Аналог scipy.stats.rankdata(method='average') без scipy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Усреднение для ties.
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # средний ранг блока
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    return ranks


def _f_sf(f_stat: float, df1: int, df2: int) -> float:
    """Хвост F-распределения: P(F > f_stat).

    Реализован через регуляризованную неполную бета-функцию I_x(a,b).
    P(F > f) = I_{df2 / (df2 + df1*f)}(df2/2, df1/2).
    """
    if not np.isfinite(f_stat) or f_stat <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    return _betainc(df2 / 2.0, df1 / 2.0, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Регуляризованная неполная бета I_x(a,b) через цепные дроби (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # ln(B(a,b))
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # log префактора x^a (1-x)^b / (a B(a,b))
    log_pref = a * math.log(x) + b * math.log(1.0 - x) - ln_beta
    if x < (a + 1.0) / (a + b + 2.0):
        cf = _betacf(a, b, x)
        return math.exp(log_pref) * cf / a
    else:
        cf = _betacf(b, a, 1.0 - x)
        return 1.0 - math.exp(log_pref) * cf / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Цепная дробь для неполной бета-функции."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _normal_sf(z: float) -> float:
    """Хвост стандартной нормали."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> float:
    """Парный знаковый тест Вилкоксона (двусторонний), приближённое p-value."""
    diff = x - y
    nz = diff[diff != 0]
    n = nz.size
    if n == 0:
        return 1.0
    if n < 8:
        # Для совсем малых n используем точный перебор (медленно, но n<8).
        return _wilcoxon_exact(nz)
    ranks = _rankdata_average(np.abs(nz))
    w_pos = float(np.sum(ranks[nz > 0]))
    w_neg = float(np.sum(ranks[nz < 0]))
    w = min(w_pos, w_neg)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # Поправка на ties.
    _, counts = np.unique(np.abs(nz), return_counts=True)
    ties = counts[counts > 1]
    if ties.size:
        var -= np.sum(ties**3 - ties) / 48.0
    if var <= 0:
        return 1.0
    z = (w - mean) / math.sqrt(var)
    return 2.0 * _normal_sf(abs(z))


def _wilcoxon_exact(nz: np.ndarray) -> float:
    """Точный двусторонний p-value для очень маленьких выборок."""
    n = nz.size
    ranks = _rankdata_average(np.abs(nz))
    observed_w = float(np.sum(ranks[nz > 0]))
    # Перебор всех 2^n знаков.
    total = 0
    extreme = 0
    expected = float(np.sum(ranks)) / 2.0
    obs_dev = abs(observed_w - expected)
    for mask in range(1 << n):
        w = 0.0
        for i in range(n):
            if (mask >> i) & 1:
                w += ranks[i]
        total += 1
        if abs(w - expected) >= obs_dev - 1e-12:
            extreme += 1
    return extreme / total


def friedman_test(scores, algorithm_names: Sequence[str]) -> FriedmanResult:
    """То же, что `statistical_tests.friedman_test`, но без scipy."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores должна быть матрицей (n_datasets, n_algorithms)")
    n_datasets, n_algorithms = scores.shape
    if len(algorithm_names) != n_algorithms:
        raise ValueError("длина algorithm_names должна совпадать со столбцами scores")
    if n_datasets < 2 or n_algorithms < 2:
        raise ValueError("нужно минимум 2 датасета и 2 алгоритма")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores не должна содержать NaN/Inf")

    ranks = np.zeros_like(scores)
    for i in range(n_datasets):
        ranks[i] = _rankdata_average(-scores[i])
    avg_ranks = ranks.mean(axis=0)
    chi2 = (
        12.0
        * n_datasets
        / (n_algorithms * (n_algorithms + 1))
        * (np.sum(avg_ranks**2) - n_algorithms * (n_algorithms + 1) ** 2 / 4.0)
    )
    denom = n_datasets * (n_algorithms - 1) - chi2
    if denom <= 0:
        f_stat = float("inf")
        p_value = 0.0
    else:
        f_stat = (n_datasets - 1) * chi2 / denom
        df1 = n_algorithms - 1
        df2 = (n_algorithms - 1) * (n_datasets - 1)
        p_value = float(_f_sf(f_stat, df1, df2))
    return FriedmanResult(
        statistic=float(f_stat),
        p_value=p_value,
        average_ranks=avg_ranks,
        n_datasets=n_datasets,
        n_algorithms=n_algorithms,
        algorithm_names=list(algorithm_names),
    )


def wilcoxon_holm(
    scores,
    algorithm_names: Sequence[str],
    baseline: str,
    alpha: float = 0.05,
) -> WilcoxonHolmResult:
    """Вилкоксон + Холм без scipy."""
    scores = np.asarray(scores, dtype=np.float64)
    names = list(algorithm_names)
    if baseline not in names:
        raise ValueError(f"Опорный алгоритм '{baseline}' не найден")
    base_idx = names.index(baseline)
    raw_results: list[tuple[str, float]] = []
    for j, name in enumerate(names):
        if j == base_idx:
            continue
        p = _wilcoxon_signed_rank(scores[:, base_idx], scores[:, j])
        raw_results.append((name, float(p)))
    raw_sorted = sorted(raw_results, key=lambda t: t[1])
    m = len(raw_sorted)
    holm: dict[str, tuple[float, float, bool]] = {}
    keep_rejecting = True
    for i, (name, p) in enumerate(raw_sorted):
        adj = min(1.0, p * (m - i))
        reject = keep_rejecting and (adj < alpha)
        if not reject:
            keep_rejecting = False
        holm[name] = (p, adj, reject)
    comparisons = [(name, *holm[name]) for name, _ in raw_results]
    return WilcoxonHolmResult(baseline=baseline, comparisons=comparisons, alpha=alpha)


__all__ = [
    "friedman_test",
    "nemenyi_post_hoc",
    "wilcoxon_holm",
    "bootstrap_mean_ci",
    "format_friedman_summary",
    "format_nemenyi_summary",
    "format_wilcoxon_summary",
]
