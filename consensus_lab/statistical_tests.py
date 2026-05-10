"""Статистические тесты значимости для сравнения консенсус-алгоритмов.

Модуль закрывает классическую претензию рецензента «mean ± std не является
доказательством превосходства метода». Реализованы стандартные процедуры,
принятые в литературе по сравнению множества классификаторов / алгоритмов
кластеризации на множестве датасетов:

- критерий Фридмана (Friedman) — есть ли вообще различия между алгоритмами;
- post-hoc тест Неменьи (Nemenyi) с критическим расстоянием CD — какие пары
  различаются после отвержения H0 Фридмана;
- парный знаковый тест Вилкоксона (Wilcoxon signed-rank) с поправкой
  Холма-Бонферрони — для попарных сравнений с одним опорным алгоритмом;
- bootstrap-доверительные интервалы для среднего по запускам — вместо
  «mean ± std».

Литература:
    Demsar J. Statistical Comparisons of Classifiers over Multiple Data Sets.
    Journal of Machine Learning Research, 2006, vol. 7, pp. 1-30.

API сделано минималистичным: на вход — pandas-подобная таблица
(`numpy.ndarray` формы (n_datasets, n_algorithms)) и список названий
алгоритмов. На выход — dataclass с p-value, рангами и матрицей значимости.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


@dataclass
class FriedmanResult:
    """Результат теста Фридмана.

    Attributes:
        statistic: значение статистики Фридмана с поправкой Иманн-Давенпорта.
        p_value: достигнутый уровень значимости.
        average_ranks: средний ранг каждого алгоритма (1 = лучший).
        n_datasets: число датасетов.
        n_algorithms: число алгоритмов.
        algorithm_names: имена алгоритмов в порядке столбцов исходной матрицы.
    """

    statistic: float
    p_value: float
    average_ranks: np.ndarray
    n_datasets: int
    n_algorithms: int
    algorithm_names: list[str]

    def reject_h0(self, alpha: float = 0.05) -> bool:
        """True, если на уровне alpha отвергается гипотеза «все алгоритмы равны»."""
        return self.p_value < alpha


@dataclass
class NemenyiResult:
    """Результат post-hoc теста Неменьи.

    Attributes:
        critical_distance: критическое расстояние CD для уровня alpha.
        rank_diff_matrix: матрица |rank_i - rank_j| (симметричная).
        significant_matrix: булева матрица: True, если |rank_i - rank_j| > CD.
        alpha: уровень значимости.
        algorithm_names: имена алгоритмов.
    """

    critical_distance: float
    rank_diff_matrix: np.ndarray
    significant_matrix: np.ndarray
    alpha: float
    algorithm_names: list[str]

    def significant_pairs(self) -> list[tuple[str, str, float]]:
        """Список пар (a, b, разница рангов), для которых различие значимо."""
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
    """Парные тесты Вилкоксона с поправкой Холма к опорному алгоритму.

    Attributes:
        baseline: имя опорного алгоритма.
        comparisons: список (имя_алгоритма, raw_p, holm_p, reject) для остальных.
        alpha: уровень значимости.
    """

    baseline: str
    comparisons: list[tuple[str, float, float, bool]]
    alpha: float


@dataclass
class BootstrapCI:
    """Bootstrap-доверительный интервал для среднего по запускам."""

    mean: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int


# Критические значения q_{alpha, k} для теста Неменьи (двусторонний).
# Источник: Demsar 2006, табл. 5; k — число алгоритмов.
# Допустимые ключи alpha: 0.05, 0.10. Для произвольного alpha вызывайте Tukey HSD.
_NEMENYI_Q = {
    0.05: {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    },
    0.10: {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    },
}


def friedman_test(
    scores: np.ndarray, algorithm_names: Sequence[str]
) -> FriedmanResult:
    """Критерий Фридмана с поправкой Иманн-Давенпорта.

    Args:
        scores: матрица (n_datasets, n_algorithms) со значениями метрики.
            Большее значение = лучше (NMI/ARI/F-score). Если метрика «меньше=лучше»,
            подайте `-scores`.
        algorithm_names: имена алгоритмов в порядке столбцов.

    Returns:
        FriedmanResult с p-value и средними рангами.

    Raises:
        ValueError: если форма матрицы не соответствует именам или мало данных.
    """
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

    # Ранги по строкам, лучший = 1. Поскольку у нас «больше=лучше»,
    # инвертируем знак внутри rankdata.
    ranks = np.zeros_like(scores)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(-scores[i], method="average")
    avg_ranks = ranks.mean(axis=0)

    # Классическая статистика Фридмана.
    chi2 = (
        12.0
        * n_datasets
        / (n_algorithms * (n_algorithms + 1))
        * (np.sum(avg_ranks**2) - n_algorithms * (n_algorithms + 1) ** 2 / 4.0)
    )
    # Поправка Иманн-Давенпорта на F-распределение (мощнее, чем chi2).
    denom = n_datasets * (n_algorithms - 1) - chi2
    if denom <= 0:
        f_stat = float("inf")
        p_value = 0.0
    else:
        f_stat = (n_datasets - 1) * chi2 / denom
        df1 = n_algorithms - 1
        df2 = (n_algorithms - 1) * (n_datasets - 1)
        p_value = float(1.0 - stats.f.cdf(f_stat, df1, df2))

    return FriedmanResult(
        statistic=float(f_stat),
        p_value=p_value,
        average_ranks=avg_ranks,
        n_datasets=n_datasets,
        n_algorithms=n_algorithms,
        algorithm_names=list(algorithm_names),
    )


def nemenyi_post_hoc(
    friedman: FriedmanResult, alpha: float = 0.05
) -> NemenyiResult:
    """Post-hoc тест Неменьи: попарные сравнения после отвержения Фридмана.

    Критическое расстояние:
        CD = q_{alpha,k} * sqrt(k(k+1) / (6 N))

    где k — число алгоритмов, N — число датасетов, q — критическое значение
    студентизированного размаха (двусторонний тест).

    Args:
        friedman: результат `friedman_test`.
        alpha: уровень значимости (поддерживается 0.05 и 0.10).

    Returns:
        NemenyiResult с матрицей значимости.
    """
    if alpha not in _NEMENYI_Q:
        raise ValueError("Поддерживаются alpha=0.05 и alpha=0.10")
    k = friedman.n_algorithms
    n = friedman.n_datasets
    if k not in _NEMENYI_Q[alpha]:
        raise ValueError(f"Таблица q не содержит k={k} (поддерживаются 2..10)")

    q = _NEMENYI_Q[alpha][k]
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n))

    diffs = np.abs(
        friedman.average_ranks[:, None] - friedman.average_ranks[None, :]
    )
    significant = diffs > cd
    np.fill_diagonal(significant, False)

    return NemenyiResult(
        critical_distance=cd,
        rank_diff_matrix=diffs,
        significant_matrix=significant,
        alpha=alpha,
        algorithm_names=list(friedman.algorithm_names),
    )


def wilcoxon_holm(
    scores: np.ndarray,
    algorithm_names: Sequence[str],
    baseline: str,
    alpha: float = 0.05,
) -> WilcoxonHolmResult:
    """Попарные тесты Вилкоксона с поправкой Холма-Бонферрони к одному опорному.

    Используйте, когда важно сравнить новый алгоритм со всеми остальными
    (классический паттерн «модифицированный vs все базовые»). В отличие от
    Неменьи, Вилкоксон не требует нормальности и устойчив к выбросам по
    датасетам, но имеет меньше мощности, чем классический парный t-тест
    при выполненных предположениях.

    Args:
        scores: матрица (n_datasets, n_algorithms) с метрикой.
        algorithm_names: имена алгоритмов в порядке столбцов.
        baseline: имя опорного алгоритма (должно быть в `algorithm_names`).
        alpha: уровень значимости после поправки Холма.

    Returns:
        WilcoxonHolmResult с raw_p, holm_p и решением reject H0 для каждой пары.
    """
    scores = np.asarray(scores, dtype=np.float64)
    names = list(algorithm_names)
    if baseline not in names:
        raise ValueError(f"Опорный алгоритм '{baseline}' не найден в algorithm_names")
    base_idx = names.index(baseline)

    raw_results: list[tuple[str, float]] = []
    for j, name in enumerate(names):
        if j == base_idx:
            continue
        diff = scores[:, base_idx] - scores[:, j]
        if np.allclose(diff, 0.0):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(
                    scores[:, base_idx], scores[:, j], zero_method="wilcox"
                )
            except ValueError:
                p = 1.0
        raw_results.append((name, float(p)))

    # Поправка Холма: упорядочить p по возрастанию и применить
    # alpha_i = alpha / (m - i) для i-го по порядку (i=0..m-1).
    raw_results_sorted = sorted(raw_results, key=lambda t: t[1])
    m = len(raw_results_sorted)
    holm_results: dict[str, tuple[float, float, bool]] = {}
    reject_so_far = True
    for i, (name, p) in enumerate(raw_results_sorted):
        adjusted = min(1.0, p * (m - i))
        # Холм-step-down: как только не отвергли — все следующие тоже не отвергаем.
        reject = reject_so_far and (adjusted < alpha)
        if not reject:
            reject_so_far = False
        holm_results[name] = (p, adjusted, reject)

    comparisons = [
        (name, *holm_results[name]) for name, _ in raw_results
    ]
    return WilcoxonHolmResult(
        baseline=baseline, comparisons=comparisons, alpha=alpha
    )


def bootstrap_mean_ci(
    samples: Iterable[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> BootstrapCI:
    """Bootstrap-percentile доверительный интервал для среднего.

    Использовать вместо `mean ± std` при отчёте по cnt_times запускам одного
    алгоритма на одном датасете: интервал не предполагает нормальности и
    показывает реальную неопределённость оценки среднего.

    Args:
        samples: значения метрики по запускам.
        confidence: уровень доверия (0.95 = 95% CI).
        n_resamples: число bootstrap-выборок.
        rng: numpy.random.Generator или seed.

    Returns:
        BootstrapCI(mean, lower, upper, confidence, n_resamples).
    """
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size < 2:
        raise ValueError("нужно минимум 2 значения для bootstrap")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence должно быть в (0, 1)")
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    n = arr.size
    means = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = gen.integers(0, n, size=n)
        means[b] = float(np.mean(arr[idx]))
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapCI(
        mean=float(np.mean(arr)),
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_resamples=n_resamples,
    )


def format_friedman_summary(result: FriedmanResult, alpha: float = 0.05) -> str:
    """Текстовая сводка для отчёта (Markdown)."""
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
        lines.append(
            f"| {result.algorithm_names[j]} | {result.average_ranks[j]:.3f} |"
        )
    return "\n".join(lines)


def format_nemenyi_summary(result: NemenyiResult) -> str:
    """Текстовая сводка результатов Неменьи."""
    lines = [
        "## Nemenyi post-hoc",
        "",
        f"- Критическое расстояние CD = {result.critical_distance:.3f} "
        f"(alpha = {result.alpha})",
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
    """Текстовая сводка попарных Вилкоксонов."""
    lines = [
        f"## Wilcoxon vs {result.baseline} (Holm correction)",
        "",
        "| Сравнение | raw p | holm p | reject H0 |",
        "|---|---:|---:|:---:|",
    ]
    for name, raw_p, holm_p, reject in sorted(
        result.comparisons, key=lambda t: t[2]
    ):
        mark = "да" if reject else "нет"
        lines.append(
            f"| {result.baseline} vs {name} | {raw_p:.4g} | {holm_p:.4g} | {mark} |"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI-обвязка: читает results/analysis_full_suite.tsv и печатает сводку.

    Ожидаемые столбцы: dataset, algorithm, method, nmi_mean (или nmi).
    Сводка строится по столбцу метрики, выбираемой через --metric.
    """
    import argparse
    import csv
    from collections import defaultdict
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=Path(__file__).resolve().parents[1] / "results" / "analysis_full_suite.tsv",
    )
    parser.add_argument("--metric", default="nmi_mean")
    parser.add_argument("--method", default=None, help="Фильтр по linkage")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Имя опорного алгоритма для Wilcoxon-Holm",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    path = Path(args.input)
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if args.method and row.get("method") != args.method:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"Нет данных в {path} (фильтр method={args.method})")

    table: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        try:
            value = float(row[args.metric])
        except (KeyError, ValueError):
            continue
        algo = row["algorithm"]
        if args.method is None:
            algo = f"{algo}/{row.get('method','?')}"
        table[row["dataset"]][algo] = value

    algorithms = sorted({a for d in table.values() for a in d})
    datasets = [d for d in sorted(table) if all(a in table[d] for a in algorithms)]
    if len(datasets) < 2:
        raise SystemExit("Мало датасетов с полным покрытием для теста Фридмана")

    matrix = np.array(
        [[table[d][a] for a in algorithms] for d in datasets], dtype=np.float64
    )
    friedman = friedman_test(matrix, algorithms)
    print(format_friedman_summary(friedman, args.alpha))
    print()
    if friedman.reject_h0(args.alpha) and friedman.n_algorithms <= 10:
        print(format_nemenyi_summary(nemenyi_post_hoc(friedman, args.alpha)))
        print()
    if args.baseline:
        wh = wilcoxon_holm(matrix, algorithms, args.baseline, args.alpha)
        print(format_wilcoxon_summary(wh))


if __name__ == "__main__":
    main()
