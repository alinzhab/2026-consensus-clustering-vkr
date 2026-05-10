"""Полный статистический анализ существующих результатов.

Подготавливает три анализа:
1. По всем 4 алгоритмам на лучших linkage — Фридман + Неменьи + Wilcoxon-Holm.
2. По 2 иерархическим алгоритмам на множестве linkage — Wilcoxon отдельно
   (нужно для проверки, помогает ли взвешивание).
3. Bootstrap-CI среднего NMI каждого алгоритма по датасетам.

Результаты пишутся в `results/statistical_analysis.md` — этот файл готов
к включению в раздел 3 ВКР.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consensus_lab"))

from statistical_tests_numpy import (  # noqa: E402
    bootstrap_mean_ci,
    format_friedman_summary,
    format_nemenyi_summary,
    format_wilcoxon_summary,
    friedman_test,
    nemenyi_post_hoc,
    wilcoxon_holm,
)


def load_combined() -> pd.DataFrame:
    parts = []
    for name in [
        "full_benchmark.tsv",
        "analysis_full_suite.tsv",
        "hierarchical_linkage_full_suite.tsv",
    ]:
        p = ROOT / "results" / name
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            if "runs" in df.columns:
                df = df[df["runs"] >= 5]
            df["__source"] = name
            parts.append(df)
    df = pd.concat(parts, ignore_index=True, sort=False)
    df = df[df["status"] == "ok"].copy()
    return df


def best_per_dataset(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.sort_values(metric, ascending=False)
        .drop_duplicates(["dataset", "algorithm"])
        .pivot(index="dataset", columns="algorithm", values=metric)
        .dropna(axis=0, how="any")
    )


def section_friedman_4_algorithms(df: pd.DataFrame, metric: str) -> str:
    pivot = best_per_dataset(df, metric)
    cols = [
        "hierarchical_baseline",
        "hierarchical_weighted",
        "sdgca",
        "sdgca_modified",
    ]
    pivot = pivot.reindex(columns=cols).dropna()
    if pivot.shape[0] < 2:
        return "## Friedman 4 alg\n\nНедостаточно общих датасетов.\n"

    fr = friedman_test(pivot.values, list(pivot.columns))
    out = [f"# Статистический анализ ({metric})", "", "## Все 4 алгоритма (лучший linkage)"]
    out.append("")
    out.append(f"Датасеты с полным покрытием: {pivot.shape[0]}.")
    out.append("")
    out.append("Таблица метрики по датасетам:")
    out.append("")
    out.append(pivot.round(3).to_markdown())
    out.append("")
    out.append(format_friedman_summary(fr))
    out.append("")
    if fr.reject_h0(0.05) and 2 <= fr.n_algorithms <= 10:
        nm = nemenyi_post_hoc(fr, alpha=0.05)
        out.append(format_nemenyi_summary(nm))
        out.append("")
    else:
        cd = 1.66  # Для 4 алгоритмов и 8 датасетов: q=2.569, sqrt(4*5/(6*8))≈0.645, CD≈1.66.
        ranks = fr.average_ranks
        all_in_cd = (ranks.max() - ranks.min()) <= cd
        if all_in_cd:
            out.append(
                "**Все четыре алгоритма попадают внутрь критического "
                "расстояния CD ≈ "
                f"{cd:.2f} от лучшего среднего ранга. "
                "На имеющихся данных невозможно утверждать статистически "
                "значимое превосходство одного алгоритма над другим.**"
            )
            out.append("")

    for baseline in ["sdgca_modified", "hierarchical_weighted"]:
        if baseline in pivot.columns:
            wh = wilcoxon_holm(pivot.values, list(pivot.columns), baseline)
            out.append(format_wilcoxon_summary(wh))
            out.append("")
    return "\n".join(out)


def section_hc_baseline_vs_weighted(df: pd.DataFrame, metric: str) -> str:
    """На большом наборе iz hierarchical_linkage_full_suite — реально ли weighted > baseline."""
    sub = df[df["algorithm"].isin(["hierarchical_baseline", "hierarchical_weighted"])]
    if sub.empty:
        return ""
    # Каждая строка — (dataset, method) — независимое сравнение.
    pivot = (
        sub.pivot_table(
            index=["dataset", "method"],
            columns="algorithm",
            values=metric,
            aggfunc="mean",
        )
        .dropna()
    )
    if pivot.shape[0] < 5:
        return ""
    diff = pivot["hierarchical_weighted"] - pivot["hierarchical_baseline"]
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())

    out = [
        "## HC weighted vs HC baseline (по парам dataset × linkage)",
        "",
        f"Сравнений: {pivot.shape[0]}. weighted > baseline: {wins}, "
        f"baseline > weighted: {losses}, равенство: {ties}.",
        f"Среднее улучшение Δ{metric} = {diff.mean():+.4f}, "
        f"медиана Δ = {diff.median():+.4f}.",
        "",
    ]
    # Бинарный знаковый тест: P(wins ≥ k | binom(n=wins+losses, p=0.5))
    n_eff = wins + losses
    if n_eff > 0:
        # Двусторонний знаковый тест без scipy.
        from math import comb

        def two_sided(n: int, k: int) -> float:
            k = min(k, n - k)
            tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
            return min(1.0, 2 * tail)

        p = two_sided(n_eff, min(wins, losses))
        out.append(
            f"Знаковый тест (двусторонний, без учёта величины Δ): p = {p:.4f}."
        )
        out.append("")

    # Парный Вилкоксон.
    diffs = diff.values
    arr = np.column_stack([pivot["hierarchical_baseline"].values, pivot["hierarchical_weighted"].values])
    wh = wilcoxon_holm(arr, ["hierarchical_baseline", "hierarchical_weighted"], "hierarchical_baseline")
    out.append(format_wilcoxon_summary(wh))
    out.append("")
    return "\n".join(out)


def section_bootstrap(df: pd.DataFrame) -> str:
    out = ["## Bootstrap-CI среднего NMI по датасетам", "", "| Алгоритм | mean | 95% CI | n |", "|---|---:|---|---:|"]
    rng = np.random.default_rng(0)
    for alg in [
        "hierarchical_baseline",
        "hierarchical_weighted",
        "sdgca",
        "sdgca_modified",
    ]:
        sub = df[df["algorithm"] == alg]["nmi_mean"].dropna().values
        if sub.size < 2:
            continue
        ci = bootstrap_mean_ci(sub, confidence=0.95, n_resamples=5000, rng=rng)
        out.append(
            f"| {alg} | {ci.mean:.3f} | [{ci.lower:.3f}; {ci.upper:.3f}] | {sub.size} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    df = load_combined()
    print(f"Объединённая таблица: {df.shape}")

    sections = ["# Статистический анализ результатов", ""]
    sections.append(section_friedman_4_algorithms(df, "nmi_mean"))
    sections.append(section_hc_baseline_vs_weighted(df, "nmi_mean"))
    sections.append(section_bootstrap(df))

    out_path = ROOT / "results" / "statistical_analysis.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Записано: {out_path.relative_to(ROOT)}")
    print()
    print("\n".join(sections))


if __name__ == "__main__":
    main()
