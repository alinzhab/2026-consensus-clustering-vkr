"""Сборка графиков для главы 3 ВКР из существующих TSV-результатов.

На вход — `results/full_benchmark.tsv` и
`results/hierarchical_linkage_full_suite.tsv` (оба прогона с `runs=5`).
На выход — PNG в `results/plots/`. Каждый график рассчитан на одну
страницу пояснительной записки или на одну вкладку слайда защиты.

Запуск:
    python experiments/plot_results.py

Зависимости: numpy, pandas, matplotlib (без scipy/sklearn).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALG_ORDER = [
    "hierarchical_baseline",
    "hierarchical_weighted",
    "sdgca",
    "sdgca_modified",
]
ALG_LABEL = {
    "hierarchical_baseline": "HC base",
    "hierarchical_weighted": "HC weighted",
    "sdgca": "SDGCA",
    "sdgca_modified": "SDGCA mod",
}
METHOD_ORDER = ["average", "complete", "single", "ward"]
METRIC_ORDER = ["nmi_mean", "ari_mean", "f_mean"]
METRIC_LABEL = {"nmi_mean": "NMI", "ari_mean": "ARI", "f_mean": "F-score"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    out = PLOTS / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_nmi_by_algorithm(df: pd.DataFrame) -> None:
    """Boxplot NMI/ARI/F по алгоритмам, агрегируя по датасетам и linkage."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, metric in zip(axes, METRIC_ORDER):
        data = [
            df.loc[df["algorithm"] == alg, metric].dropna().values
            for alg in ALG_ORDER
        ]
        ax.boxplot(
            data,
            labels=[ALG_LABEL[a] for a in ALG_ORDER],
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 5},
        )
        ax.set_title(METRIC_LABEL[metric])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Метрика качества")
    fig.suptitle(
        "Распределение метрик по алгоритмам (по датасетам × linkage)",
        fontsize=12,
    )
    _save(fig, "01_metrics_by_algorithm.png")


def plot_heatmap_nmi(df: pd.DataFrame) -> None:
    """Тепловая карта: средний NMI по парам алгоритм × linkage."""
    pivot = (
        df.groupby(["algorithm", "method"])["nmi_mean"]
        .mean()
        .unstack("method")
        .reindex(index=ALG_ORDER, columns=METHOD_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_yticks(np.arange(len(ALG_ORDER)))
    ax.set_yticklabels([ALG_LABEL[a] for a in ALG_ORDER])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.5 else "black",
                    fontsize=10,
                )
    fig.colorbar(im, ax=ax, label="средний NMI")
    ax.set_title("Средний NMI по комбинациям алгоритм × linkage")
    _save(fig, "02_heatmap_nmi.png")


def plot_per_dataset_grouped(df: pd.DataFrame, metric: str = "nmi_mean") -> None:
    """Сгруппированные столбцы: по каждому датасету — лучший linkage каждого алгоритма."""
    best = (
        df.sort_values(metric, ascending=False)
        .drop_duplicates(["dataset", "algorithm"])
    )
    datasets = sorted(best["dataset"].unique())
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(datasets) + 4), 5))
    x = np.arange(len(datasets))
    for i, alg in enumerate(ALG_ORDER):
        sub = best[best["algorithm"] == alg].set_index("dataset")
        values = [sub.loc[d, metric] if d in sub.index else np.nan for d in datasets]
        ax.bar(x + (i - 1.5) * width, values, width, label=ALG_LABEL[alg])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("analysis_", "").replace("_hard", "") for d in datasets],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Лучший {METRIC_LABEL[metric]} по каждому датасету (по выбору linkage)"
    )
    ax.legend(loc="lower right", ncol=4, fontsize=9, framealpha=0.95)
    _save(fig, f"03_best_per_dataset_{metric}.png")


def plot_average_ranks(df: pd.DataFrame, metric: str = "nmi_mean") -> None:
    """Столбцы со средним рангом каждого алгоритма + критическое расстояние Неменьи."""
    # Берём лучший linkage каждого алгоритма для каждого датасета.
    best = (
        df.sort_values(metric, ascending=False)
        .drop_duplicates(["dataset", "algorithm"])
    )
    pivot = best.pivot(index="dataset", columns="algorithm", values=metric)
    # Отбрасываем датасеты без полного покрытия (NaN хотя бы в одном алгоритме).
    pivot = pivot.dropna(axis=0, how="any").reindex(columns=ALG_ORDER)
    if pivot.empty:
        print("plot_average_ranks: нет датасетов с полным покрытием")
        return
    n = pivot.shape[0]
    k = pivot.shape[1]
    # Ранги по строкам, лучший = 1 (большее значение метрики = лучше).
    ranks = (-pivot.values).argsort(axis=1).argsort(axis=1) + 1.0
    avg_ranks = ranks.mean(axis=0)

    # Критическое расстояние Неменьи (k=4, alpha=0.05): q=2.569.
    cd = 2.569 * np.sqrt(k * (k + 1) / (6.0 * n))

    order = np.argsort(avg_ranks)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        np.arange(k),
        avg_ranks[order],
        color="#3b7dd8",
    )
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels([ALG_LABEL[ALG_ORDER[j]] for j in order])
    ax.invert_yaxis()
    ax.set_xlabel("Средний ранг (меньше = лучше)")
    ax.set_xlim(0, k + 0.5)
    for bar, value in zip(bars, avg_ranks[order]):
        ax.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=10,
        )
    ax.set_title(
        f"Средний ранг по {METRIC_LABEL[metric]} ({n} датасетов, "
        f"CD$_{{0.05}}$ = {cd:.2f})"
    )
    ax.grid(axis="x", alpha=0.3)
    # Линия критического расстояния от лучшего среднего ранга.
    best_rank = avg_ranks[order][0]
    ax.axvspan(best_rank, best_rank + cd, alpha=0.12, color="red")
    ax.text(
        best_rank + cd / 2,
        -0.6,
        "зона неотличимости от лидера (CD)",
        fontsize=8,
        color="darkred",
        ha="center",
    )
    _save(fig, f"04_average_ranks_{metric}.png")


# Известные размеры датасетов (для тех, по которым нет столбца n_objects).
# Реальные .mat — стандартные, размеры из карт датасетов;
# .npz — реальные размеры из массива X в файле.
DATASET_SIZES = {
    # Real
    "Aggregation": 788,
    "BBC": 2225,
    "Ecoli": 336,
    "GLIOMA": 50,
    "Lung": 203,
    "orlraws10P": 100,
    # Synthetic .npz
    "custom_densired_dataset": 2000,
    "densired_compact_hard": 3000,
    "densired_mix_hard": 3500,
    "densired_stretched_hard": 3500,
    "repliclust_heterogeneous_hard": 3200,
    "repliclust_highdim_hard": 4000,
    "repliclust_oblong_overlap": 2500,
    # analysis_*
    "analysis_densired_compact": 3200,
    "analysis_densired_stretched": 3800,
    "analysis_highdim": 3500,
    "analysis_imbalanced": 3500,
    "analysis_repliclust_heterogeneous": 4000,
    "analysis_repliclust_oblong": 3000,
    "analysis_simple_overlap": 3500,
    "analysis_simple_separated": 3000,
}


def plot_complexity_scatter(df: pd.DataFrame) -> None:
    """Время прогона vs число объектов: проверка квадратичной сложности.

    Перед подгонкой подмешиваем известные `n_objects` для тех датасетов,
    в которых столбец отсутствует в TSV (`hierarchical_linkage_full_suite`).
    """
    if "seconds" not in df.columns:
        print("plot_complexity_scatter: нет столбца seconds, пропускаю")
        return
    enriched = df.copy()
    if "n_objects" not in enriched.columns:
        enriched["n_objects"] = np.nan
    fill_mask = enriched["n_objects"].isna()
    enriched.loc[fill_mask, "n_objects"] = enriched.loc[fill_mask, "dataset"].map(
        DATASET_SIZES
    )
    sub = enriched.dropna(subset=["seconds", "n_objects"]).copy()
    sub = sub[sub["seconds"] > 0]
    if sub.empty:
        print("plot_complexity_scatter: нет валидных точек")
        return

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = dict(
        zip(ALG_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    )

    # Подгонки делаем отдельно по группе HC (hierarchical_*) и SDGCA — они
    # сильно отличаются по абсолютному времени, общая прямая мисляидит.
    groups = {
        "HC": ["hierarchical_baseline", "hierarchical_weighted"],
        "SDGCA": ["sdgca", "sdgca_modified"],
    }
    fit_summaries: list[str] = []
    for group_name, algs in groups.items():
        group_data = sub[sub["algorithm"].isin(algs)]
        for alg in algs:
            s = group_data[group_data["algorithm"] == alg]
            if s.empty:
                continue
            ax.scatter(
                s["n_objects"],
                s["seconds"],
                label=ALG_LABEL[alg],
                alpha=0.65,
                color=colors[alg],
                s=28,
                edgecolor="black",
                linewidth=0.4,
            )
        if len(group_data) >= 4 and group_data["n_objects"].nunique() >= 3:
            xs = group_data["n_objects"].astype(float).values
            ys = group_data["seconds"].astype(float).values
            b, log_a = np.polyfit(np.log(xs), np.log(ys), 1)
            a_coef = float(np.exp(log_a))
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ls = "--" if group_name == "HC" else ":"
            ax.plot(
                x_line,
                a_coef * x_line**b,
                ls,
                color="black",
                alpha=0.5,
            )
            fit_summaries.append(f"{group_name}: t ∝ n^{b:.2f}")

    ax.set_xlabel("Число объектов n")
    ax.set_ylabel("Время прогона, с")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    title = "Время прогона как функция размера датасета (log-log)"
    if fit_summaries:
        title += "\n" + "; ".join(fit_summaries)
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, "05_complexity_scatter.png")


def plot_linkage_comparison(df: pd.DataFrame) -> None:
    """Сравнение четырёх linkage по среднему NMI отдельно для каждого алгоритма."""
    fig, axes = plt.subplots(1, len(ALG_ORDER), figsize=(13, 3.6), sharey=True)
    for ax, alg in zip(axes, ALG_ORDER):
        sub = df[df["algorithm"] == alg]
        if sub.empty:
            ax.set_title(ALG_LABEL[alg])
            continue
        values = [sub.loc[sub["method"] == m, "nmi_mean"].mean() for m in METHOD_ORDER]
        ax.bar(METHOD_ORDER, values, color="#5e8c61")
        ax.set_ylim(0, 1.0)
        ax.set_title(ALG_LABEL[alg])
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            if not np.isnan(v):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Средний NMI")
    fig.suptitle("Сравнение linkage в каждом алгоритме", fontsize=12)
    _save(fig, "06_linkage_comparison.png")


def plot_uncertainty(df: pd.DataFrame) -> None:
    """Bar plot со средним и std для каждого алгоритма по NMI."""
    grouped = df.groupby("algorithm")["nmi_mean"]
    means = grouped.mean().reindex(ALG_ORDER)
    stds = grouped.std().reindex(ALG_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ALG_ORDER))
    ax.bar(
        x,
        means.values,
        yerr=stds.values,
        capsize=6,
        color="#7c3aed",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ALG_LABEL[a] for a in ALG_ORDER], rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("NMI: среднее ± std (по датасетам × linkage)")
    ax.set_title("Разброс качества по алгоритмам")
    ax.grid(axis="y", alpha=0.3)
    for xi, m, s in zip(x, means.values, stds.values):
        ax.text(xi, m + s + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=9)
    _save(fig, "07_uncertainty.png")


def main() -> None:
    print("Загружаю данные ...")
    sources = []
    for name in ["full_benchmark.tsv", "analysis_full_suite.tsv"]:
        p = RESULTS / name
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            if "runs" in df.columns:
                df = df[df["runs"] >= 5]  # отбрасываем «runs=1» — там std=0
            df["__source"] = name
            sources.append(df)
            print(f"  {name}: {df.shape}")
    if not sources:
        raise SystemExit("Нет TSV для построения графиков")
    df_all = pd.concat(sources, ignore_index=True, sort=False)
    df_all = df_all[df_all["status"] == "ok"]

    # Дополним hierarchical_linkage_full_suite (у него больше данных по HC).
    extra = RESULTS / "hierarchical_linkage_full_suite.tsv"
    if extra.exists():
        df_extra = pd.read_csv(extra, sep="\t")
        df_extra = df_extra[df_extra["status"] == "ok"]
        df_extra["__source"] = "hierarchical_linkage_full_suite.tsv"
        df_all = pd.concat([df_all, df_extra], ignore_index=True, sort=False)

    print(f"Объединённая таблица: {df_all.shape}")
    plot_nmi_by_algorithm(df_all)
    plot_heatmap_nmi(df_all)
    plot_per_dataset_grouped(df_all, "nmi_mean")
    plot_per_dataset_grouped(df_all, "ari_mean")
    plot_average_ranks(df_all, "nmi_mean")
    plot_complexity_scatter(df_all)
    plot_linkage_comparison(df_all)
    plot_uncertainty(df_all)

    files = sorted(PLOTS.glob("*.png"))
    print(f"\nСгенерировано {len(files)} графиков:")
    for f in files:
        print(f"  {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
