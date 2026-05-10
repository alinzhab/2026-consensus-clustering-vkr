from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
INPUT_TSV = RESULTS_DIR / "sdgca_vs_modified_detailed.tsv"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def parse_float(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def prepare_rows(rows):
    prepared = []
    for row in rows:
        prepared.append(
            {
                "dataset": row["dataset"],
                "type": row["type"],
                "objects": int(row["objects"]) if row.get("objects") else None,
                "features": int(row["features"]) if row.get("features") else None,
                "classes": int(row["classes"]) if row.get("classes") else None,
                "imbalance_ratio": parse_float(row.get("imbalance_ratio", "")),
                "sdgca_nmi": parse_float(row.get("sdgca_nmi", "")),
                "sdgca_nmi_std": parse_float(row.get("sdgca_nmi_std", "")),
                "sdgca_ari": parse_float(row.get("sdgca_ari", "")),
                "sdgca_ari_std": parse_float(row.get("sdgca_ari_std", "")),
                "sdgca_f": parse_float(row.get("sdgca_f", "")),
                "sdgca_f_std": parse_float(row.get("sdgca_f_std", "")),
                "mod_nmi": parse_float(row.get("mod_nmi", "")),
                "mod_nmi_std": parse_float(row.get("mod_nmi_std", "")),
                "mod_ari": parse_float(row.get("mod_ari", "")),
                "mod_ari_std": parse_float(row.get("mod_ari_std", "")),
                "mod_f": parse_float(row.get("mod_f", "")),
                "mod_f_std": parse_float(row.get("mod_f_std", "")),
                "delta_nmi": parse_float(row.get("delta_nmi", "")),
                "delta_ari": parse_float(row.get("delta_ari", "")),
                "delta_f": parse_float(row.get("delta_f", "")),
                "winner": row.get("winner", ""),
            }
        )
    return prepared


def rows_with_complete_metrics(rows):
    return [
        row
        for row in rows
        if row["delta_nmi"] is not None and row["delta_ari"] is not None and row["delta_f"] is not None
    ]


def rows_grouped(rows, type_name):
    return [row for row in rows if row["type"] == type_name and row["sdgca_nmi"] is not None and row["mod_nmi"] is not None]


def save_delta_barplot(rows):
    rows = rows_with_complete_metrics(rows)
    rows = sorted(rows, key=lambda x: (x["type"], x["delta_nmi"] + x["delta_ari"] + x["delta_f"]))
    labels = [row["dataset"] for row in rows]
    y = np.arange(len(labels))
    height = 0.24

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y - height, [row["delta_nmi"] for row in rows], height=height, label="Δ NMI")
    ax.barh(y, [row["delta_ari"] for row in rows], height=height, label="Δ ARI")
    ax.barh(y + height, [row["delta_f"] for row in rows], height=height, label="Δ F-score")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Прирост modified версии относительно базовой")
    ax.set_title("Выигрыш SDGCA modified по датасетам")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sdgca_modified_delta_by_dataset.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_delta_heatmap(rows):
    rows = rows_with_complete_metrics(rows)
    rows = sorted(rows, key=lambda x: (x["type"], -(x["delta_nmi"] + x["delta_ari"] + x["delta_f"])))
    labels = [f"{row['dataset']} ({'R' if row['type']=='real' else 'S'})" for row in rows]
    matrix = np.array([[row["delta_nmi"], row["delta_ari"], row["delta_f"]] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.6 * len(rows))))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Δ NMI", "Δ ARI", "Δ F-score"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Тепловая карта выигрышей SDGCA modified")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9, color="black")

    # visually separate real and synthetic blocks
    real_count = sum(1 for row in rows if row["type"] == "real")
    if 0 < real_count < len(rows):
        ax.axhline(real_count - 0.5, color="black", linewidth=1.5)
        ax.text(2.9, real_count / 2 - 0.5, "Реальные", rotation=90, va="center", ha="left", fontsize=10)
        ax.text(2.9, real_count + (len(rows) - real_count) / 2 - 0.5, "Синтетические", rotation=90, va="center", ha="left", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Прирост modified относительно baseline")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sdgca_delta_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_nmi_comparison(rows):
    rows = [row for row in rows if row["sdgca_nmi"] is not None and row["mod_nmi"] is not None]
    rows = sorted(rows, key=lambda x: (x["type"], x["dataset"]))
    labels = [row["dataset"] for row in rows]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, [row["sdgca_nmi"] for row in rows], width=width, label="SDGCA")
    ax.bar(x + width / 2, [row["mod_nmi"] for row in rows], width=width, label="SDGCA modified")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("NMI")
    ax.set_title("Сравнение SDGCA и SDGCA modified по метрике NMI")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sdgca_vs_modified_nmi.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_dumbbell_panels(rows):
    metric_specs = [
        ("nmi", "NMI"),
        ("ari", "ARI"),
        ("f", "F-score"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=False)
    type_titles = [("real", "Реальные датасеты"), ("synthetic", "Синтетические датасеты")]

    for col, (type_name, title) in enumerate(type_titles):
        subset = rows_grouped(rows, type_name)
        subset = sorted(subset, key=lambda x: x["mod_nmi"] - x["sdgca_nmi"], reverse=True)
        labels = [row["dataset"] for row in subset]
        y = np.arange(len(labels))
        for row_idx, (metric_key, metric_title) in enumerate(metric_specs):
            ax = axes[row_idx, col]
            base_key = f"sdgca_{metric_key}"
            mod_key = f"mod_{metric_key}"
            base_vals = [row[base_key] for row in subset]
            mod_vals = [row[mod_key] for row in subset]
            for yi, b, m in zip(y, base_vals, mod_vals):
                ax.plot([b, m], [yi, yi], color="#9aa0a6", linewidth=2, zorder=1)
            ax.scatter(base_vals, y, color="#1f77b4", s=55, label="SDGCA", zorder=2)
            ax.scatter(mod_vals, y, color="#d62728", s=55, label="SDGCA modified", zorder=3)
            ax.set_yticks(y)
            ax.set_yticklabels(labels if row_idx == 1 else labels)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.set_xlabel(metric_title)
            if row_idx == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(metric_title)
            xmin = min(base_vals + mod_vals) - 0.03
            xmax = max(base_vals + mod_vals) + 0.03
            ax.set_xlim(max(0, xmin), min(1.02, xmax))
            if row_idx == 0 and col == 1:
                ax.legend(loc="lower right")
    fig.suptitle("Попарное сравнение baseline и modified по метрикам", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sdgca_dumbbell_panels.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_imbalance_scatter(rows):
    rows = rows_with_complete_metrics(rows)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"real": "#1f77b4", "synthetic": "#d62728"}
    for row in rows:
        ax.scatter(
            row["imbalance_ratio"],
            row["delta_nmi"],
            color=colors.get(row["type"], "#333333"),
            s=70,
            alpha=0.85,
        )
        ax.annotate(row["dataset"], (row["imbalance_ratio"], row["delta_nmi"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент дисбаланса классов")
    ax.set_ylabel("Δ NMI")
    ax.set_title("Связь между дисбалансом классов и выигрышем modified версии")
    ax.grid(linestyle="--", alpha=0.35)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["real"], markersize=8, label="Реальные"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["synthetic"], markersize=8, label="Синтетические"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "delta_nmi_vs_imbalance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_std_errorbars_real(rows):
    rows = [
        row
        for row in rows_grouped(rows, "real")
        if row["sdgca_nmi_std"] is not None
        and row["mod_nmi_std"] is not None
        and row["sdgca_ari_std"] is not None
        and row["mod_ari_std"] is not None
        and row["sdgca_f_std"] is not None
        and row["mod_f_std"] is not None
    ]
    rows = sorted(rows, key=lambda x: x["dataset"])
    labels = [row["dataset"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    metric_specs = [
        ("nmi", "NMI"),
        ("ari", "ARI"),
        ("f", "F-score"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for ax, (metric_key, title) in zip(axes, metric_specs):
        base_vals = [row[f"sdgca_{metric_key}"] for row in rows]
        base_std = [row[f"sdgca_{metric_key}_std"] for row in rows]
        mod_vals = [row[f"mod_{metric_key}"] for row in rows]
        mod_std = [row[f"mod_{metric_key}_std"] for row in rows]

        ax.bar(x - width / 2, base_vals, width=width, yerr=base_std, capsize=4, label="SDGCA")
        ax.bar(x + width / 2, mod_vals, width=width, yerr=mod_std, capsize=4, label="SDGCA modified")
        ax.set_ylabel(title)
        ax.set_title(f"{title}: среднее значение и разброс на реальных датасетах")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sdgca_real_errorbars.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_objects_scatter(rows):
    rows = rows_with_complete_metrics(rows)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"real": "#1f77b4", "synthetic": "#2ca02c"}
    for row in rows:
        ax.scatter(
            row["objects"],
            row["delta_f"],
            color=colors.get(row["type"], "#333333"),
            s=70,
            alpha=0.85,
        )
        ax.annotate(row["dataset"], (row["objects"], row["delta_f"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Число объектов")
    ax.set_ylabel("Δ F-score")
    ax.set_title("Связь между размером датасета и выигрышем modified версии")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "delta_f_vs_objects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_profile_bubble(rows):
    rows = rows_with_complete_metrics(rows)
    fig, ax = plt.subplots(figsize=(11, 8))
    color_map = {"modified": "#2ca02c", "baseline": "#d62728", "tie": "#7f7f7f"}
    marker_map = {"real": "o", "synthetic": "s"}

    for row in rows:
        size = max(80, row["objects"] / 12)
        ax.scatter(
            row["imbalance_ratio"],
            row["delta_ari"],
            s=size,
            c=color_map.get(row["winner"], "#1f77b4"),
            marker=marker_map.get(row["type"], "o"),
            alpha=0.7,
            edgecolors="black",
            linewidths=0.6,
        )
        ax.annotate(row["dataset"], (row["imbalance_ratio"], row["delta_ari"]), fontsize=8, xytext=(5, 4), textcoords="offset points")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент дисбаланса классов")
    ax.set_ylabel("Δ ARI")
    ax.set_title("Профиль датасетов: дисбаланс, размер и выигрыш modified версии")
    ax.grid(linestyle="--", alpha=0.3)

    type_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markeredgecolor="black", label="Реальные", markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="lightgray", markeredgecolor="black", label="Синтетические", markersize=8),
    ]
    winner_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["modified"], label="Лучше modified", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["baseline"], label="Лучше baseline", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["tie"], label="Ничья", markersize=8),
    ]
    legend1 = ax.legend(handles=type_handles, loc="upper right", title="Тип данных")
    ax.add_artist(legend1)
    ax.legend(handles=winner_handles, loc="lower right", title="Исход сравнения")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sdgca_profile_bubble.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows(load_rows(INPUT_TSV))
    save_delta_barplot(rows)
    save_delta_heatmap(rows)
    save_nmi_comparison(rows)
    save_dumbbell_panels(rows)
    save_imbalance_scatter(rows)
    save_std_errorbars_real(rows)
    save_objects_scatter(rows)
    save_profile_bubble(rows)
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
