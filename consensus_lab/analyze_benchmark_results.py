from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DATASET_GROUPS = {
    "Aggregation": "real_geometric",
    "BBC": "real_text",
    "Ecoli": "real_biological",
    "GLIOMA": "real_biological",
    "Lung": "real_biological",
    "orlraws10P": "real_images",
    "custom_densired_dataset": "synthetic_density",
    "densired_compact_hard": "synthetic_density",
    "densired_mix_hard": "synthetic_density",
    "densired_stretched_hard": "synthetic_density",
    "repliclust_heterogeneous_hard": "synthetic_archetype",
    "repliclust_highdim_hard": "synthetic_archetype",
    "repliclust_oblong_overlap": "synthetic_archetype",
}


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(row: dict, key: str) -> float | None:
    try:
        value = row.get(key, "")
        return None if value == "" else float(value)
    except ValueError:
        return None


def normalize_rows(rows: list[dict], source: str) -> list[dict]:
    normalized = []
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        if row.get("runs") not in {"", "5"}:
            continue
        nmi = as_float(row, "nmi_mean")
        ari = as_float(row, "ari_mean")
        f_score = as_float(row, "f_mean")
        if nmi is None or ari is None or f_score is None:
            continue
        normalized.append(
            {
                "dataset": row["dataset"],
                "group": DATASET_GROUPS.get(row["dataset"], "other"),
                "algorithm": row.get("algorithm") or row.get("version") or row.get("mode"),
                "method": row.get("method", "average") or "average",
                "nmi": nmi,
                "ari": ari,
                "f_score": f_score,
                "source": source,
            }
        )
    return normalized


def best_by_key(rows: list[dict], key_fields: tuple[str, ...], metric: str) -> dict[tuple, dict]:
    best = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key not in best or row[metric] > best[key][metric]:
            best[key] = row
    return best


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_row(row: dict) -> str:
    return f"{row['algorithm']} / {row['method']} (NMI={row['nmi']:.3f}, ARI={row['ari']:.3f}, F={row['f_score']:.3f})"


def main() -> None:
    full = normalize_rows(read_tsv(RESULTS_DIR / "full_benchmark.tsv"), "full_benchmark")
    sdgca_linkage = normalize_rows(read_tsv(RESULTS_DIR / "sdgca_linkage_full_suite.tsv"), "sdgca_linkage")
    hierarchical_linkage = normalize_rows(read_tsv(RESULTS_DIR / "hierarchical_linkage_full_suite.tsv"), "hierarchical_linkage")

    average_rows = [row for row in full if row["method"] == "average"]
    linkage_rows = sdgca_linkage + hierarchical_linkage
    all_rows = average_rows + linkage_rows

    report = []
    report.append("# Benchmark Analysis Report")
    report.append("")
    report.append("Параметры: seed=19, m=20, runs=5. Метрики: NMI, ARI, F-score.")
    report.append("")
    report.append("## Coverage")
    report.append("")
    report.append(f"- Average-сравнение всех алгоритмов: {len(average_rows)} строк.")
    report.append(f"- Linkage-сравнение SDGCA/SDGCA modified: {len(sdgca_linkage)} строк.")
    report.append(f"- Linkage-сравнение hierarchical baseline/weighted: {len(hierarchical_linkage)} строк.")
    report.append("")

    report.append("## Best Algorithm Per Dataset By NMI")
    report.append("")
    report.append("| Dataset | Group | Best configuration |")
    report.append("|---|---|---|")
    for (dataset,), row in sorted(best_by_key(all_rows, ("dataset",), "nmi").items()):
        report.append(f"| {dataset} | {row['group']} | {format_row(row)} |")
    report.append("")

    report.append("## Best Linkage Per Algorithm By Dataset")
    report.append("")
    report.append("| Dataset | Algorithm | Best linkage | NMI | ARI | F-score |")
    report.append("|---|---|---|---:|---:|---:|")
    for (dataset, algorithm), row in sorted(best_by_key(linkage_rows, ("dataset", "algorithm"), "nmi").items()):
        report.append(
            f"| {dataset} | {algorithm} | {row['method']} | {row['nmi']:.3f} | {row['ari']:.3f} | {row['f_score']:.3f} |"
        )
    report.append("")

    report.append("## Mean NMI By Algorithm And Linkage")
    report.append("")
    report.append("| Algorithm | Linkage | Mean NMI | Datasets |")
    report.append("|---|---|---:|---:|")
    grouped = defaultdict(list)
    for row in linkage_rows:
        grouped[(row["algorithm"], row["method"])].append(row["nmi"])
    for (algorithm, method), values in sorted(grouped.items()):
        report.append(f"| {algorithm} | {method} | {mean(values):.3f} | {len(values)} |")
    report.append("")

    report.append("## Linkage Win Counts")
    report.append("")
    winners = best_by_key(linkage_rows, ("dataset", "algorithm"), "nmi")
    counts = Counter(row["method"] for row in winners.values())
    for method, count in counts.most_common():
        report.append(f"- {method}: {count}")
    report.append("")

    report.append("## Algorithm Win Counts")
    report.append("")
    winners_dataset = best_by_key(all_rows, ("dataset",), "nmi")
    algorithm_counts = Counter(row["algorithm"] for row in winners_dataset.values())
    for algorithm, count in algorithm_counts.most_common():
        report.append(f"- {algorithm}: {count}")
    report.append("")

    report.append("## Notes")
    report.append("")
    report.append("- `single` часто нестабилен на текстовых и части плотностных данных из-за chain effect.")
    report.append("- `ward` хорошо проявляется на компактных/геометрически выраженных кластерах, но не всегда подходит для consensus-матриц.")
    report.append("- `average` обычно наиболее устойчивый базовый выбор, особенно когда заранее неизвестна структура данных.")
    report.append("- Взвешенная иерархическая версия полезна, когда качество базовых разбиений неоднородно.")
    report.append("- SDGCA даёт сильный прирост на части реальных датасетов, но тяжелее вычислительно.")

    output = RESULTS_DIR / "analysis_report.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
