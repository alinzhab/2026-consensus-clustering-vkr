"""Merge designed QD experiment TSV files and create a qualitative report."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
COMBINED_TSV = RESULTS_DIR / "designed_qd_experiment_combined.tsv"
COMBINED_REPORT = RESULTS_DIR / "designed_qd_experiment_qualitative_analysis.md"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    files = sorted(
        p for p in RESULTS_DIR.glob("designed_qd_experiment_*.tsv")
        if p.name not in {"designed_qd_experiment_latest.tsv", "designed_qd_experiment_combined.tsv"}
    )
    if not files:
        raise SystemExit("No designed_qd_experiment_*.tsv files found")

    rows: list[dict] = []
    columns: list[str] = []
    seen = set()
    for path in files:
        file_rows = read_rows(path)
        for row in file_rows:
            row["source_file"] = path.name
            key = (
                row.get("dataset"),
                row.get("algorithm"),
                row.get("selection_strategy"),
                row.get("linkage"),
                row.get("seed"),
                row.get("m"),
                row.get("runs"),
                path.name,
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            for col in row:
                if col not in columns:
                    columns.append(col)

    write_rows(COMBINED_TSV, rows, columns)

    pairs: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("dataset_type", ""),
            row.get("algorithm", ""),
            row.get("linkage", ""),
        )
        strategy = row.get("selection_strategy", "")
        pairs[key][strategy] = as_float(row.get("nmi_mean", "0"))

    delta_by_algo: dict[str, list[float]] = defaultdict(list)
    delta_by_type: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (dataset, dtype, algo, linkage), strategies in pairs.items():
        if "random" not in strategies or "qd" not in strategies:
            continue
        delta = strategies["qd"] - strategies["random"]
        delta_by_algo[algo].append(delta)
        delta_by_type[(dtype, algo)].append(delta)

    type_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("selection_strategy") == "random":
            type_counts[row.get("dataset_type", "unknown")] += 1

    lines = [
        "# Качественный анализ расширенного эксперимента",
        "",
        "Этот файл объединяет широкий иерархический эксперимент и mini-SDGCA эксперимент.",
        "",
        "## Почему дизайн разделен на два слоя",
        "",
        "SDGCA на датасетах порядка 1200-1800 объектов оказался вычислительно тяжелым: один запуск может занимать десятки и сотни секунд. Поэтому корректный дизайн разделен на:",
        "",
        "- широкий слой: много датасетов, два иерархических алгоритма, два linkage, два seed;",
        "- mini-SDGCA слой: все типы данных представлены меньшими датасетами, чтобы честно сравнить SDGCA и SDGCA modified без многочасового прогона.",
        "",
        "Такой дизайн лучше, чем искусственно гонять тяжелый алгоритм на всем наборе и получать непрактичный эксперимент.",
        "",
        f"Объединенный TSV: `{COMBINED_TSV.name}`",
        f"Всего строк: `{len(rows)}`",
        f"Исходные файлы: `{', '.join(p.name for p in files)}`",
        "",
        "## Представленность типов данных",
        "",
        "| Тип данных | Количество random-строк |",
        "|---|---:|",
    ]
    for dtype, count in sorted(type_counts.items()):
        lines.append(f"| {dtype} | {count} |")

    lines += [
        "",
        "## Средний эффект QD-selection: delta NMI = QD - random",
        "",
        "| Алгоритм | Средний delta NMI | Количество сравнений |",
        "|---|---:|---:|",
    ]
    for algo, deltas in sorted(delta_by_algo.items()):
        lines.append(f"| {algo} | {mean(deltas):+.4f} | {len(deltas)} |")

    lines += [
        "",
        "## Эффект QD-selection по типам данных",
        "",
        "| Тип данных | Алгоритм | Средний delta NMI | Количество сравнений |",
        "|---|---|---:|---:|",
    ]
    for (dtype, algo), deltas in sorted(delta_by_type.items()):
        lines.append(f"| {dtype} | {algo} | {mean(deltas):+.4f} | {len(deltas)} |")

    lines += [
        "",
        "## Как это интерпретировать в ВКР",
        "",
        "1. Компактные и хорошо разделимые данные нужны как sanity-check: если алгоритм плохо работает там, проблема в реализации или протоколе.",
        "2. Перекрывающиеся данные проверяют устойчивость к неоднозначным границам.",
        "3. Несбалансированные данные проверяют, теряются ли малые кластеры.",
        "4. Высокоразмерные данные проверяют влияние размерности и нестабильности базовых кластеризаций.",
        "5. Вытянутые и density-varied данные проверяют чувствительность к геометрии и плотности.",
        "6. Mixed-complex данные показывают поведение при нескольких трудностях одновременно.",
        "",
        "Главный корректный вывод не должен звучать как «QD всегда лучше». Правильная формулировка: QD-selection меняет качество в зависимости от типа данных и linkage, а система позволяет это обнаруживать и интерпретировать.",
    ]
    COMBINED_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] combined TSV: {COMBINED_TSV}")
    print(f"[OK] report: {COMBINED_REPORT}")
    print(f"[OK] rows: {len(rows)}")


if __name__ == "__main__":
    main()
