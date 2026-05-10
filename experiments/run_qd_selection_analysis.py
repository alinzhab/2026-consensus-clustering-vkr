"""Сравнение random vs QD-selection для алгоритмов консенсус кластеризации.

Научная логика:
    Базовые алгоритмы консенсус кластеризации (HC, SDGCA) случайно выбирают m
    разбиений из пула. Случайный выбор может:
    (а) включать дублирующие разбиения, смещающие co-association матрицу;
    (б) включать нестабильные разбиения, добавляющие шум;
    (в) пропускать информативно разнообразные разбиения.

    QD-selection (Quality-Diversity) жадно выбирает m разбиений, балансируя
    качество (согласованность с большинством) и разнообразие (непохожесть
    на уже выбранные). Гипотеза: при достаточно богатом пуле QD-selection
    формирует более информативный субансамбль.

Эксперимент:
    Для каждого датасета и каждого алгоритма запускаем:
    - {algo}_random: стандартная случайная субдискретизация
    - {algo}_qd:     QD-selection с qd_alpha=0.5

    Метрики: NMI, ARI, F-score, runtime (секунды).
    Результаты: results/qd_selection_analysis.tsv
                results/qd_selection_summary.md

Корректные выводы по эксперименту:
    МОЖНО утверждать:
    - «QD-selection улучшило NMI на X датасетах из Y»
    - «На датасетах типа elongated/mixed_complex QD даёт стабильный прирост»
    - «Разность ΔNMI = QD − random статистически значима по Вилкоксону»
    НЕЛЬЗЯ утверждать без дополнительных тестов:
    - «QD-selection универсально лучше random»
    - «Прирост обусловлен балансом quality/diversity, а не просто детерминизмом»

Запуск:
    python experiments/run_qd_selection_analysis.py               # полный набор
    python experiments/run_qd_selection_analysis.py --smoke       # быстрый тест
    python experiments/run_qd_selection_analysis.py --m 20 --runs 10 --methods average
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


# ── пути ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSENSUS_LAB = PROJECT_ROOT / "consensus_lab"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(CONSENSUS_LAB))

from ensemble_selection import get_dataset_type
from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from sdgca import run_sdgca
from sdgca_modified import run_sdgca_modified


# ── константы ─────────────────────────────────────────────────────────────────

# Датасеты с аннотацией типа (для поканального анализа)
FULL_DATASETS: list[tuple[str, str]] = [
    # Реальные биологические
    ("Ecoli", ".mat"),
    ("GLIOMA", ".mat"),
    ("Lung", ".mat"),
    # Реальные текстовые / высокоразмерные
    ("BBC", ".mat"),
    ("orlraws10P", ".mat"),
    # Синтетические компактные
    ("Aggregation", ".mat"),
    ("densired_compact_hard", ".npz"),
    # Синтетические вытянутые
    ("densired_stretched_hard", ".npz"),
    # Синтетические сложные
    ("densired_mix_hard", ".npz"),
    # Repliclust разнообразные
    ("repliclust_heterogeneous_hard", ".npz"),
    ("repliclust_highdim_hard", ".npz"),
]

# Smoke-test: только 2 датасета, 1 run, минимальные параметры
SMOKE_DATASETS: list[tuple[str, str]] = [
    ("Ecoli", ".mat"),
    ("densired_compact_hard", ".npz"),
]

# Алгоритмы: (имя, run-функция, принимает ли selection_strategy?)
ALGORITHMS = [
    ("hierarchical_baseline", run_hierarchical_consensus),
    ("hierarchical_weighted", run_weighted_hierarchical_consensus),
    ("sdgca", run_sdgca),
    ("sdgca_modified", run_sdgca_modified),
]

TSV_COLUMNS = [
    "dataset",
    "dataset_type",
    "algorithm",
    "selection_strategy",
    "qd_alpha",
    "linkage",
    "m",
    "runs",
    "seed",
    "nmi_mean",
    "nmi_std",
    "ari_mean",
    "ari_std",
    "f_mean",
    "f_std",
    "runtime_sec",
]


# ── вспомогательные функции ──────────────────────────────────────────────────

def _find_dataset(name: str, suffix: str) -> Path | None:
    """Искать датасет в datasets/ и datasets/uploaded/."""
    for d in [DATASETS_DIR, DATASETS_DIR / "uploaded"]:
        p = d / f"{name}{suffix}"
        if p.exists():
            return p
    return None


def _run_one(
    algo_name: str,
    run_fn,
    dataset_path: Path,
    dataset_name: str,
    strategy: str,
    qd_alpha: float,
    m: int,
    runs: int,
    method: str,
    seed: int,
) -> dict | None:
    """Один запуск алгоритма — возвращает словарь с метриками или None при ошибке."""
    kwargs: dict = dict(
        dataset_path=dataset_path,
        data_name=dataset_name,
        seed=seed,
        m=m,
        cnt_times=runs,
        method=method,
        selection_strategy=strategy,
        qd_alpha=qd_alpha,
    )
    # sdgca_modified требует diffusion_time, используем дефолт 1.0
    if algo_name == "sdgca_modified":
        kwargs["diffusion_time"] = 1.0

    t0 = time.perf_counter()
    try:
        result = run_fn(**kwargs)
    except Exception as exc:
        print(f"  [SKIP] {algo_name}/{strategy} на {dataset_name}: {exc}", flush=True)
        return None
    elapsed = time.perf_counter() - t0

    return {
        "dataset": dataset_name,
        "dataset_type": get_dataset_type(dataset_name),
        "algorithm": algo_name,
        "selection_strategy": strategy,
        "qd_alpha": qd_alpha if strategy == "qd" else "",
        "linkage": method,
        "m": m,
        "runs": runs,
        "seed": seed,
        "nmi_mean": round(result["nmi_mean"], 6),
        "nmi_std": round(result["nmi_std"], 6),
        "ari_mean": round(result["ari_mean"], 6),
        "ari_std": round(result["ari_std"], 6),
        "f_mean": round(result["f_mean"], 6),
        "f_std": round(result["f_std"], 6),
        "runtime_sec": round(elapsed, 3),
    }


def _write_tsv(rows: list[dict], path: Path) -> None:
    """Добавить строки в TSV-файл (создаёт заголовок при первом запуске)."""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        if write_header:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in TSV_COLUMNS) + "\n")


def _write_summary(rows: list[dict], path: Path, args) -> None:
    """Сгенерировать Markdown-отчёт по результатам эксперимента."""
    from collections import defaultdict

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Группируем по алгоритму + стратегии: среднее NMI по всем датасетам
    group_nmi: dict[str, list[float]] = defaultdict(list)
    group_delta: dict[str, list[float]] = defaultdict(list)  # qd-random для каждого датасета

    pairs: dict[tuple[str, str], dict[str, float]] = {}  # (algo, dataset) → {random, qd}
    for row in rows:
        key = (row["algorithm"], row["dataset"])
        if key not in pairs:
            pairs[key] = {}
        pairs[key][row["selection_strategy"]] = row["nmi_mean"]

    for (algo, ds), strategies in pairs.items():
        if "random" in strategies and "qd" in strategies:
            delta = strategies["qd"] - strategies["random"]
            group_delta[algo].append(delta)

    for row in rows:
        group_nmi[f"{row['algorithm']}_{row['selection_strategy']}"].append(row["nmi_mean"])

    lines = [
        f"# QD-selection vs Random — Отчёт",
        f"",
        f"Сгенерировано: {ts}  ",
        f"Параметры: `m={args.m}`, `runs={args.runs}`, `methods={args.methods}`, `seed={args.seed}`",
        f"",
        f"## Научный контекст",
        f"",
        f"Стандартные алгоритмы консенсус кластеризации случайно выбирают `m` базовых кластеризаций",
        f"из пула. QD-selection (Quality-Diversity) использует жадный алгоритм, балансирующий:",
        f"- **Качество** кластеризации — согласованность с большинством ансамбля;",
        f"- **Разнообразие** — непохожесть на уже выбранные кластеризации.",
        f"",
        f"## Среднее NMI по всем датасетам",
        f"",
        f"| Алгоритм + стратегия | Среднее NMI | N датасетов |",
        f"|---|---|---|",
    ]
    for key, vals in sorted(group_nmi.items()):
        mean_nmi = sum(vals) / len(vals)
        lines.append(f"| {key} | {mean_nmi:.4f} | {len(vals)} |")

    lines += [
        f"",
        f"## ΔNMI = QD − Random (по алгоритмам)",
        f"",
        f"| Алгоритм | Среднее ΔNMI | Побед QD | Поражений QD | Ничья |",
        f"|---|---|---|---|---|",
    ]
    for algo, deltas in sorted(group_delta.items()):
        mean_d = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0.002)
        losses = sum(1 for d in deltas if d < -0.002)
        draws = len(deltas) - wins - losses
        lines.append(f"| {algo} | {mean_d:+.4f} | {wins} | {losses} | {draws} |")

    lines += [
        f"",
        f"## Результаты по типам данных",
        f"",
        f"| Тип данных | Алгоритм | ΔNMI (QD−random) |",
        f"|---|---|---|",
    ]
    type_delta: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (algo, ds), strategies in pairs.items():
        if "random" in strategies and "qd" in strategies:
            ds_type = get_dataset_type(ds)
            type_delta[(ds_type, algo)].append(strategies["qd"] - strategies["random"])
    for (ds_type, algo), deltas in sorted(type_delta.items()):
        mean_d = sum(deltas) / len(deltas)
        lines.append(f"| {ds_type} | {algo} | {mean_d:+.4f} |")

    lines += [
        f"",
        f"## Полная таблица результатов",
        f"",
        f"| Датасет | Тип | Алгоритм | Стратегия | NMI | ARI | F-score | Время (с) |",
        f"|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["dataset"], r["algorithm"], r["selection_strategy"])):
        lines.append(
            f"| {row['dataset']} | {row['dataset_type']} | {row['algorithm']} "
            f"| {row['selection_strategy']} "
            f"| {row['nmi_mean']:.4f}±{row['nmi_std']:.4f} "
            f"| {row['ari_mean']:.4f} "
            f"| {row['f_mean']:.4f} "
            f"| {row['runtime_sec']:.1f} |"
        )

    lines += [
        f"",
        f"## Ограничения интерпретации",
        f"",
        f"- При `selection_strategy=qd` все `runs` прогонов используют одно и то же",
        f"  подмножество кластеризаций → std метрик отражает детерминированность",
        f"  алгоритма, а не дисперсию субдискретизации. Сравнение std(random) vs std(qd)",
        f"  показывает, насколько случайный выбор нестабилен.",
        f"- Результаты получены при `m={args.m}` и фиксированном `seed={args.seed}`.",
        f"  Для статистически значимых выводов необходимо повторить с 5 независимыми seed.",
        f"- Улучшение NMI может быть частично обусловлено детерминизмом QD, а не",
        f"  собственно балансом quality/diversity. Это разделяется в аблационном анализе",
        f"  (сравнение с `qd_alpha=1.0` / `qd_alpha=0.0`).",
        f"",
        f"_Файл сгенерирован автоматически скриптом `experiments/run_qd_selection_analysis.py`_",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── основная логика ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение random vs QD-selection для алгоритмов консенсус кластеризации."
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Быстрый smoke-test: 2 датасета, 1 прогон, только average."
    )
    parser.add_argument("--m", type=int, default=20, help="Размер подансамбля (default: 20).")
    parser.add_argument("--runs", type=int, default=10, help="Число прогонов (default: 10).")
    parser.add_argument(
        "--methods", nargs="+", default=["average", "complete", "single", "ward"],
        choices=["average", "complete", "single", "ward"],
        help="Правила иерархической агрегации."
    )
    parser.add_argument("--seed", type=int, default=19, help="Seed для воспроизводимости.")
    parser.add_argument("--qd_alpha", type=float, default=0.5,
                        help="Параметр QD: вес качества (0=только разнообразие, 1=только качество).")
    parser.add_argument(
        "--algos", nargs="+",
        default=["hierarchical_baseline", "hierarchical_weighted", "sdgca", "sdgca_modified"],
        choices=["hierarchical_baseline", "hierarchical_weighted", "sdgca", "sdgca_modified"],
        help="Выбранные алгоритмы."
    )
    args = parser.parse_args()

    # ── параметры smoke / full ────────────────────────────────────────────────
    if args.smoke:
        datasets = SMOKE_DATASETS
        methods = ["average"]
        runs = 1
        m = 10
        print("=" * 60)
        print("SMOKE-TEST: QD-selection vs Random")
        print("=" * 60)
    else:
        datasets = FULL_DATASETS
        methods = args.methods
        runs = args.runs
        m = args.m
        print("=" * 60)
        print("QD-selection vs Random: полный эксперимент")
        print(f"  датасетов: {len(datasets)}, алгоритмов: {len(args.algos)}")
        print(f"  m={m}, runs={runs}, methods={methods}, qd_alpha={args.qd_alpha}")
        print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = RESULTS_DIR / "qd_selection_analysis.tsv"
    summary_path = RESULTS_DIR / "qd_selection_summary.md"

    # Фильтрация алгоритмов по --algos
    selected_algos = [(n, fn) for (n, fn) in ALGORITHMS if n in args.algos]

    all_rows: list[dict] = []
    total = len(datasets) * len(selected_algos) * 2 * len(methods)
    done = 0

    for ds_name, ds_suffix in datasets:
        ds_path = _find_dataset(ds_name, ds_suffix)
        if ds_path is None:
            print(f"  [MISS] Датасет не найден: {ds_name}{ds_suffix}", flush=True)
            continue

        ds_type = get_dataset_type(ds_name)
        print(f"\n[DATASET] {ds_name} ({ds_type})")

        for algo_name, run_fn in selected_algos:
            for method in methods:
                for strategy in ("random", "qd"):
                    done += 1
                    tag = f"{algo_name}/{strategy}/{method}"
                    print(f"  [{done}/{total}] {tag}", end=" ... ", flush=True)
                    row = _run_one(
                        algo_name=algo_name,
                        run_fn=run_fn,
                        dataset_path=ds_path,
                        dataset_name=ds_name,
                        strategy=strategy,
                        qd_alpha=args.qd_alpha,
                        m=m,
                        runs=runs,
                        method=method,
                        seed=args.seed,
                    )
                    if row is not None:
                        nmi_str = f"NMI={row['nmi_mean']:.4f}+/-{row['nmi_std']:.4f}"
                        t_str = f"{row['runtime_sec']:.1f}s"
                        print(f"{nmi_str}  {t_str}", flush=True)
                        all_rows.append(row)

    # Сохранение
    if all_rows:
        _write_tsv(all_rows, tsv_path)
        _write_summary(all_rows, summary_path, args if not args.smoke else
                       type("A", (), {"m": m, "runs": runs, "methods": methods,
                                      "seed": args.seed})())
        print(f"\n[OK] TSV saved:     {tsv_path}")
        print(f"[OK] Report saved:  {summary_path}")
        print(f"[OK] Rows written:  {len(all_rows)}")
    else:
        print("\n[WARN] Нет результатов для сохранения.")

    # Краткий итог в stdout
    if all_rows:
        from collections import defaultdict
        delta_by_algo: dict[str, list[float]] = defaultdict(list)
        pairs: dict[tuple[str, str, str], dict[str, float]] = {}
        for row in all_rows:
            key = (row["algorithm"], row["dataset"], row["linkage"])
            if key not in pairs:
                pairs[key] = {}
            pairs[key][row["selection_strategy"]] = row["nmi_mean"]
        for (algo, ds, m_), strategies in pairs.items():
            if "random" in strategies and "qd" in strategies:
                delta_by_algo[algo].append(strategies["qd"] - strategies["random"])

        print("\n-- Summary: delta NMI = QD - Random ----------------------")
        for algo, deltas in sorted(delta_by_algo.items()):
            mean_d = sum(deltas) / len(deltas)
            wins = sum(1 for d in deltas if d > 0.002)
            losses = sum(1 for d in deltas if d < -0.002)
            print(f"  {algo:<28}: avg delta NMI={mean_d:+.4f}  "
                  f"(up={wins} down={losses} same={len(deltas)-wins-losses})")
        print("---------------------------------------------------------")


if __name__ == "__main__":
    main()
