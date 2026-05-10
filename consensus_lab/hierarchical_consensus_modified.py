"""Взвешенная иерархическая консенсус-кластеризация (вклад работы).

Отличия от `hierarchical_consensus.py`:
    1. Каждой базовой кластеризации присваивается вес, равный её
       среднему парному согласию с остальными разбиениями ансамбля
       (см. `compute_base_clustering_weights`). Идея: «согласные с
       большинством» базовые разбиения вносят больший вклад в финальную
       co-association матрицу, шумные/случайные — меньший.
    2. Параметр `sharpen` контролирует «жёсткость» весовой схемы:
       `sharpen > 1` усиливает контраст между весами, `sharpen < 1`
       выравнивает их.

Подсчёт парного согласия выполнен через `metrics.contingency`, что даёт
сложность `O(m² · n)` вместо наивных `O(m² · n²)` через построение
n × n индикаторных матриц.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from hierarchical_consensus import (
    SUPPORTED_LINKAGE_METHODS,
    validate_members,
)
from metrics import contingency


__all__ = [
    "build_partition_matrix",
    "build_weighted_consensus_matrix",
    "compute_base_clustering_weights",
    "partition_agreement",
    "run_weighted_hierarchical_consensus",
]


def build_partition_matrix(labels: np.ndarray) -> np.ndarray:
    """Индикаторная `n × n` матрица «лежат ли объекты в одном кластере».

    Args:
        labels: вектор меток длины n.

    Returns:
        `(n, n)` float64 матрица из `0/1`.
    """
    labels = np.asarray(labels).reshape(-1)
    return (labels[:, None] == labels[None, :]).astype(np.float64)


def partition_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Парное согласие двух разбиений (Rand-подобный коэффициент).

    Считается как доля пар объектов, которые два разбиения «одинаково»
    относят (либо вместе в одном кластере, либо порознь). Эквивалент
    «unadjusted Rand index», но вычислен через `contingency` —
    `O(n)` память вместо `O(n²)`, важно для больших n.

    Args:
        labels_a: вектор меток первого разбиения, длина n.
        labels_b: вектор меток второго разбиения, длина n.

    Returns:
        Согласие в `[0, 1]`. Полное совпадение → 1, разные структуры → ниже.
    """
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    cont = contingency(labels_a, labels_b).astype(np.float64)
    n = labels_a.size
    row_sums = np.sum(cont, axis=1)
    col_sums = np.sum(cont, axis=0)
    same_a = float(np.sum(row_sums**2))
    same_b = float(np.sum(col_sums**2))
    both_same = float(np.sum(cont**2))
    return float((n**2 - same_a - same_b + 2.0 * both_same) / n**2)


def compute_base_clustering_weights(base_cls: np.ndarray) -> np.ndarray:
    """Веса базовых кластеризаций по средневзвешенному согласию.

    Для каждой кластеризации `i` считается её среднее парное согласие
    со всеми кластеризациями `j` (включая саму себя). Полученные m
    значений нормируются в распределение, сумма которого равна 1.

    Args:
        base_cls: `(n, m)` матрица меток.

    Returns:
        `(m,)` float64 вектор весов, сумма = 1. При вырожденном случае
        (все веса нулевые) возвращается равномерное распределение `1/m`.
    """
    base_cls = validate_members(base_cls)
    m = base_cls.shape[1]
    agreement = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            score = partition_agreement(base_cls[:, i], base_cls[:, j])
            agreement[i, j] = score
            agreement[j, i] = score
    weights = np.mean(agreement, axis=1)
    weights_sum = float(np.sum(weights))
    if weights_sum == 0.0:
        return np.full(m, 1.0 / m, dtype=np.float64)
    return weights / weights_sum


def build_weighted_consensus_matrix(
    base_cls: np.ndarray, sharpen: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Взвешенная co-association матрица.

    Args:
        base_cls: `(n, m)` матрица меток.
        sharpen: степень «заострения» весов (`>0`). 1.0 — без изменений,
            `>1` усиливает контраст, `<1` выравнивает.

    Returns:
        Кортеж `(consensus, weights)`:
            `consensus` — `(n, n)` симметричная матрица в `[0, 1]`,
                диагональ = 1.
            `weights` — `(m,)` итоговые (после `sharpen` и нормализации)
                веса базовых кластеризаций.

    Raises:
        ValueError: если `sharpen <= 0`.
    """
    base_cls = validate_members(base_cls)
    if sharpen <= 0:
        raise ValueError("sharpen must be positive")
    n, m = base_cls.shape
    weights = compute_base_clustering_weights(base_cls)
    if sharpen != 1.0:
        weights = weights**sharpen
        weights_sum = float(np.sum(weights))
        weights = (
            np.full(m, 1.0 / m, dtype=np.float64)
            if weights_sum == 0.0
            else weights / weights_sum
        )
    consensus = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        consensus += weights[j] * build_partition_matrix(base_cls[:, j])
    consensus = (consensus + consensus.T) / 2.0
    consensus = np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus, weights


def run_weighted_hierarchical_consensus(
    dataset_path: str | Path,
    data_name: str | None = None,
    seed: int = 19,
    m: int = 40,
    cnt_times: int = 20,
    method: str = "average",
    sharpen: float = 1.0,
    selection_strategy: str = "random",
    qd_alpha: float = 0.5,
) -> dict:
    """Прогнать взвешенную иерархическую консенсус-кластеризацию.

    Тот же протокол, что и в `run_hierarchical_consensus`, но вместо
    обычной co-association матрицы используется взвешенная.
    """
    from consensus_runner import run_consensus_loop

    if sharpen <= 0:
        raise ValueError("sharpen must be positive")

    weight_bank: list[np.ndarray] = []

    def _build(base_cls, _gt, _m):
        consensus, weights = build_weighted_consensus_matrix(base_cls, sharpen=sharpen)
        weight_bank.append(weights)
        return consensus

    result = run_consensus_loop(
        dataset_path, _build,
        data_name=data_name, seed=seed, m=m, cnt_times=cnt_times,
        method=method, selection_strategy=selection_strategy, qd_alpha=qd_alpha,
    )
    result["sharpen"] = float(sharpen)
    result["avg_weights"] = np.mean(np.vstack(weight_bank), axis=0).astype(float).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Взвешенная иерархическая консенсус-кластеризация."
    )
    parser.add_argument("--dataset", default="Ecoli")
    parser.add_argument(
        "--root", default=Path(__file__).resolve().parents[1] / "datasets"
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--method", default="average", choices=sorted(SUPPORTED_LINKAGE_METHODS)
    )
    parser.add_argument("--sharpen", type=float, default=1.0)
    args = parser.parse_args()
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_weighted_hierarchical_consensus(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        method=args.method,
        sharpen=args.sharpen,
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
