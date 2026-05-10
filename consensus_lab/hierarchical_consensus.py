"""Базовая иерархическая консенсус-кластеризация.

Алгоритм:
    1. Строится `n × n` матрица совместной встречаемости (co-association)
       по ансамблю базовых кластеризаций.
    2. Из неё получается матрица расстояний `1 - CA`.
    3. Применяется `scipy.cluster.hierarchy.linkage` с одним из правил
       агрегации (`single`, `complete`, `average`, `ward`).
    4. Дерево обрезается до `K` кластеров (`K` берётся из числа классов
       в `gt`, чтобы сравнение метрик было корректным).
    5. Считаются NMI/ARI/F-score по `metrics.py`.

Поддерживаются два формата датасетов: `.mat` (MATLAB) и `.npz` (numpy).

Файл намеренно не содержит CLI-логики кроме `main()`, чтобы все функции
можно было переиспользовать из `algorithms_base.AlgorithmRegistry`,
веб-приложения и из `experiments/`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.io import loadmat
from scipy.spatial.distance import squareform

from metrics import compute_ari, compute_nmi, compute_pairwise_f_score  # noqa: F401 — re-exported


SUPPORTED_LINKAGE_METHODS: frozenset[str] = frozenset(
    {"average", "complete", "single", "ward"}
)


__all__ = [
    "SUPPORTED_LINKAGE_METHODS",
    "build_coassociation_matrix",
    "get_cls_result",
    "load_dataset",
    "load_dataset_full",
    "run_hierarchical_consensus",
    "validate_gt",
    "validate_members",
    "validate_method",
]


def load_dataset(dataset_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузить ансамбль базовых кластеризаций и эталонные метки.

    Args:
        dataset_path: путь к файлу `.mat` или `.npz`.

    Returns:
        Кортеж `(members, gt)`, где
            `members` — `(n, m)` int64 матрица меток (m базовых кластеризаций);
            `gt` — `(n,)` int64 вектор истинных меток (нумерация с 1).

    Raises:
        ValueError: формат файла не поддерживается, либо число строк в
            `members` не совпадает с длиной `gt`.
    """
    dataset_path = Path(dataset_path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".mat":
        data = loadmat(dataset_path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
    elif suffix == ".npz":
        data = np.load(dataset_path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")
    if gt.size > 0 and gt.min() == 0:
        gt = gt + 1
    if members.shape[0] != gt.shape[0]:
        raise ValueError("members row count must match gt length")
    return members, gt


def load_dataset_full(
    dataset_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Загрузить `(members, gt, X)` из `.mat` или `.npz`.

    Расширенная версия `load_dataset`, дополнительно возвращающая матрицу
    признаков `X` (если она есть в файле).
    """
    dataset_path = Path(dataset_path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".mat":
        data = loadmat(dataset_path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        x: np.ndarray | None = None
        for key in ("X", "x", "data", "fea", "features"):
            if key in data:
                x = np.asarray(data[key], dtype=np.float64)
                break
    elif suffix == ".npz":
        data = np.load(dataset_path, allow_pickle=True)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        x = np.asarray(data["X"], dtype=np.float64) if "X" in data.files else None
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")
    if gt.size > 0 and gt.min() == 0:
        gt = gt + 1
    return members, gt, x


def validate_members(members: np.ndarray, m: int | None = None) -> np.ndarray:
    """Проверить и привести матрицу базовых кластеризаций к int64.

    Args:
        members: `(n, m)` матрица целочисленных меток.
        m: если задано — желаемый размер ансамбля; должен быть в
            диапазоне `[1, members.shape[1]]`.

    Returns:
        Та же матрица `dtype=np.int64`.

    Raises:
        ValueError: при некорректной форме, нечисловых значениях или
            недопустимом значении `m`.
    """
    members = np.asarray(members)
    if members.ndim != 2:
        raise ValueError("members must be a 2D matrix")
    if members.shape[0] < 2:
        raise ValueError("dataset must contain at least two objects")
    if members.shape[1] < 1:
        raise ValueError("dataset must contain at least one base clustering")
    if not np.all(np.isfinite(members)):
        raise ValueError("members must contain only finite numeric values")
    if m is not None and (m < 1 or m > members.shape[1]):
        raise ValueError("m must be between 1 and the number of base clusterings")
    return members.astype(np.int64)


def validate_gt(gt: np.ndarray, n_objects: int | None = None) -> np.ndarray:
    """Проверить вектор истинных меток.

    Args:
        gt: вектор меток.
        n_objects: если задано — длина `gt` должна совпадать с этим числом.

    Returns:
        Тот же вектор `dtype=np.int64`.

    Raises:
        ValueError: при некорректной форме, нечисловых значениях или
            числе уникальных классов меньше 2.
    """
    gt = np.asarray(gt).reshape(-1)
    if gt.size < 2:
        raise ValueError("gt must contain at least two labels")
    if n_objects is not None and gt.size != n_objects:
        raise ValueError("gt length must match the number of objects")
    if not np.all(np.isfinite(gt)):
        raise ValueError("gt must contain only finite numeric values")
    gt = gt.astype(np.int64)
    if np.unique(gt).size < 2:
        raise ValueError("gt must contain at least two classes")
    return gt


def validate_method(method: str) -> str:
    """Проверить, что правило агрегации поддерживается."""
    if method not in SUPPORTED_LINKAGE_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_LINKAGE_METHODS))
        raise ValueError(f"method must be one of: {allowed}")
    return method


def build_coassociation_matrix(base_cls: np.ndarray) -> np.ndarray:
    """Построить нормированную `(n, n)` co-association матрицу.

    Каждый элемент `[i, j]` — доля базовых кластеризаций, в которых
    объекты `i` и `j` попали в один кластер. Возвращается симметричная
    матрица в `[0, 1]` с диагональю, равной 1.

    Args:
        base_cls: `(n, m)` матрица меток базовых кластеризаций.

    Returns:
        `(n, n)` float64 матрица.
    """
    n, m = base_cls.shape
    consensus = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        labels = base_cls[:, j]
        consensus += (labels[:, None] == labels[None, :]).astype(np.float64)
    consensus /= m
    consensus = (consensus + consensus.T) / 2.0
    consensus = np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def get_cls_result(
    consensus_matrix: np.ndarray, cls_num: int, method: str = "average"
) -> np.ndarray:
    """Финальная иерархическая агрегация по матрице сходства.

    Args:
        consensus_matrix: `(n, n)` симметричная матрица в `[0, 1]`.
        cls_num: число кластеров на выходе (`maxclust` критерий).
        method: правило `linkage` (`average` / `complete` / `single` / `ward`).

    Returns:
        `(n,)` int64 вектор меток (нумерация с 1).

    Notes:
        Для `ward` будет выдано предупреждение: метод предполагает
        евклидово пространство признаков, а здесь применяется к
        дистанциям `1 - CA`. Использовать только для контроля и сравнения,
        не как основной режим.
    """
    method = validate_method(method)
    if method == "ward":
        warnings.warn(
            "Ward linkage is designed for Euclidean feature vectors; "
            "here it is applied to 1 - co-association distances.",
            RuntimeWarning,
            stacklevel=2,
        )
    consensus_matrix = np.clip(consensus_matrix, 0.0, 1.0)
    consensus_matrix = np.maximum(consensus_matrix, consensus_matrix.T)
    matrix = consensus_matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    similarity = squareform(matrix, checks=False)
    distance = 1.0 - similarity
    tree = linkage(distance, method=method)
    return fcluster(tree, t=cls_num, criterion="maxclust").astype(np.int64)


def run_hierarchical_consensus(
    dataset_path: str | Path,
    data_name: str | None = None,
    seed: int = 19,
    m: int = 40,
    cnt_times: int = 20,
    method: str = "average",
    selection_strategy: str = "random",
    qd_alpha: float = 0.5,
) -> dict:
    """Прогнать базовую иерархическую консенсус-кластеризацию.

    Делает `cnt_times` независимых прогонов: на каждом случайно
    выбирается под-ансамбль размера `m` из пула базовых кластеризаций,
    строится `CA`-матрица, агрегируется в финальное разбиение,
    считаются NMI/ARI/F-score.
    """
    from consensus_runner import run_consensus_loop

    def _build(base_cls, _gt, _m):
        return build_coassociation_matrix(base_cls)

    return run_consensus_loop(
        dataset_path, _build,
        data_name=data_name, seed=seed, m=m, cnt_times=cnt_times,
        method=method, selection_strategy=selection_strategy, qd_alpha=qd_alpha,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Базовая иерархическая консенсус-кластеризация."
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
    args = parser.parse_args()
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_hierarchical_consensus(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        method=args.method,
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
