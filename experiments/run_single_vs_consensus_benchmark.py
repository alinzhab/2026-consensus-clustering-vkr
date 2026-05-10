"""Benchmark single clustering baselines against consensus clustering.

This script is resume-friendly: every row is appended immediately and already
completed successful rows are skipped on the next run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "consensus_lab"
sys.path.insert(0, str(LAB_DIR))

from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from metrics import compute_ari, compute_nmi, compute_pairwise_f_score
from sdgca import run_sdgca
from sdgca_modified import run_sdgca_modified


OUTPUT_COLUMNS = [
    "dataset_id",
    "dataset_type",
    "difficulty_level",
    "split",
    "fold_id",
    "size_profile",
    "n_samples",
    "n_clusters",
    "dim",
    "algorithm_family",
    "algorithm",
    "variant",
    "selection_strategy",
    "qd_alpha",
    "linkage",
    "m",
    "seed",
    "NMI",
    "ARI",
    "F-score",
    "runtime_sec",
    "memory_peak_mb",
    "status",
    "error",
]

DEFAULT_SINGLE = [
    "kmeans",
    "agglomerative_average",
    "agglomerative_complete",
    "agglomerative_single",
    "spectral",
]
DEFAULT_CONSENSUS = [
    "hierarchical_baseline",
    "hierarchical_weighted",
    "sdgca",
    "sdgca_modified",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    x = np.asarray(data["X"], dtype=np.float64)
    gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
    return x, gt


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def fit_agglomerative(x: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric="euclidean")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="euclidean")
    return model.fit_predict(x).astype(np.int64) + 1


def run_single_algorithm(algorithm: str, dataset_path: Path, seed: int) -> dict[str, float]:
    x, gt = load_npz(dataset_path)
    x_scaled = StandardScaler().fit_transform(x)
    n_clusters = int(np.unique(gt).size)

    if algorithm == "kmeans":
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x_scaled) + 1
    elif algorithm == "agglomerative_average":
        labels = fit_agglomerative(x_scaled, n_clusters, "average")
    elif algorithm == "agglomerative_complete":
        labels = fit_agglomerative(x_scaled, n_clusters, "complete")
    elif algorithm == "agglomerative_single":
        labels = fit_agglomerative(x_scaled, n_clusters, "single")
    elif algorithm == "spectral":
        if x.shape[0] > 900:
            raise RuntimeError("spectral skipped for n > 900 in this benchmark")
        labels = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels="kmeans",
            random_state=seed,
            affinity="nearest_neighbors",
            n_neighbors=min(10, max(2, x.shape[0] - 1)),
        ).fit_predict(x_scaled) + 1
    else:
        raise ValueError(f"Unknown single algorithm: {algorithm}")

    return {
        "NMI": compute_nmi(labels, gt),
        "ARI": compute_ari(labels, gt),
        "F-score": compute_pairwise_f_score(labels, gt),
    }


def run_consensus_algorithm(
    algorithm: str,
    dataset_path: Path,
    dataset_id: str,
    seed: int,
    m: int,
    runs: int,
    linkage: str,
    selection_strategy: str,
    qd_alpha: float,
    diffusion_mode: str,
) -> dict[str, float]:
    kwargs = {
        "dataset_path": dataset_path,
        "data_name": dataset_id,
        "seed": seed,
        "m": m,
        "cnt_times": runs,
        "method": linkage,
        "selection_strategy": selection_strategy,
        "qd_alpha": qd_alpha,
    }
    if algorithm == "hierarchical_baseline":
        result = run_hierarchical_consensus(**kwargs)
    elif algorithm == "hierarchical_weighted":
        result = run_weighted_hierarchical_consensus(**kwargs, sharpen=1.5)
    elif algorithm == "sdgca":
        result = run_sdgca(**kwargs, nwca_para=0.09, eta=0.75, theta=0.65)
    elif algorithm == "sdgca_modified":
        diffusion_time = None if diffusion_mode == "adaptive" else 1.0
        result = run_sdgca_modified(
            **kwargs,
            nwca_para=0.09,
            eta=0.75,
            theta=0.65,
            diffusion_time=diffusion_time,
        )
    else:
        raise ValueError(f"Unknown consensus algorithm: {algorithm}")
    return {
        "NMI": float(result["nmi_mean"]),
        "ARI": float(result["ari_mean"]),
        "F-score": float(result["f_mean"]),
    }


def completed_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, ...]] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") == "ok":
                keys.add(row_key(row))
    return keys


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("dataset_id", "")),
        str(row.get("algorithm_family", "")),
        str(row.get("algorithm", "")),
        str(row.get("variant", "")),
        str(row.get("selection_strategy", "")),
        str(row.get("qd_alpha", "")),
        str(row.get("linkage", "")),
        str(row.get("m", "")),
        str(row.get("seed", "")),
    )


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})


def measure(call: Callable[[], dict[str, float]]) -> tuple[dict[str, float], float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    result = call()
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, runtime, peak / (1024 * 1024)


def filter_manifest(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    out = []
    type_filter = set(args.dataset_types or [])
    for row in rows:
        if args.split != "all" and row["split"] != args.split:
            continue
        if args.fold is not None and int(row["fold_id"]) != args.fold:
            continue
        if type_filter and row["dataset_type"] not in type_filter:
            continue
        if args.max_n is not None and int(row["n_samples"]) > args.max_n:
            continue
        out.append(row)
    if args.limit:
        out = out[: args.limit]
    return out


def build_jobs(dataset_row: dict[str, str], args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    algorithms = args.algorithms or (DEFAULT_SINGLE + DEFAULT_CONSENSUS)
    for seed in args.seeds:
        for algorithm in algorithms:
            if algorithm in DEFAULT_SINGLE:
                if algorithm == "spectral" and int(dataset_row["n_samples"]) > 900:
                    continue
                jobs.append(
                    {
                        "algorithm_family": "single",
                        "algorithm": algorithm,
                        "variant": "default",
                        "selection_strategy": "",
                        "qd_alpha": "",
                        "linkage": algorithm.replace("agglomerative_", "") if algorithm.startswith("agglomerative_") else "",
                        "m": "",
                        "seed": seed,
                    }
                )
            elif algorithm in DEFAULT_CONSENSUS:
                if algorithm.startswith("sdgca") and not as_bool(dataset_row["allow_sdgca"]):
                    continue
                for m in args.m_values:
                    for linkage in args.linkages:
                        for selection_strategy in args.selection_strategies:
                            qd_values = args.qd_alphas if selection_strategy == "qd" else [""]
                            for qd_alpha in qd_values:
                                variant = "qd" if selection_strategy == "qd" else "random"
                                if algorithm == "sdgca_modified" and args.diffusion_mode == "adaptive":
                                    variant += "_adaptive_t"
                                jobs.append(
                                    {
                                        "algorithm_family": "consensus",
                                        "algorithm": algorithm,
                                        "variant": variant,
                                        "selection_strategy": selection_strategy,
                                        "qd_alpha": qd_alpha,
                                        "linkage": linkage,
                                        "m": m,
                                        "seed": seed,
                                    }
                                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
    return jobs


def run_job(dataset_row: dict[str, str], job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(dataset_row["path"])
    row = {
        **{key: dataset_row.get(key, "") for key in OUTPUT_COLUMNS if key in dataset_row},
        **job,
        "NMI": "",
        "ARI": "",
        "F-score": "",
        "runtime_sec": "",
        "memory_peak_mb": "",
        "status": "ok",
        "error": "",
    }
    try:
        if job["algorithm_family"] == "single":
            metrics, runtime, peak = measure(
                lambda: run_single_algorithm(job["algorithm"], dataset_path, int(job["seed"]))
            )
        else:
            metrics, runtime, peak = measure(
                lambda: run_consensus_algorithm(
                    job["algorithm"],
                    dataset_path,
                    dataset_row["dataset_id"],
                    int(job["seed"]),
                    int(job["m"]),
                    args.runs,
                    job["linkage"],
                    job["selection_strategy"],
                    float(job["qd_alpha"]) if job["qd_alpha"] != "" else 0.5,
                    args.diffusion_mode,
                )
            )
        row.update({key: round(float(value), 8) for key, value in metrics.items()})
        row["runtime_sec"] = round(runtime, 6)
        row["memory_peak_mb"] = round(peak, 3)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        if args.verbose_errors:
            row["error"] += "\n" + traceback.format_exc()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets" / "massive_synthetic" / "manifest.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "single_vs_consensus_benchmark.tsv")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--dataset-types", nargs="*")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=[19])
    parser.add_argument("--m-values", nargs="*", type=int, default=[8])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--linkages", nargs="*", default=["average"])
    parser.add_argument("--selection-strategies", nargs="*", default=["random", "qd"])
    parser.add_argument("--qd-alphas", nargs="*", type=float, default=[0.5])
    parser.add_argument("--diffusion-mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    args = parser.parse_args()

    manifest_rows = filter_manifest(read_manifest(args.manifest), args)
    done = set() if args.overwrite else completed_keys(args.output)
    total_jobs = sum(len(build_jobs(row, args)) for row in manifest_rows)
    written = 0
    skipped = 0
    print(f"Datasets: {len(manifest_rows)}; planned jobs: {total_jobs}; output: {args.output}")

    for dataset_row in manifest_rows:
        for job in build_jobs(dataset_row, args):
            preview = {
                "dataset_id": dataset_row["dataset_id"],
                **job,
            }
            if row_key(preview) in done:
                skipped += 1
                continue
            row = run_job(dataset_row, job, args)
            append_row(args.output, row)
            written += 1
            print(
                f"[{written}/{total_jobs}] {row['dataset_id']} {row['algorithm']} "
                f"{row['variant']} {row['status']} NMI={row['NMI']}"
            )
    print(f"Done. Written={written}, skipped_existing={skipped}")


if __name__ == "__main__":
    main()
