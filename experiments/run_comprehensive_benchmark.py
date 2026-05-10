"""Comprehensive benchmark: generate 500-1000 datasets, run simple + consensus algorithms, compare.

Usage:
    python experiments/run_comprehensive_benchmark.py --count 700 --phase all
    python experiments/run_comprehensive_benchmark.py --count 700 --phase generate
    python experiments/run_comprehensive_benchmark.py --phase benchmark
    python experiments/run_comprehensive_benchmark.py --phase summary

Phases:
    generate  -- generate synthetic datasets (adds to datasets/massive_synthetic/)
    benchmark -- run simple and consensus algorithms on all datasets
    summary   -- print comparison table from existing results
    all       -- generate + benchmark + summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.io import loadmat
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "consensus_lab"
sys.path.insert(0, str(LAB_DIR))

from metrics import compute_ari, compute_nmi, compute_pairwise_f_score
from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from sdgca import run_sdgca
from sdgca_modified import run_sdgca_modified

# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation helpers
# ─────────────────────────────────────────────────────────────────────────────

DATASET_TYPES = [
    "compact",
    "overlapping",
    "imbalanced",
    "high_dimensional",
    "elongated",
    "density_varied",
    "mixed_complex",
]


def _size_profile(index: int, rng: np.random.Generator) -> tuple[str, int]:
    bucket = index % 10
    if bucket < 5:
        return "small", int(rng.integers(300, 601))
    if bucket < 8:
        return "medium", int(rng.integers(800, 1501))
    return "large", int(rng.integers(2000, 4001))


def _common_base_params(n_clusters: int, size_profile: str) -> dict[str, Any]:
    base_clusterings = 30 if size_profile != "large" else 18
    return {
        "base_clusterings": base_clusterings,
        "base_k_min": max(2, n_clusters - 2),
        "base_k_max": n_clusters + 2,
        "base_strategy": "mixed" if size_profile != "large" else "kmeans",
    }


def _build_params(dataset_type: str, dataset_id: str, seed: int,
                  index: int, rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    size_profile, n_samples = _size_profile(index, rng)
    n_clusters = int(rng.integers(3, 11))

    if dataset_type == "compact":
        dim = int(rng.choice([2, 3, 5]))
        return "simple", {
            "name": dataset_id, "n_samples": n_samples, "n_clusters": n_clusters,
            "dim": dim, "cluster_std": float(rng.uniform(0.18, 0.35)),
            "separation": float(rng.uniform(5.5, 8.0)),
            "imbalance_ratio": float(rng.uniform(1.0, 1.4)), "seed": seed,
            **_common_base_params(n_clusters, size_profile),
        }
    elif dataset_type == "overlapping":
        dim = int(rng.choice([2, 3, 5, 10]))
        return "simple", {
            "name": dataset_id, "n_samples": n_samples, "n_clusters": n_clusters,
            "dim": dim, "cluster_std": float(rng.uniform(0.65, 1.15)),
            "separation": float(rng.uniform(1.8, 3.0)),
            "imbalance_ratio": float(rng.uniform(1.0, 1.8)), "seed": seed,
            **_common_base_params(n_clusters, size_profile),
        }
    elif dataset_type == "imbalanced":
        dim = int(rng.choice([2, 5, 10]))
        return "simple", {
            "name": dataset_id, "n_samples": n_samples, "n_clusters": n_clusters,
            "dim": dim, "cluster_std": float(rng.uniform(0.35, 0.75)),
            "separation": float(rng.uniform(3.2, 5.5)),
            "imbalance_ratio": float(rng.uniform(3.0, 8.0)), "seed": seed,
            **_common_base_params(n_clusters, size_profile),
        }
    elif dataset_type == "high_dimensional":
        dim = int(rng.choice([10, 20, 50, 100]))
        return "simple", {
            "name": dataset_id, "n_samples": n_samples, "n_clusters": n_clusters,
            "dim": dim, "cluster_std": float(rng.uniform(0.45, 0.95)),
            "separation": float(rng.uniform(2.4, 4.8)),
            "imbalance_ratio": float(rng.uniform(1.0, 2.5)), "seed": seed,
            **_common_base_params(n_clusters, size_profile),
        }
    elif dataset_type == "elongated":
        dim = int(rng.choice([2, 3, 5, 10]))
        return "repliclust", {
            "name": dataset_id, "n_clusters": n_clusters, "dim": dim,
            "n_samples": n_samples,
            "aspect_ref": float(rng.uniform(4.5, 7.0)),
            "aspect_maxmin": float(rng.uniform(4.0, 8.0)),
            "radius_ref": 1.0, "radius_maxmin": float(rng.uniform(1.5, 3.0)),
            "min_overlap": float(rng.uniform(0.02, 0.08)),
            "max_overlap": float(rng.uniform(0.12, 0.28)),
            "imbalance_ratio": float(rng.uniform(1.5, 3.5)),
            "distributions": ["normal", "student_t", "lognormal"],
            "distribution_proportions": [0.45, 0.35, 0.20], "seed": seed,
            **_common_base_params(n_clusters, size_profile),
        }
    elif dataset_type == "density_varied":
        dim = int(rng.choice([2, 3, 5]))
        n_clusters_clamped = min(n_clusters, 7)
        factors = rng.uniform(0.4, 2.0, size=n_clusters_clamped).round(3).tolist()
        core_num = int(rng.integers(
            max(n_clusters_clamped * 6, 24),
            max(n_clusters_clamped * 16, 50)
        ))
        return "densired", {
            "name": dataset_id, "dim": dim, "clunum": n_clusters_clamped,
            "core_num": core_num,
            "data_num": min(n_samples, 1500),
            "seed": seed, "domain_size": 20.0,
            "radius": float(rng.uniform(0.03, 0.06)),
            "step": float(rng.uniform(0.05, 0.08)),
            "noise_ratio": float(rng.uniform(0.03, 0.12)),
            "density_factors": factors,
            "momentum": float(rng.uniform(0.0, 0.4)),
            "branch": float(rng.uniform(0.0, 0.06)),
            "star": float(rng.uniform(0.0, 0.15)),
            "distribution": str(rng.choice(["uniform", "gaussian"])),
            **_common_base_params(n_clusters_clamped, "small"),
        }
    elif dataset_type == "mixed_complex":
        dim = int(rng.choice([10, 20, 50]))
        return "repliclust", {
            "name": dataset_id, "n_clusters": n_clusters, "dim": dim,
            "n_samples": n_samples,
            "aspect_ref": float(rng.uniform(4.5, 7.0)),
            "aspect_maxmin": float(rng.uniform(4.0, 8.0)),
            "radius_ref": 1.0, "radius_maxmin": float(rng.uniform(2.0, 4.0)),
            "min_overlap": float(rng.uniform(0.04, 0.12)),
            "max_overlap": float(rng.uniform(0.18, 0.35)),
            "imbalance_ratio": float(rng.uniform(3.0, 7.0)),
            "distributions": ["normal", "student_t", "exponential", "lognormal"],
            "distribution_proportions": [0.30, 0.30, 0.20, 0.20], "seed": seed,
            **_common_base_params(n_clusters, size_profile),
        }
    else:
        raise ValueError(f"Unknown type: {dataset_type}")


def generate_datasets(count: int, output_dir: Path, seed: int, force: bool) -> list[Path]:
    from simple_dataset_generator import generate_simple_gaussian_dataset, save_dataset as save_simple
    from densired_style_generator import generate_densired_style_dataset, save_dataset as save_densired
    from repliclust_style_generator import generate_archetype_dataset, save_dataset as save_repliclust

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    errors = 0

    print(f"Generating {count} datasets -> {output_dir}")
    for i in range(count):
        dataset_type = DATASET_TYPES[i % len(DATASET_TYPES)]
        per_type_idx = i // len(DATASET_TYPES)
        dataset_id = f"bench_{dataset_type}_{per_type_idx:03d}"
        path = output_dir / f"{dataset_id}.npz"

        if path.exists() and not force:
            paths.append(path)
            continue

        dataset_seed = int(seed + 1009 * i + 17)
        try:
            gen, params = _build_params(dataset_type, dataset_id, dataset_seed, i, rng)
            if gen == "simple":
                x, gt, members, meta = generate_simple_gaussian_dataset(**params)
                save_simple(path, x, gt, members, meta)
            elif gen == "densired":
                x, gt, members, meta = generate_densired_style_dataset(**params)
                save_densired(path, x, gt, members, meta)
            elif gen == "repliclust":
                x, gt, members, meta = generate_archetype_dataset(**params)
                save_repliclust(path, x, gt, members, meta)
            paths.append(path)
        except Exception as exc:
            errors += 1
            # Retry with simple gaussian fallback
            fallback_seed = dataset_seed + 99999
            try:
                fallback_params = {
                    "name": dataset_id,
                    "n_samples": int(rng.integers(300, 601)),
                    "n_clusters": int(rng.integers(3, 8)),
                    "dim": int(rng.choice([2, 5, 10])),
                    "cluster_std": 0.5,
                    "separation": 4.0,
                    "imbalance_ratio": 2.0,
                    "seed": fallback_seed,
                    "base_clusterings": 24,
                    "base_k_min": 2,
                    "base_k_max": 8,
                    "base_strategy": "mixed",
                }
                x, gt, members, meta = generate_simple_gaussian_dataset(**fallback_params)
                meta["original_type"] = dataset_type
                meta["fallback_reason"] = str(exc)[:120]
                save_simple(path, x, gt, members, meta)
                paths.append(path)
            except Exception as exc2:
                print(f"  SKIP {dataset_id}: {exc2}")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{count} done ({errors} errors with fallback)")

    print(f"Generated/registered {len(paths)} datasets ({errors} used fallback)")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_full(path: Path) -> dict[str, Any]:
    """Load X (if available), members, gt from .npz or .mat."""
    suffix = path.suffix.lower()
    if suffix == ".mat":
        data = loadmat(path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        x = np.asarray(data["X"]) if "X" in data else None
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        x = np.asarray(data["X"]) if "X" in data else None
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    if gt.size > 0 and gt.min() == 0:
        gt = gt + 1

    return {"X": x, "members": members, "gt": gt, "n": members.shape[0],
            "m": members.shape[1], "K": int(np.unique(gt).size)}


# ─────────────────────────────────────────────────────────────────────────────
# Simple clustering algorithms
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64) + 1


def run_kmeans_simple(x: np.ndarray, gt: np.ndarray, K: int,
                      n_runs: int = 5, seed: int = 42) -> dict[str, float]:
    from scipy.cluster.vq import kmeans2
    nmis, aris, fs = [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_runs):
        try:
            _, labels = kmeans2(x, K, minit="points", iter=100, seed=rng)
            labels = _normalize_labels(labels)
            nmis.append(compute_nmi(gt, labels))
            aris.append(compute_ari(gt, labels))
            fs.append(compute_pairwise_f_score(gt, labels))
        except Exception:
            pass
    if not nmis:
        return {"nmi_mean": 0.0, "nmi_std": 0.0, "ari_mean": 0.0,
                "ari_std": 0.0, "f_mean": 0.0, "f_std": 0.0}
    return {
        "nmi_mean": float(np.mean(nmis)), "nmi_std": float(np.std(nmis)),
        "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris)),
        "f_mean": float(np.mean(fs)), "f_std": float(np.std(fs)),
    }


def run_hierarchical_simple(x: np.ndarray, gt: np.ndarray, K: int,
                             method: str = "average") -> dict[str, float]:
    try:
        from sklearn.preprocessing import StandardScaler
        x_s = StandardScaler().fit_transform(x)
    except Exception:
        x_s = x
    try:
        tree = linkage(x_s, method=method, metric="euclidean")
        labels = fcluster(tree, t=K, criterion="maxclust").astype(np.int64)
        return {
            "nmi_mean": float(compute_nmi(gt, labels)),
            "nmi_std": 0.0,
            "ari_mean": float(compute_ari(gt, labels)),
            "ari_std": 0.0,
            "f_mean": float(compute_pairwise_f_score(gt, labels)),
            "f_std": 0.0,
        }
    except Exception:
        return {"nmi_mean": 0.0, "nmi_std": 0.0, "ari_mean": 0.0,
                "ari_std": 0.0, "f_mean": 0.0, "f_std": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Result I/O
# ─────────────────────────────────────────────────────────────────────────────

HEADER = [
    "dataset", "algorithm", "method", "seed", "m", "runs",
    "n_objects", "n_clusters", "dim",
    "nmi_mean", "nmi_std", "ari_mean", "ari_std", "f_mean", "f_std",
    "seconds", "status", "error",
]


def load_completed(path: Path) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") == "ok":
                done.add((row["dataset"], row["algorithm"], row["method"]))
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _base_row(path: Path, algorithm: str, method: str, d: dict) -> dict:
    return {
        "dataset": path.stem,
        "algorithm": algorithm,
        "method": method,
        "seed": 42,
        "m": d["m"],
        "runs": "",
        "n_objects": d["n"],
        "n_clusters": d["K"],
        "dim": d["X"].shape[1] if d["X"] is not None else "",
        "nmi_mean": "", "nmi_std": "",
        "ari_mean": "", "ari_std": "",
        "f_mean": "", "f_std": "",
        "seconds": "",
        "status": "ok",
        "error": "",
    }


def run_benchmark(
    dataset_paths: list[Path],
    output_path: Path,
    seed: int = 42,
    m: int = 30,
    runs: int = 5,
    max_n_sdgca: int = 400,
    sdgca_methods: tuple[str, ...] = ("average", "ward"),
    sdgca_runs: int = 2,
    run_simple: bool = True,
    run_consensus: bool = True,
    max_n_hierarchical_simple: int = 2000,
    max_n_consensus: int = 2000,
) -> None:
    done = load_completed(output_path)
    total = len(dataset_paths)

    print(f"\nBenchmark: {total} datasets -> {output_path}")
    print(f"  Simple algorithms: {run_simple}")
    print(f"  Consensus algorithms: {run_consensus}")
    print(f"  Already completed entries: {len(done)}")

    for idx, path in enumerate(dataset_paths):
        try:
            d = load_dataset_full(path)
        except Exception as exc:
            print(f"  [{idx+1}/{total}] LOAD ERROR {path.stem}: {exc}")
            continue

        n, K = d["n"], d["K"]
        x, members, gt = d["X"], d["members"], d["gt"]

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{total}] {path.stem} (n={n}, K={K})")

        # ── Simple algorithms (require X) ─────────────────────────────────
        if run_simple and x is not None:
            # K-Means
            algo_key = ("kmeans", "—")
            if (path.stem, "kmeans", "—") not in done:
                t0 = time.time()
                row = _base_row(path, "kmeans", "—", d)
                row["runs"] = runs
                try:
                    res = run_kmeans_simple(x, gt, K, n_runs=runs, seed=seed)
                    row.update(res)
                    row["seconds"] = round(time.time() - t0, 3)
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)[:200]
                append_row(output_path, row)

            # Hierarchical (skip huge n for memory)
            if n <= max_n_hierarchical_simple:
                for method in ("average", "complete", "single", "ward"):
                    key = (path.stem, f"hierarchical_simple", method)
                    if key not in done:
                        t0 = time.time()
                        row = _base_row(path, "hierarchical_simple", method, d)
                        row["runs"] = 1
                        try:
                            res = run_hierarchical_simple(x, gt, K, method=method)
                            row.update(res)
                            row["seconds"] = round(time.time() - t0, 3)
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = str(exc)[:200]
                        append_row(output_path, row)

        # ── Consensus algorithms ──────────────────────────────────────────
        if run_consensus and n <= max_n_consensus:
            eff_m = min(m, members.shape[1])

            # hierarchical_baseline
            for method in ("average", "complete", "single", "ward"):
                key = (path.stem, "hierarchical_baseline", method)
                if key not in done:
                    t0 = time.time()
                    row = _base_row(path, "hierarchical_baseline", method, d)
                    row["m"] = eff_m
                    row["runs"] = runs
                    try:
                        res = run_hierarchical_consensus(
                            dataset_path=path, seed=seed, m=eff_m,
                            cnt_times=runs, method=method,
                        )
                        row.update({k: res[k] for k in
                                    ("nmi_mean", "nmi_std", "ari_mean", "ari_std",
                                     "f_mean", "f_std")})
                        row["seconds"] = round(time.time() - t0, 3)
                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = str(exc)[:200]
                    append_row(output_path, row)

            # hierarchical_weighted
            for method in ("average", "complete", "single", "ward"):
                key = (path.stem, "hierarchical_weighted", method)
                if key not in done:
                    t0 = time.time()
                    row = _base_row(path, "hierarchical_weighted", method, d)
                    row["m"] = eff_m
                    row["runs"] = runs
                    try:
                        res = run_weighted_hierarchical_consensus(
                            dataset_path=path, seed=seed, m=eff_m,
                            cnt_times=runs, method=method,
                        )
                        row.update({k: res[k] for k in
                                    ("nmi_mean", "nmi_std", "ari_mean", "ari_std",
                                     "f_mean", "f_std")})
                        row["seconds"] = round(time.time() - t0, 3)
                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = str(exc)[:200]
                    append_row(output_path, row)

            # SDGCA (only small datasets due to ADMM cost)
            if n <= max_n_sdgca:
                for method in sdgca_methods:
                    key = (path.stem, "sdgca", method)
                    if key not in done:
                        t0 = time.time()
                        row = _base_row(path, "sdgca", method, d)
                        row["m"] = eff_m
                        row["runs"] = sdgca_runs
                        try:
                            res = run_sdgca(
                                dataset_path=path, seed=seed, m=eff_m,
                                cnt_times=sdgca_runs, method=method,
                            )
                            row.update({k: res[k] for k in
                                        ("nmi_mean", "nmi_std", "ari_mean", "ari_std",
                                         "f_mean", "f_std")})
                            row["seconds"] = round(time.time() - t0, 3)
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = str(exc)[:200]
                        append_row(output_path, row)

                    key = (path.stem, "sdgca_modified", method)
                    if key not in done:
                        t0 = time.time()
                        row = _base_row(path, "sdgca_modified", method, d)
                        row["m"] = eff_m
                        row["runs"] = sdgca_runs
                        try:
                            res = run_sdgca_modified(
                                dataset_path=path, seed=seed, m=eff_m,
                                cnt_times=sdgca_runs, method=method,
                            )
                            row.update({k: res[k] for k in
                                        ("nmi_mean", "nmi_std", "ari_mean", "ari_std",
                                         "f_mean", "f_std")})
                            row["seconds"] = round(time.time() - t0, 3)
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = str(exc)[:200]
                        append_row(output_path, row)


# ─────────────────────────────────────────────────────────────────────────────
# Summary / comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results_path: Path) -> None:
    if not results_path.exists():
        print("No results file found.")
        return

    rows: list[dict] = []
    with results_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") == "ok" and row.get("nmi_mean"):
                rows.append(row)

    if not rows:
        print("No completed results yet.")
        return

    # Group by algorithm, best method per dataset
    from collections import defaultdict

    # For each (dataset, algorithm): take the method with highest NMI
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["dataset"], row["algorithm"])
        nmi = float(row["nmi_mean"])
        if key not in best or nmi > float(best[key]["nmi_mean"]):
            best[key] = row

    # Aggregate per algorithm
    algo_stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (dataset, algo), row in best.items():
        algo_stats[algo]["nmi"].append(float(row["nmi_mean"]))
        algo_stats[algo]["ari"].append(float(row["ari_mean"]))
        algo_stats[algo]["f"].append(float(row["f_mean"]))
        algo_stats[algo]["n_datasets"].append(1)

    algo_order = [
        "kmeans", "hierarchical_simple",
        "hierarchical_baseline", "hierarchical_weighted",
        "sdgca", "sdgca_modified",
    ]

    print("\n" + "=" * 82)
    print("BENCHMARK SUMMARY — Best method per dataset, then averaged across datasets")
    print("=" * 82)
    print(f"{'Algorithm':<26} {'N datasets':>10}  {'NMI mean':>10}  {'ARI mean':>10}  {'F mean':>10}")
    print("-" * 82)

    for algo in algo_order:
        if algo not in algo_stats:
            continue
        st = algo_stats[algo]
        n = len(st["nmi"])
        print(f"{algo:<26} {n:>10}  "
              f"{np.mean(st['nmi']):>10.4f}  "
              f"{np.mean(st['ari']):>10.4f}  "
              f"{np.mean(st['f']):>10.4f}")

    # Also show per-method breakdown for consensus algorithms
    print("\n" + "=" * 82)
    print("PER-METHOD BREAKDOWN (consensus algorithms, average across datasets)")
    print("=" * 82)
    print(f"{'Algorithm':<26} {'Method':<10} {'N':>6}  {'NMI':>8}  {'ARI':>8}  {'F':>8}")
    print("-" * 82)

    method_stats: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        algo = row["algorithm"]
        if algo.startswith("hierarchical_baseline") or algo.startswith("hierarchical_weighted") \
                or algo in ("sdgca", "sdgca_modified"):
            key = (algo, row["method"])
            method_stats[key]["nmi"].append(float(row["nmi_mean"]))
            method_stats[key]["ari"].append(float(row["ari_mean"]))
            method_stats[key]["f"].append(float(row["f_mean"]))

    for algo in ["hierarchical_baseline", "hierarchical_weighted", "sdgca", "sdgca_modified"]:
        for method in ("average", "complete", "single", "ward"):
            key = (algo, method)
            if key not in method_stats:
                continue
            st = method_stats[key]
            n = len(st["nmi"])
            print(f"{algo:<26} {method:<10} {n:>6}  "
                  f"{np.mean(st['nmi']):>8.4f}  "
                  f"{np.mean(st['ari']):>8.4f}  "
                  f"{np.mean(st['f']):>8.4f}")
        print()

    print(f"\nTotal rows: {len(rows)}")
    print(f"Results file: {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_all_datasets(datasets_root: Path, synthetic_dir: Path) -> list[Path]:
    paths: list[Path] = []
    # Real datasets
    for p in sorted(datasets_root.glob("*.mat")):
        if "smoke" not in p.stem.lower() and "uploaded" not in str(p):
            paths.append(p)
    # Existing synthetic (.npz in datasets root, not smoke/web)
    for p in sorted(datasets_root.glob("*.npz")):
        stem = p.stem.lower()
        if "smoke" not in stem and "web_" not in stem:
            paths.append(p)
    # New massive synthetic
    for p in sorted(synthetic_dir.glob("bench_*.npz")):
        paths.append(p)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive clustering benchmark")
    parser.add_argument("--count", type=int, default=700,
                        help="Number of synthetic datasets to generate")
    parser.add_argument("--phase", choices=["generate", "benchmark", "summary", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--m", type=int, default=30,
                        help="Ensemble size for consensus algorithms")
    parser.add_argument("--runs", type=int, default=5,
                        help="Independent runs per hierarchical consensus")
    parser.add_argument("--sdgca-runs", type=int, default=2,
                        help="Independent runs per SDGCA (fewer because slow)")
    parser.add_argument("--sdgca-methods", nargs="+",
                        default=["average", "ward"],
                        choices=["average", "complete", "single", "ward"],
                        help="Linkage methods to test for SDGCA")
    parser.add_argument("--max-n-sdgca", type=int, default=400,
                        help="Skip SDGCA for datasets larger than this (0=disable)")
    parser.add_argument("--max-n-consensus", type=int, default=2000,
                        help="Skip hierarchical consensus for n > this (avoids O(n^2) OOM)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "comprehensive_benchmark.tsv")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate datasets even if they already exist")
    parser.add_argument("--no-simple", action="store_true",
                        help="Skip simple clustering algorithms")
    parser.add_argument("--no-consensus", action="store_true",
                        help="Skip consensus algorithms")
    args = parser.parse_args()

    synthetic_dir = ROOT / "datasets" / "massive_synthetic"
    datasets_root = ROOT / "datasets"

    if args.phase in ("generate", "all"):
        generate_datasets(args.count, synthetic_dir, args.seed, args.force)

    if args.phase in ("benchmark", "all"):
        all_paths = discover_all_datasets(datasets_root, synthetic_dir)
        print(f"\nDiscovered {len(all_paths)} datasets total")
        run_benchmark(
            dataset_paths=all_paths,
            output_path=args.output,
            seed=args.seed,
            m=args.m,
            runs=args.runs,
            max_n_sdgca=args.max_n_sdgca,
            sdgca_methods=tuple(args.sdgca_methods),
            sdgca_runs=args.sdgca_runs,
            run_simple=not args.no_simple,
            run_consensus=not args.no_consensus,
            max_n_hierarchical_simple=args.max_n_consensus,
            max_n_consensus=args.max_n_consensus,
        )

    if args.phase in ("summary", "all"):
        print_summary(args.output)


if __name__ == "__main__":
    main()
