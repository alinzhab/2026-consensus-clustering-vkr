from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus


DATASETS = [
    "Aggregation",
    "BBC",
    "Ecoli",
    "GLIOMA",
    "Lung",
    "orlraws10P",
    "custom_densired_dataset",
    "densired_compact_hard",
    "densired_mix_hard",
    "densired_stretched_hard",
    "repliclust_heterogeneous_hard",
    "repliclust_highdim_hard",
    "repliclust_oblong_overlap",
]

METHODS = ["average", "complete", "single", "ward"]
ALGORITHMS = ["hierarchical_baseline", "hierarchical_weighted"]


def resolve_dataset_path(root: Path, dataset: str) -> Path:
    for suffix in (".mat", ".npz"):
        path = root / f"{dataset}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Dataset file not found for {dataset}")


def load_completed(path: Path) -> set[tuple[str, str, str, str, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") == "ok":
                completed.add((row["dataset"], row["algorithm"], row["method"], row["m"], row["runs"]))
    return completed


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "algorithm",
                "method",
                "seed",
                "m",
                "runs",
                "nmi_mean",
                "nmi_std",
                "ari_mean",
                "ari_std",
                "f_mean",
                "f_std",
                "seconds",
                "status",
                "error",
            ],
            delimiter="\t",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(dataset_path: Path, dataset: str, algorithm: str, method: str, seed: int, m: int, runs: int, sharpen: float) -> dict:
    started = time.time()
    row = {
        "dataset": dataset,
        "algorithm": algorithm,
        "method": method,
        "seed": seed,
        "m": m,
        "runs": runs,
        "nmi_mean": "",
        "nmi_std": "",
        "ari_mean": "",
        "ari_std": "",
        "f_mean": "",
        "f_std": "",
        "seconds": "",
        "status": "ok",
        "error": "",
    }
    try:
        if algorithm == "hierarchical_baseline":
            result = run_hierarchical_consensus(dataset_path, dataset, seed=seed, m=m, cnt_times=runs, method=method)
        elif algorithm == "hierarchical_weighted":
            result = run_weighted_hierarchical_consensus(
                dataset_path,
                dataset,
                seed=seed,
                m=m,
                cnt_times=runs,
                method=method,
                sharpen=sharpen,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        row.update(
            {
                "nmi_mean": f"{result['nmi_mean']:.6f}",
                "nmi_std": f"{result['nmi_std']:.6f}",
                "ari_mean": f"{result['ari_mean']:.6f}",
                "ari_std": f"{result['ari_std']:.6f}",
                "f_mean": f"{result['f_mean']:.6f}",
                "f_std": f"{result['f_std']:.6f}",
            }
        )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    row["seconds"] = f"{time.time() - started:.2f}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--output", default=Path(__file__).resolve().parents[1] / "results" / "hierarchical_linkage_full_suite.tsv")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sharpen", type=float, default=1.5)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    completed = load_completed(output)

    for dataset in DATASETS:
        dataset_path = resolve_dataset_path(root, dataset)
        for method in METHODS:
            for algorithm in ALGORITHMS:
                key = (dataset, algorithm, method, str(args.m), str(args.runs))
                if key in completed:
                    print(f"SKIP {dataset:<30} {algorithm:<24} {method}")
                    continue
                row = run_one(dataset_path, dataset, algorithm, method, args.seed, args.m, args.runs, args.sharpen)
                append_row(output, row)
                completed.add(key)
                print(
                    f"{dataset:<30} {algorithm:<24} {method:<8} "
                    f"NMI={row['nmi_mean'] or 'ERR'} ARI={row['ari_mean'] or 'ERR'} F={row['f_mean'] or 'ERR'} "
                    f"{row['seconds']}s"
                )


if __name__ == "__main__":
    main()
