import argparse
import csv
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
from sdgca import run_sdgca
from sdgca_modified import resolve_params, run_sdgca_modified


def dataset_member_count(path: Path) -> int:
    if path.suffix.lower() == ".mat":
        data = loadmat(path)
        return int(np.asarray(data["members"]).shape[1])
    data = np.load(path, allow_pickle=True)
    return int(data["members"].shape[1])


def discover_datasets(root: Path):
    datasets = []
    for path in sorted(root.glob("*.mat")) + sorted(root.glob("*.npz")):
        stem = path.stem.lower()
        if stem.startswith("web_"):
            continue
        datasets.append(path)
    return datasets


def append_row(tsv_path: Path, row: dict):
    file_exists = tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
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
                "status",
                "error",
            ],
            delimiter="\t",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_completed(tsv_path: Path):
    completed = set()
    if not tsv_path.exists():
        return completed
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") == "ok":
                completed.add((row.get("dataset"), row.get("algorithm"), row.get("method")))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--output", default=Path(__file__).resolve().parents[1] / "results" / "full_benchmark.tsv")
    parser.add_argument("--summary", default=Path(__file__).resolve().parents[1] / "results" / "full_benchmark.json")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single", "ward"])
    parser.add_argument("--sharpen", type=float, default=1.5)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    summary = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(root)
    results = []
    completed = load_completed(output)

    for dataset_path in datasets:
        members_count = dataset_member_count(dataset_path)
        effective_m = min(args.m, members_count)
        jobs = [
            ("hierarchical_baseline", lambda: run_hierarchical_consensus(dataset_path, dataset_path.stem, args.seed, effective_m, args.runs, args.method)),
            ("hierarchical_weighted", lambda: run_weighted_hierarchical_consensus(dataset_path, dataset_path.stem, args.seed, effective_m, args.runs, args.method, args.sharpen)),
            ("sdgca", lambda: run_sdgca(dataset_path, dataset_path.stem, args.seed, effective_m, args.runs, 0.09, 0.75, 0.65, args.method)),
            (
                "sdgca_modified",
                lambda: (
                    lambda params: run_sdgca_modified(
                        dataset_path,
                        dataset_path.stem,
                        args.seed,
                        effective_m,
                        args.runs,
                        params["lambda_"],
                        params["eta"],
                        params["theta"],
                        args.method,
                        params["diffusion_time"],
                    )
                )(resolve_params(dataset_path.stem, None, None, None, None)),
            ),
        ]

        for algorithm_name, runner in jobs:
            if (dataset_path.stem, algorithm_name, args.method) in completed:
                print(f"SKIP {dataset_path.stem:<32} {algorithm_name:<24} already computed")
                continue
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_path.stem,
                "algorithm": algorithm_name,
                "method": args.method,
                "seed": args.seed,
                "m": effective_m,
                "runs": args.runs,
                "nmi_mean": "",
                "nmi_std": "",
                "ari_mean": "",
                "ari_std": "",
                "f_mean": "",
                "f_std": "",
                "status": "ok",
                "error": "",
            }
            try:
                result = runner()
                row["nmi_mean"] = f"{result['nmi_mean']:.6f}"
                row["nmi_std"] = f"{result['nmi_std']:.6f}"
                row["ari_mean"] = f"{result['ari_mean']:.6f}"
                row["ari_std"] = f"{result['ari_std']:.6f}"
                row["f_mean"] = f"{result['f_mean']:.6f}"
                row["f_std"] = f"{result['f_std']:.6f}"
                print(
                    f"{dataset_path.stem:<32} {algorithm_name:<24} "
                    f"NMI={row['nmi_mean']} ARI={row['ari_mean']} F={row['f_mean']}"
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{dataset_path.stem:<32} {algorithm_name:<24} ERROR {exc}")
                traceback.print_exc()
            append_row(output, row)
            results.append(row)
            summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved TSV to {output}")
    print(f"Saved JSON to {summary}")


if __name__ == "__main__":
    main()
