from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

from sdgca import run_sdgca
from sdgca_modified import resolve_params, run_sdgca_modified


REAL_DATASETS = [
    "Aggregation",
    "BBC",
    "Ecoli",
    "GLIOMA",
    "Lung",
    "orlraws10P",
]

SYNTHETIC_DATASETS = [
    "custom_densired_dataset",
    "densired_compact_hard",
    "densired_mix_hard",
    "densired_stretched_hard",
    "repliclust_heterogeneous_hard",
    "repliclust_highdim_hard",
    "repliclust_oblong_overlap",
]

METHODS = ["average", "complete", "single", "ward"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(root: Path, dataset: str) -> Path:
    for suffix in (".mat", ".npz"):
        path = root / f"{dataset}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Dataset file not found for {dataset}")


def append_tsv_row(path: Path, row: dict, header: list[str]) -> None:
    ensure_parent(path)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl_row(path: Path, row: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    completed: set[tuple[str, str, str, str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (
                row.get("dataset", ""),
                row.get("algorithm", ""),
                row.get("method", ""),
                row.get("m", ""),
                row.get("runs", ""),
            )
            completed.add(key)
    return completed


def run_one(
    dataset_path: Path,
    dataset_name: str,
    algorithm: str,
    method: str,
    seed: int,
    m: int,
    runs: int,
) -> dict:
    started = time.time()
    try:
        if algorithm == "sdgca":
            result = run_sdgca(
                dataset_path=dataset_path,
                data_name=dataset_name,
                seed=seed,
                m=m,
                cnt_times=runs,
                method=method,
            )
        elif algorithm == "sdgca_modified":
            params = resolve_params(dataset_name, None, None, None, None)
            result = run_sdgca_modified(
                dataset_path=dataset_path,
                data_name=dataset_name,
                seed=seed,
                m=m,
                cnt_times=runs,
                nwca_para=params["lambda_"],
                eta=params["eta"],
                theta=params["theta"],
                method=method,
                diffusion_time=params["diffusion_time"],
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        elapsed = time.time() - started
        return {
            "dataset": dataset_name,
            "algorithm": algorithm,
            "method": method,
            "seed": seed,
            "m": m,
            "runs": runs,
            "nmi_mean": f"{result['nmi_mean']:.6f}",
            "nmi_std": f"{result['nmi_std']:.6f}",
            "ari_mean": f"{result['ari_mean']:.6f}",
            "ari_std": f"{result['ari_std']:.6f}",
            "f_mean": f"{result['f_mean']:.6f}",
            "f_std": f"{result['f_std']:.6f}",
            "seconds": f"{elapsed:.2f}",
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        elapsed = time.time() - started
        return {
            "dataset": dataset_name,
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
            "seconds": f"{elapsed:.2f}",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--results-dir", default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.root)
    results_dir = Path(args.results_dir)
    tsv_path = results_dir / "sdgca_linkage_full_suite.tsv"
    jsonl_path = results_dir / "sdgca_linkage_full_suite.jsonl"
    progress_path = results_dir / "sdgca_linkage_full_suite_progress.log"
    completed_keys = load_completed_keys(tsv_path)

    datasets = REAL_DATASETS + SYNTHETIC_DATASETS
    if args.include_smoke:
        datasets += ["web_smoke_dataset", "web_ui_smoke", "web_multi_page_smoke"]

    header = [
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
    ]

    with progress_path.open("a", encoding="utf-8") as log:
        log.write(f"=== linkage suite started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    for dataset_name in datasets:
        dataset_path = resolve_dataset_path(dataset_root, dataset_name)
        for method in METHODS:
            for algorithm in ("sdgca", "sdgca_modified"):
                key = (dataset_name, algorithm, method, str(args.m), str(args.runs))
                if key in completed_keys:
                    with progress_path.open("a", encoding="utf-8") as log:
                        log.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {dataset_name} | {algorithm} | {method} | skipped_existing\n"
                        )
                    continue
                row = run_one(
                    dataset_path=dataset_path,
                    dataset_name=dataset_name,
                    algorithm=algorithm,
                    method=method,
                    seed=args.seed,
                    m=args.m,
                    runs=args.runs,
                )
                append_tsv_row(tsv_path, row, header)
                append_jsonl_row(jsonl_path, row)
                completed_keys.add(key)
                with progress_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {dataset_name} | {algorithm} | {method} | {row['status']} | {row['seconds']} sec\n"
                    )
                    if row["status"] != "ok" and "traceback" in row:
                        log.write(row["traceback"] + "\n")

    with progress_path.open("a", encoding="utf-8") as log:
        log.write(f"=== linkage suite finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


if __name__ == "__main__":
    main()
