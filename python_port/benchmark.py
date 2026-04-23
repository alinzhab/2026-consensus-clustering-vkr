import argparse
import csv
import time
from pathlib import Path

from hierarchical_consensus import run_hierarchical_consensus
from hierarchical_consensus_modified import run_weighted_hierarchical_consensus


def resolve_dataset_paths(root, names):
    root = Path(root)
    paths = []
    for name in names:
        mat_path = root / f"{name}.mat"
        npz_path = root / f"{name}.npz"
        if mat_path.exists():
            paths.append(mat_path)
        elif npz_path.exists():
            paths.append(npz_path)
        else:
            raise FileNotFoundError(f"Dataset not found: {name}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "weighted"])
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single"])
    parser.add_argument("--sharpen", type=float, default=1.0)
    parser.add_argument("--output", default=Path(__file__).resolve().parents[1] / "results" / "benchmark.tsv")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner = run_hierarchical_consensus if args.mode == "baseline" else run_weighted_hierarchical_consensus
    dataset_paths = resolve_dataset_paths(args.root, args.dataset)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["dataset", "mode", "nmi_mean", "nmi_std", "ari_mean", "ari_std", "f_mean", "f_std", "seconds"])
        for dataset_path in dataset_paths:
            start = time.time()
            if args.mode == "baseline":
                result = runner(dataset_path, dataset_path.stem, args.seed, args.m, args.runs, args.method)
            else:
                result = runner(dataset_path, dataset_path.stem, args.seed, args.m, args.runs, args.method, args.sharpen)
            seconds = time.time() - start
            writer.writerow(
                [
                    dataset_path.stem,
                    args.mode,
                    f"{result['nmi_mean']:.6f}",
                    f"{result['nmi_std']:.6f}",
                    f"{result['ari_mean']:.6f}",
                    f"{result['ari_std']:.6f}",
                    f"{result['f_mean']:.6f}",
                    f"{result['f_std']:.6f}",
                    f"{seconds:.2f}",
                ]
            )
            print(dataset_path.stem, args.mode, f"{result['nmi_mean']:.3f}", f"{result['ari_mean']:.3f}", f"{result['f_mean']:.3f}")
    print(output_path)


if __name__ == "__main__":
    main()
