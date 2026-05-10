"""Прямое измерение сложности: время прогона vs n при фиксированных m, k.

Отвечает на вопрос «насколько действительно квадратична сложность» —
не косвенно через таблицу из существующих результатов (где n коррелирует
с типом датасета), а напрямую: один синтетический генератор, фиксированные
m, K, варьируем только n.

Прогоняет каждый из 4 алгоритмов на n ∈ {300, 600, 1000, 1500, 2000, 3000}
с фиксированными m=20, K=4, runs=3. Подгоняет степенной закон t = a · n^b
по линейной регрессии log-log и пишет в результат отдельно показатель b
для каждого алгоритма.

Запуск (Windows):
    .\\.venv\\Scripts\\Activate.ps1
    python experiments\\run_complexity_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consensus_lab"))


def make_dataset(n: int, k: int, d: int, seed: int) -> Path:
    """Простая гауссовская смесь, сохраняем как .npz рядом с датасетами."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 5, size=(k, d))
    pts_per = n // k
    x = np.vstack(
        [centres[i] + rng.normal(scale=0.6, size=(pts_per, d)) for i in range(k)]
    )
    gt = np.repeat(np.arange(1, k + 1), pts_per)
    if x.shape[0] < n:
        # дополнение, чтобы получить ровно n.
        extra = n - x.shape[0]
        x = np.vstack([x, centres[0] + rng.normal(scale=0.6, size=(extra, d))])
        gt = np.concatenate([gt, np.ones(extra, dtype=int)])

    from base_clusterings import build_base_clusterings

    members = build_base_clusterings(x, n_clusterings=20, k_min=2, k_max=k + 2, rng=seed)
    out = ROOT / "datasets" / f"complexity_n{n}.npz"
    np.savez(out, X=x, gt=gt.astype(np.int64), members=members.astype(np.int64))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[300, 600, 1000, 1500, 2000, 3000],
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--method", default="average")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "hierarchical_baseline",
            "hierarchical_weighted",
            "sdgca",
            "sdgca_modified",
        ],
    )
    parser.add_argument(
        "--out",
        default=ROOT / "results" / "complexity_benchmark.tsv",
    )
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()

    # Подгрузим алгоритмы лениво, чтобы этот файл импортировался без scipy.
    from hierarchical_consensus import run_hierarchical_consensus
    from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
    from sdgca import run_sdgca
    from sdgca_modified import run_sdgca_modified, resolve_params

    runners = {
        "hierarchical_baseline": (run_hierarchical_consensus, {}),
        "hierarchical_weighted": (run_weighted_hierarchical_consensus, {}),
        "sdgca": (run_sdgca, {}),
        "sdgca_modified": (run_sdgca_modified, "resolve"),
    }

    rows: list[dict] = []
    for n in args.sizes:
        path = make_dataset(n=n, k=args.k, d=args.d, seed=args.seed)
        print(f"\n=== n={n} ({path.name}) ===")
        for alg in args.algorithms:
            runner, kwargs = runners[alg]
            if kwargs == "resolve":
                p = resolve_params(path.stem, None, None, None, None)
                kwargs = {
                    "nwca_para": p["lambda_"],
                    "eta": p["eta"],
                    "theta": p["theta"],
                    "diffusion_time": p["diffusion_time"],
                }
            t0 = time.time()
            try:
                res = runner(
                    dataset_path=path,
                    data_name=path.stem,
                    seed=args.seed,
                    m=args.m,
                    cnt_times=args.runs,
                    method=args.method,
                    **kwargs,
                )
                status = "ok"
                err = ""
                nmi = float(res["nmi_mean"])
            except Exception as exc:
                status = "error"
                err = str(exc)
                nmi = float("nan")
            t1 = time.time()
            rows.append(
                {
                    "n": n,
                    "algorithm": alg,
                    "seconds": round(t1 - t0, 3),
                    "nmi_mean": round(nmi, 4),
                    "method": args.method,
                    "m": args.m,
                    "runs": args.runs,
                    "status": status,
                    "error": err,
                }
            )
            print(f"  {alg}: t = {t1 - t0:.2f}s, NMI = {nmi:.3f}")

        if not args.keep_data:
            path.unlink(missing_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nЗаписано: {out_path}")

    # Подгонка показателя степени.
    print("\nПоказатель сложности b (t ∝ n^b):")
    by_alg: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r["status"] == "ok":
            by_alg.setdefault(r["algorithm"], []).append((r["n"], r["seconds"]))
    for alg, pairs in by_alg.items():
        if len(pairs) >= 3:
            ns = np.array([p[0] for p in pairs], dtype=float)
            ts = np.array([p[1] for p in pairs], dtype=float)
            mask = (ns > 0) & (ts > 0)
            if mask.sum() >= 3:
                b, log_a = np.polyfit(np.log(ns[mask]), np.log(ts[mask]), 1)
                print(f"  {alg}: b = {b:.2f}")


if __name__ == "__main__":
    main()
