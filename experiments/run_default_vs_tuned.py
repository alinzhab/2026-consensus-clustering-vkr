"""Эксперимент: SDGCA modified с подобранными vs дефолтными гиперпараметрами.

Закрывает претензию рецензента про data leakage в `DEFAULT_PARAMS`:
словарь в `consensus_lab/sdgca_modified.py` содержит индивидуальные
`(lambda, eta, theta, diffusion_time)` для каждого известного датасета.
Это даёт алгоритму неявное преимущество по сравнению с `sdgca` (у которого
универсальные дефолты `lambda=0.09, eta=0.75, theta=0.65`).

Этот скрипт прогоняет `sdgca_modified` дважды для каждого датасета:
    1. С `DEFAULT_PARAMS` (текущая практика проекта).
    2. С общими дефолтами для всех датасетов (`lambda=0.09, eta=0.75,
       theta=0.65, diffusion_time=1.0`).

Если разница между двумя режимами статистически значима — значит,
`DEFAULT_PARAMS` действительно «вытягивает» результат, и сравнение
текущей версии с базовым `sdgca` нужно переинтерпретировать.

Запуск (Windows):
    .\\.venv\\Scripts\\Activate.ps1
    python experiments\\run_default_vs_tuned.py --runs 20 --m 30

Зависимости: scipy, numpy (берутся из requirements.txt).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consensus_lab"))

from sdgca_modified import DEFAULT_PARAMS, run_sdgca_modified  # noqa: E402

# Универсальные «честные» дефолты — те же, что у исходного `sdgca`.
GENERIC_DEFAULTS = {"lambda_": 0.09, "eta": 0.75, "theta": 0.65, "diffusion_time": 1.0}


def resolve_dataset_path(name: str, root: Path) -> Path:
    for ext in (".mat", ".npz"):
        p = root / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_PARAMS.keys()),
        help="Имена датасетов (без расширения). По умолчанию — все из DEFAULT_PARAMS.",
    )
    parser.add_argument("--method", default="average")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--out",
        default=ROOT / "results" / "default_vs_tuned.tsv",
    )
    parser.add_argument(
        "--root",
        default=ROOT / "datasets",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(
        f"Datasets: {len(args.datasets)} | runs={args.runs}, m={args.m}, "
        f"method={args.method}"
    )
    for name in args.datasets:
        try:
            path = resolve_dataset_path(name, Path(args.root))
        except FileNotFoundError:
            print(f"[skip] {name}: датасет не найден")
            continue
        for mode_name, params in (
            ("tuned", DEFAULT_PARAMS.get(name, GENERIC_DEFAULTS)),
            ("generic", GENERIC_DEFAULTS),
        ):
            t0 = time.time()
            try:
                result = run_sdgca_modified(
                    dataset_path=path,
                    data_name=name,
                    seed=args.seed,
                    m=args.m,
                    cnt_times=args.runs,
                    nwca_para=params["lambda_"],
                    eta=params["eta"],
                    theta=params["theta"],
                    method=args.method,
                    diffusion_time=params["diffusion_time"],
                )
                status = "ok"
                error = ""
            except Exception as exc:
                result = {
                    "nmi_mean": float("nan"),
                    "nmi_std": float("nan"),
                    "ari_mean": float("nan"),
                    "ari_std": float("nan"),
                    "f_mean": float("nan"),
                    "f_std": float("nan"),
                }
                status = "error"
                error = str(exc)
            seconds = time.time() - t0
            rows.append(
                {
                    "dataset": name,
                    "mode": mode_name,
                    "method": args.method,
                    "lambda_": params["lambda_"],
                    "eta": params["eta"],
                    "theta": params["theta"],
                    "diffusion_time": params["diffusion_time"],
                    "nmi_mean": result["nmi_mean"],
                    "nmi_std": result["nmi_std"],
                    "ari_mean": result["ari_mean"],
                    "ari_std": result["ari_std"],
                    "f_mean": result["f_mean"],
                    "f_std": result["f_std"],
                    "seconds": round(seconds, 2),
                    "runs": args.runs,
                    "m": args.m,
                    "seed": args.seed,
                    "status": status,
                    "error": error,
                }
            )
            print(
                f"  {name} [{mode_name}]: NMI = {result['nmi_mean']:.3f} ± "
                f"{result['nmi_std']:.3f}, t = {seconds:.1f}s"
            )

    if not rows:
        print("Нет данных, выход.")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nЗаписано: {out_path}")

    # Подсчитаем парную статистику.
    by_dataset: dict[str, dict] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["mode"]] = row
    deltas = []
    for ds, modes in by_dataset.items():
        if "tuned" in modes and "generic" in modes:
            deltas.append(modes["tuned"]["nmi_mean"] - modes["generic"]["nmi_mean"])
    if deltas:
        import statistics

        mean_delta = statistics.mean(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        print()
        print(
            f"Δ NMI (tuned − generic): mean = {mean_delta:+.3f}, "
            f"wins/losses = {wins}/{losses}, n = {len(deltas)}"
        )


if __name__ == "__main__":
    main()
