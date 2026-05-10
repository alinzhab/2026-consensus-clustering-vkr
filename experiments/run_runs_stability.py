"""Эксперимент: насколько меняется метрика при увеличении числа запусков.

Закрывает претензию рецензента про малую выборку запусков (`runs=5`).
Прогоняет фиксированный алгоритм/датасет с `runs ∈ {5, 10, 20, 30, 50}`,
сохраняет среднее, std и bootstrap-CI 95% для каждого значения runs.

Это даёт график стабилизации метрики и явный ответ на вопрос «сколько
запусков достаточно для надёжного среднего». Результат — табличка и
PNG-график, которые идут в раздел «Ограничения» текста ВКР.

Запуск (Windows):
    .\\.venv\\Scripts\\Activate.ps1
    python experiments\\run_runs_stability.py --algorithm hierarchical_weighted --dataset Ecoli
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consensus_lab"))


def _bootstrap_ci(arr: np.ndarray, n_resamples: int = 5000, rng=None) -> tuple[float, float]:
    gen = np.random.default_rng(rng)
    n = arr.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = gen.integers(0, n, size=n)
        means[i] = float(arr[idx].mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=[
            "hierarchical_baseline",
            "hierarchical_weighted",
            "sdgca",
            "sdgca_modified",
        ],
        default="hierarchical_weighted",
    )
    parser.add_argument("--dataset", default="Ecoli")
    parser.add_argument("--root", default=ROOT / "datasets")
    parser.add_argument("--method", default="average")
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--out",
        default=ROOT / "results" / "runs_stability.tsv",
    )
    args = parser.parse_args()

    # Загружаем датасет один раз, прогоняем по нему один большой `cnt_times` —
    # потом анализируем подвыборки разной длины из накопленного массива метрик.
    if args.algorithm == "hierarchical_baseline":
        from hierarchical_consensus import run_hierarchical_consensus

        run_kwargs = {}
        runner = run_hierarchical_consensus
    elif args.algorithm == "hierarchical_weighted":
        from hierarchical_consensus_modified import run_weighted_hierarchical_consensus

        run_kwargs = {}
        runner = run_weighted_hierarchical_consensus
    elif args.algorithm == "sdgca":
        from sdgca import run_sdgca

        run_kwargs = {}
        runner = run_sdgca
    else:
        from sdgca_modified import run_sdgca_modified, resolve_params

        params = resolve_params(args.dataset, None, None, None, None)
        run_kwargs = {
            "nwca_para": params["lambda_"],
            "eta": params["eta"],
            "theta": params["theta"],
            "diffusion_time": params["diffusion_time"],
        }
        runner = run_sdgca_modified

    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    print(f"Прогон {args.algorithm} на {dataset_path.name}, "
          f"runs={args.max_runs}, m={args.m}")
    result = runner(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.max_runs,
        method=args.method,
        **run_kwargs,
    )
    nmi_scores = np.asarray(result["nmi_scores"]) if "nmi_scores" in result else None
    if nmi_scores is None:
        # Если runner не возвращает массив — повторяем прогоны вручную с разными seed.
        nmi_scores = np.empty(args.max_runs)
        for i in range(args.max_runs):
            r = runner(
                dataset_path=dataset_path,
                data_name=args.dataset,
                seed=args.seed + i,
                m=args.m,
                cnt_times=1,
                method=args.method,
                **run_kwargs,
            )
            nmi_scores[i] = r["nmi_mean"]

    # Анализируем стабильность.
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in [5, 10, 20, 30, 50]:
        if k > nmi_scores.size:
            continue
        sub = nmi_scores[:k]
        mean = float(sub.mean())
        std = float(sub.std(ddof=1)) if k > 1 else 0.0
        lo, hi = _bootstrap_ci(sub, rng=rng)
        rows.append(
            {
                "runs": k,
                "nmi_mean": round(mean, 4),
                "nmi_std": round(std, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "ci_width": round(hi - lo, 4),
            }
        )
        print(f"  runs={k:3d}: NMI = {mean:.3f} ± {std:.3f}, CI95 = [{lo:.3f}; {hi:.3f}], width = {hi-lo:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Записано: {out_path}")

    # PNG.
    try:
        import matplotlib.pyplot as plt

        runs_arr = [r["runs"] for r in rows]
        means = [r["nmi_mean"] for r in rows]
        lows = [r["ci_low"] for r in rows]
        highs = [r["ci_high"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(runs_arr, means, "o-", label="среднее NMI", color="#3b7dd8")
        ax.fill_between(runs_arr, lows, highs, alpha=0.25, color="#3b7dd8", label="95% bootstrap CI")
        ax.set_xlabel("Число прогонов runs")
        ax.set_ylabel("NMI")
        ax.set_title(
            f"Стабилизация среднего NMI с ростом runs\n"
            f"({args.algorithm}, {args.dataset}, {args.method})"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        png_path = ROOT / "results" / "plots" / "08_runs_stability.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        print(f"График: {png_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
