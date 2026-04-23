import argparse
from pathlib import Path

from sdgca import run_sdgca


DEFAULT_PARAMS = {
    "Ecoli": {"lambda_": 0.09, "eta": 0.65, "theta": 0.75},
    "GLIOMA": {"lambda_": 0.02, "eta": 0.8, "theta": 0.6},
    "Aggregation": {"lambda_": 0.08, "eta": 0.65, "theta": 0.7},
    "MF": {"lambda_": 0.05, "eta": 0.75, "theta": 0.95},
    "IS": {"lambda_": 0.03, "eta": 0.9, "theta": 0.95},
    "MNIST": {"lambda_": 0.07, "eta": 0.95, "theta": 0.95},
    "Texture": {"lambda_": 0.04, "eta": 1.0, "theta": 1.0},
    "SPF": {"lambda_": 0.06, "eta": 0.7, "theta": 0.9},
    "ODR": {"lambda_": 0.06, "eta": 0.95, "theta": 0.95},
    "LS": {"lambda_": 0.18, "eta": 0.7, "theta": 0.6},
    "ISOLET": {"lambda_": 0.06, "eta": 0.95, "theta": 0.95},
    "USPS": {"lambda_": 0.06, "eta": 0.95, "theta": 0.95},
    "orlraws10P": {"lambda_": 0.95, "eta": 1.01, "theta": 1.01},
    "BBC": {"lambda_": 0.06, "eta": 1.01, "theta": 1.01},
    "Lung": {"lambda_": 0.75, "eta": 0.75, "theta": 0.6},
    "densired_compact_hard": {"lambda_": 0.08, "eta": 0.72, "theta": 0.78},
    "densired_stretched_hard": {"lambda_": 0.06, "eta": 0.82, "theta": 0.88},
    "densired_mix_hard": {"lambda_": 0.07, "eta": 0.75, "theta": 0.82},
}


def resolve_params(dataset_name, lambda_override, eta_override, theta_override):
    params = DEFAULT_PARAMS.get(dataset_name, {"lambda_": 0.09, "eta": 0.75, "theta": 0.65}).copy()
    if lambda_override is not None:
        params["lambda_"] = lambda_override
    if eta_override is not None:
        params["eta"] = eta_override
    if theta_override is not None:
        params["theta"] = theta_override
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Ecoli")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--lambda_", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single"])
    args = parser.parse_args()
    params = resolve_params(args.dataset, args.lambda_, args.eta, args.theta)
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_sdgca(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        nwca_para=params["lambda_"],
        eta=params["eta"],
        theta=params["theta"],
        method=args.method,
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
