import argparse
from pathlib import Path

import numpy as np

from hierarchical_consensus import compute_f_score, compute_nmi, get_cls_result, load_dataset, rand_index, validate_members


def build_partition_matrix(labels):
    return (labels[:, None] == labels[None, :]).astype(np.float64)


def compute_base_clustering_weights(base_cls):
    m = base_cls.shape[1]
    partition_matrices = [build_partition_matrix(base_cls[:, j]) for j in range(m)]
    agreement = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            score = np.mean(partition_matrices[i] == partition_matrices[j])
            agreement[i, j] = score
            agreement[j, i] = score
    weights = np.mean(agreement, axis=1)
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        return np.full(m, 1.0 / m, dtype=np.float64)
    return weights / weights_sum


def build_weighted_consensus_matrix(base_cls, sharpen=1.0):
    n, m = base_cls.shape
    weights = compute_base_clustering_weights(base_cls)
    if sharpen != 1.0:
        weights = weights**sharpen
        weights = weights / np.sum(weights)
    consensus = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        consensus += weights[j] * build_partition_matrix(base_cls[:, j])
    consensus = (consensus + consensus.T) / 2.0
    consensus = np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus, weights


def run_weighted_hierarchical_consensus(dataset_path, data_name=None, seed=19, m=40, cnt_times=20, method="average", sharpen=1.0):
    members, gt = load_dataset(dataset_path)
    members = validate_members(members, m)
    cls_nums = np.unique(gt).size
    pool_size = members.shape[1]
    rng = np.random.default_rng(seed)
    bc_idx = np.vstack([rng.permutation(pool_size)[:m] for _ in range(cnt_times)])
    nmi_scores = np.zeros(cnt_times, dtype=np.float64)
    ari_scores = np.zeros(cnt_times, dtype=np.float64)
    f_scores = np.zeros(cnt_times, dtype=np.float64)
    weight_bank = []
    for run_idx in range(cnt_times):
        base_cls = members[:, bc_idx[run_idx, :]]
        consensus, weights = build_weighted_consensus_matrix(base_cls, sharpen=sharpen)
        result = get_cls_result(consensus, cls_nums, method=method)
        nmi_scores[run_idx] = compute_nmi(result, gt)
        ari_scores[run_idx] = rand_index(result, gt)
        f_scores[run_idx] = compute_f_score(result, gt)
        weight_bank.append(weights)
    return {
        "data_name": data_name or Path(dataset_path).stem,
        "nmi_mean": float(np.mean(nmi_scores)),
        "nmi_std": float(np.std(nmi_scores)),
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "f_mean": float(np.mean(f_scores)),
        "f_std": float(np.std(f_scores)),
        "avg_weights": np.mean(np.vstack(weight_bank), axis=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Ecoli")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single", "ward"])
    parser.add_argument("--sharpen", type=float, default=1.0)
    args = parser.parse_args()
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_weighted_hierarchical_consensus(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        method=args.method,
        sharpen=args.sharpen,
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
