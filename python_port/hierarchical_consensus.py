import argparse
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.io import loadmat
from scipy.spatial.distance import squareform


def load_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() == ".mat":
        data = loadmat(dataset_path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
    elif dataset_path.suffix.lower() == ".npz":
        data = np.load(dataset_path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    if gt.min() == 0:
        gt = gt + 1
    return members, gt


def validate_members(members, m):
    members = np.asarray(members, dtype=np.int64)
    if members.ndim != 2:
        raise ValueError("members must be a 2D matrix")
    if members.shape[0] < 2:
        raise ValueError("dataset must contain at least two objects")
    if members.shape[1] < 1:
        raise ValueError("dataset must contain at least one base clustering")
    if m < 1 or m > members.shape[1]:
        raise ValueError("m must be between 1 and the number of base clusterings")
    return members


def build_coassociation_matrix(base_cls):
    n, m = base_cls.shape
    consensus = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        labels = base_cls[:, j]
        consensus += (labels[:, None] == labels[None, :]).astype(np.float64)
    consensus /= m
    consensus = (consensus + consensus.T) / 2.0
    consensus = np.clip(consensus, 0.0, 1.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def get_cls_result(consensus_matrix, cls_num, method="average"):
    consensus_matrix = np.clip(consensus_matrix, 0.0, 1.0)
    consensus_matrix = np.maximum(consensus_matrix, consensus_matrix.T)
    matrix = consensus_matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    similarity = squareform(matrix, checks=False)
    distance = 1.0 - similarity
    tree = linkage(distance, method=method)
    return fcluster(tree, t=cls_num, criterion="maxclust").astype(np.int64)


def compute_f_score(h, t):
    n = len(t)
    num_t = 0
    num_h = 0
    num_i = 0
    for idx in range(n):
        t_n = t[idx + 1 :] == t[idx]
        h_n = h[idx + 1 :] == h[idx]
        num_t += int(np.sum(t_n))
        num_h += int(np.sum(h_n))
        num_i += int(np.sum(t_n & h_n))
    precision = 1.0 if num_h == 0 else num_i / num_h
    recall = 1.0 if num_t == 0 else num_i / num_t
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def contingency(mem1, mem2):
    labels1, inv1 = np.unique(mem1, return_inverse=True)
    labels2, inv2 = np.unique(mem2, return_inverse=True)
    cont = np.zeros((labels1.size, labels2.size), dtype=np.int64)
    np.add.at(cont, (inv1, inv2), 1)
    return cont


def rand_index(c1, c2):
    c = contingency(c1, c2).astype(np.float64)
    n = np.sum(c)
    nis = np.sum(np.sum(c, axis=1) ** 2)
    njs = np.sum(np.sum(c, axis=0) ** 2)
    t1 = n * (n - 1) / 2.0
    t2 = np.sum(c**2)
    t3 = 0.5 * (nis + njs)
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    a = t1 + t2 - t3
    if t1 == nc:
        return 0.0
    return (a - nc) / (t1 - nc)


def compute_nmi(h, t):
    n = len(t)
    classes = np.unique(t)
    clusters = np.unique(h)
    d = np.array([np.sum(t == cls) for cls in classes], dtype=np.float64)
    mi = 0.0
    b = np.zeros(clusters.size, dtype=np.float64)
    for i, cluster in enumerate(clusters):
        index_clust = h == cluster
        b[i] = np.sum(index_clust)
        for j, cls in enumerate(classes):
            index_class = t == cls
            a = np.sum(index_class & index_clust)
            if a != 0:
                mi += a / n * np.log2(n * a / (b[i] * d[j]))
    class_ent = np.sum(d / n * np.log2(n / d))
    clust_ent = np.sum(b / n * np.log2(n / b))
    return 2.0 * mi / (clust_ent + class_ent)


def run_hierarchical_consensus(dataset_path, data_name=None, seed=19, m=40, cnt_times=20, method="average"):
    members, gt = load_dataset(dataset_path)
    members = validate_members(members, m)
    cls_nums = np.unique(gt).size
    pool_size = members.shape[1]
    rng = np.random.default_rng(seed)
    bc_idx = np.vstack([rng.permutation(pool_size)[:m] for _ in range(cnt_times)])
    nmi_scores = np.zeros(cnt_times, dtype=np.float64)
    ari_scores = np.zeros(cnt_times, dtype=np.float64)
    f_scores = np.zeros(cnt_times, dtype=np.float64)
    for run_idx in range(cnt_times):
        base_cls = members[:, bc_idx[run_idx, :]]
        consensus = build_coassociation_matrix(base_cls)
        result = get_cls_result(consensus, cls_nums, method=method)
        nmi_scores[run_idx] = compute_nmi(result, gt)
        ari_scores[run_idx] = rand_index(result, gt)
        f_scores[run_idx] = compute_f_score(result, gt)
    return {
        "data_name": data_name or Path(dataset_path).stem,
        "nmi_mean": float(np.mean(nmi_scores)),
        "nmi_std": float(np.std(nmi_scores)),
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "f_mean": float(np.mean(f_scores)),
        "f_std": float(np.std(f_scores)),
        "nmi_scores": nmi_scores,
        "ari_scores": ari_scores,
        "f_scores": f_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Ecoli")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single", "ward"])
    args = parser.parse_args()
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_hierarchical_consensus(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        method=args.method,
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
