import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from sdgca import (
    compute_f_score,
    compute_nmi,
    compute_norm_k,
    compute_s,
    compute_w,
    contingency,
    get_all_segs,
    get_cls_result,
    load_dataset_full,
    optimize_sdgca,
    rand_index,
    simxjac,
)


DEFAULT_PARAMS = {
    "Ecoli": {"lambda_": 0.09, "eta": 0.65, "theta": 0.75, "diffusion_time": 1.0},
    "GLIOMA": {"lambda_": 0.02, "eta": 0.8, "theta": 0.6, "diffusion_time": 1.0},
    "Aggregation": {"lambda_": 0.08, "eta": 0.65, "theta": 0.7, "diffusion_time": 0.8},
    "Lung": {"lambda_": 0.75, "eta": 0.75, "theta": 0.6, "diffusion_time": 1.2},
    "BBC": {"lambda_": 0.06, "eta": 1.01, "theta": 1.01, "diffusion_time": 1.1},
    "orlraws10P": {"lambda_": 0.95, "eta": 1.01, "theta": 1.01, "diffusion_time": 1.0},
    "densired_compact_hard": {"lambda_": 0.08, "eta": 0.72, "theta": 0.78, "diffusion_time": 0.9},
    "densired_stretched_hard": {"lambda_": 0.06, "eta": 0.82, "theta": 0.88, "diffusion_time": 1.3},
    "densired_mix_hard": {"lambda_": 0.07, "eta": 0.75, "theta": 0.82, "diffusion_time": 1.1},
}


def resolve_params(dataset_name, lambda_override, eta_override, theta_override, diffusion_override=None):
    params = DEFAULT_PARAMS.get(
        dataset_name,
        {"lambda_": 0.09, "eta": 0.75, "theta": 0.65, "diffusion_time": 1.0},
    ).copy()
    if lambda_override is not None:
        params["lambda_"] = lambda_override
    if eta_override is not None:
        params["eta"] = eta_override
    if theta_override is not None:
        params["theta"] = theta_override
    if diffusion_override is not None:
        params["diffusion_time"] = diffusion_override
    return params


def build_fuzzy_membership_matrix(x, labels):
    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroids.append(np.mean(x[mask], axis=0))
    centroids = np.asarray(centroids, dtype=np.float64)
    distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
    distances = np.maximum(distances, 1e-12)
    inverse_dist = 1.0 / distances
    memberships = inverse_dist / np.sum(inverse_dist, axis=1, keepdims=True)
    hard_positions = np.searchsorted(unique_labels, labels)
    memberships[np.arange(x.shape[0]), hard_positions] += 1e-6
    memberships = memberships / np.sum(memberships, axis=1, keepdims=True)
    return memberships, unique_labels


def compute_fuzzy_entropy_weights(base_cls, x, para_theta):
    n, m = base_cls.shape
    n_cls_orig = base_cls.max(axis=0).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(n_cls_orig)[:-1]))
    total_clusters = int(np.sum(n_cls_orig))
    fuzzy_weights = np.ones(total_clusters, dtype=np.float64)
    if x is None:
        return fuzzy_weights
    for j in range(m):
        labels = base_cls[:, j]
        memberships, unique_labels = build_fuzzy_membership_matrix(x, labels)
        denom = np.log2(max(unique_labels.size, 2))
        for label in unique_labels:
            mask = labels == label
            cluster_memberships = memberships[mask]
            entropy = -np.sum(cluster_memberships * np.log2(np.maximum(cluster_memberships, 1e-12)), axis=1)
            entropy = np.mean(entropy) / denom
            global_idx = offsets[j] + int(label) - 1
            fuzzy_weights[global_idx] = np.exp(-entropy / max(para_theta, 1e-12))
    return fuzzy_weights


def compute_modified_neci(base_cls, base_cls_segs, x, para_theta):
    cluster_sizes = np.sum(base_cls_segs, axis=1)
    fuzzy_weights = compute_fuzzy_entropy_weights(base_cls, x, para_theta)
    norm_k = compute_norm_k(base_cls + np.concatenate(([0], np.cumsum(base_cls.max(axis=0).astype(np.int64))[:-1])))
    structural_penalty = np.ones_like(cluster_sizes, dtype=np.float64)
    valid = norm_k > 1
    structural_penalty[valid] = 1.0 / np.log2(norm_k[valid])
    size_factor = np.sqrt(np.maximum(cluster_sizes, 1.0))
    size_factor = size_factor / np.max(size_factor)
    weights = fuzzy_weights * structural_penalty * size_factor
    max_weight = np.max(weights)
    if max_weight > 0:
        weights = weights / max_weight
    return np.clip(weights, 1e-6, 1.0)


def compute_nwca_modified(base_cls_segs, neci, m):
    base_cls_segs_t = base_cls_segs.T
    nwca = (base_cls_segs_t * neci) @ base_cls_segs_t.T / m
    nwca = (nwca + nwca.T) / 2.0
    max_value = np.max(nwca)
    if max_value > 0:
        nwca = nwca / max_value
    np.fill_diagonal(nwca, 1.0)
    return nwca


def graph_diffusion_of_cluster(w, diffusion_time=1.0):
    w = np.asarray(w, dtype=np.float64)
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    degree = np.sum(w, axis=1)
    inv_sqrt = np.zeros_like(degree)
    valid = degree > 1e-12
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    normalized = inv_sqrt[:, None] * w * inv_sqrt[None, :]
    laplacian = np.eye(w.shape[0], dtype=np.float64) - normalized
    heat = expm(-diffusion_time * laplacian)
    heat = (heat + heat.T) / 2.0
    diag = np.sqrt(np.maximum(np.diag(heat), 1e-12))
    heat = heat / np.outer(diag, diag)
    heat = np.clip(heat, 0.0, 1.0)
    np.fill_diagonal(heat, 1.0)
    return heat


def compute_d_diffusion(bcs, base_cls_segs, tau=0.8, diffusion_time=1.0):
    n, m = bcs.shape
    d = np.zeros((n, n), dtype=np.float64)
    sim_of_cluster = simxjac(base_cls_segs)
    diffusion_similarity = graph_diffusion_of_cluster(sim_of_cluster, diffusion_time=diffusion_time)
    dis_of_cluster = 1.0 - diffusion_similarity
    for j in range(m):
        idx = bcs[:, j].astype(np.int64) - 1
        d = d + dis_of_cluster[np.ix_(idx, idx)]
    d = d / m
    d[d < tau] = 0.0
    return d


def run_sdgca_modified(
    dataset_path,
    data_name=None,
    seed=19,
    m=20,
    cnt_times=10,
    nwca_para=0.09,
    eta=0.75,
    theta=0.65,
    method="average",
    diffusion_time=1.0,
):
    members, gt, x = load_dataset_full(dataset_path)
    cls_nums = np.unique(gt).size
    pool_size = members.shape[1]
    if m > pool_size:
        m = pool_size
    rng = np.random.default_rng(seed)
    bc_idx = np.vstack([rng.permutation(pool_size)[:m] for _ in range(cnt_times)])
    nmi_scores = np.zeros(cnt_times, dtype=np.float64)
    ari_scores = np.zeros(cnt_times, dtype=np.float64)
    f_scores = np.zeros(cnt_times, dtype=np.float64)

    for run_idx in range(cnt_times):
        base_cls = members[:, bc_idx[run_idx, :]]
        bcs, base_cls_segs = get_all_segs(base_cls)
        base_x = x if x is None else np.asarray(x, dtype=np.float64)
        ca = base_cls_segs.T @ base_cls_segs / m
        neci = compute_modified_neci(base_cls, base_cls_segs, base_x, nwca_para)
        nwca = compute_nwca_modified(base_cls_segs, neci, m)
        if eta > 1:
            result = get_cls_result(nwca, cls_nums, method=method)
        else:
            hc = ca.copy()
            hc[hc < eta] = 0.0
            l_matrix = np.diag(np.sum(hc, axis=1)) - hc
            mla = ca.copy()
            mla[mla < theta] = 0.0
            ml = compute_s(nwca, mla)
            cl = compute_d_diffusion(bcs, base_cls_segs, diffusion_time=diffusion_time)
            ml[cl > 0] = 0.0
            s, d = optimize_sdgca(l_matrix, ml, cl)
            w = compute_w(s, d, nwca)
            result = get_cls_result(w, cls_nums, method=method)
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
    }


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
    parser.add_argument("--diffusion_time", type=float, default=None)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single", "ward"])
    args = parser.parse_args()
    params = resolve_params(args.dataset, args.lambda_, args.eta, args.theta, args.diffusion_time)
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_sdgca_modified(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        nwca_para=params["lambda_"],
        eta=params["eta"],
        theta=params["theta"],
        method=args.method,
        diffusion_time=params["diffusion_time"],
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
