import argparse
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.io import loadmat
from scipy.spatial.distance import squareform


def load_dataset_full(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() == ".mat":
        data = loadmat(dataset_path)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        x = None
        for key in ("X", "x", "data", "fea", "features"):
            if key in data:
                x = np.asarray(data[key], dtype=np.float64)
                break
    elif dataset_path.suffix.lower() == ".npz":
        data = np.load(dataset_path, allow_pickle=True)
        members = np.asarray(data["members"], dtype=np.int64)
        gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        x = np.asarray(data["X"], dtype=np.float64) if "X" in data.files else None
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    if gt.min() == 0:
        gt = gt + 1
    return members, gt, x


def load_dataset(dataset_path):
    members, gt, _ = load_dataset_full(dataset_path)
    return members, gt


def get_all_segs(base_cls):
    n, m = base_cls.shape
    n_cls_orig = base_cls.max(axis=0).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(n_cls_orig)[:-1]))
    bcs = base_cls + offsets
    n_cls = int(np.cumsum(n_cls_orig)[-1])
    base_cls_segs = np.zeros((n_cls, n), dtype=np.float64)
    rows = bcs.ravel(order="F") - 1
    cols = np.tile(np.arange(n), m)
    base_cls_segs[rows, cols] = 1.0
    return bcs, base_cls_segs


def compute_norm_k(bcs):
    clust = bcs.max(axis=0).astype(np.int64)
    norm_k = np.zeros(int(clust[-1]), dtype=np.float64)
    norm_k[: clust[0]] = clust[0]
    for i in range(len(clust) - 1):
        start = clust[i]
        end = clust[i + 1]
        norm_k[start:end] = end - start
    return norm_k


def get_one_cls_entropy(part_bcs, norm_k_i):
    entropy = 0.0
    for i in range(part_bcs.shape[1]):
        tmp = np.sort(part_bcs[:, i])
        unique_vals, counts = np.unique(tmp, return_counts=True)
        if unique_vals.size <= 1:
            continue
        probs = counts / counts.sum()
        entropy -= np.sum(probs * np.log2(probs))
    denom = np.log2(norm_k_i)
    if denom == 0:
        return 0.0
    return entropy / denom


def get_all_cls_entropy(bcs, base_cls_segs):
    base_cls_segs_t = base_cls_segs.T
    _, n_cls = base_cls_segs_t.shape
    entropies = np.zeros(n_cls, dtype=np.float64)
    norm_k = compute_norm_k(bcs)
    for i in range(n_cls):
        part_bcs = bcs[base_cls_segs_t[:, i] != 0, :]
        entropies[i] = get_one_cls_entropy(part_bcs, norm_k[i])
    return entropies


def compute_neci(bcs, base_cls_segs, para_theta):
    m = bcs.shape[1]
    ets = get_all_cls_entropy(bcs, base_cls_segs)
    return np.exp(-ets / para_theta / m)


def compute_nwca(base_cls_segs, neci, m):
    base_cls_segs_t = base_cls_segs.T
    nwca = (base_cls_segs_t * neci) @ base_cls_segs_t.T / m
    max_value = np.max(nwca)
    if max_value > 0:
        nwca = nwca / max_value
    np.fill_diagonal(nwca, 1.0)
    return nwca


def simxjac(a, b=None):
    if b is None:
        b = a
    temp = a @ b.T
    asquared = np.sum(a**2, axis=1, keepdims=True)
    bsquared = np.sum(b**2, axis=1, keepdims=True).T
    denom = asquared + bsquared - temp
    out = np.zeros_like(temp, dtype=np.float64)
    np.divide(temp, denom, out=out, where=denom != 0)
    return out


def random_walk_of_cluster(w, k=20, beta=1.0):
    n = w.shape[0]
    w = w.copy()
    np.fill_diagonal(w, 0.0)
    col_sums = np.sum(w, axis=0)
    d = np.zeros((n, n), dtype=np.float64)
    valid = col_sums != 0
    d[valid, valid] = 1.0 / col_sums[valid]
    w_tilde = d @ w
    tmp_o = w_tilde.copy()
    o_tilde = w_tilde @ w_tilde.T
    for _ in range(k - 1):
        tmp_o = tmp_o @ w_tilde
        o_tilde = o_tilde + beta * (tmp_o @ tmp_o.T)
    diag_vals = np.diag(o_tilde)
    denom = np.sqrt(np.outer(diag_vals, diag_vals))
    r = np.zeros_like(o_tilde, dtype=np.float64)
    np.divide(o_tilde, denom, out=r, where=denom != 0)
    isolated_idx = np.where(np.sum(w_tilde, axis=1) < 1e-9)[0]
    if isolated_idx.size > 0:
        r[isolated_idx, :] = 0.0
        r[:, isolated_idx] = 0.0
    return r


def compute_d(bcs, base_cls_segs, tau=0.8):
    n, m = bcs.shape
    d = np.zeros((n, n), dtype=np.float64)
    sim_of_cluster = simxjac(base_cls_segs)
    rw_of_cluster = random_walk_of_cluster(sim_of_cluster)
    np.fill_diagonal(rw_of_cluster, 1.0)
    dis_of_cluster = 1.0 - rw_of_cluster
    for j in range(m):
        idx = bcs[:, j].astype(np.int64) - 1
        d = d + dis_of_cluster[np.ix_(idx, idx)]
    d = d / m
    d[d < tau] = 0.0
    return d


def compute_s(nwca, mla):
    ml = nwca.copy()
    ml[mla == 0] = 0.0
    min_value = np.min(ml)
    ml = ml - min_value
    max_value = np.max(ml)
    if max_value > 0:
        ml = ml / max_value
    ml = ml / 5.0 + 0.8
    ml[ml == 0.8] = 0.0
    return ml


def optimize_sdgca(l_matrix, ml, cl, max_iter=300):
    n = l_matrix.shape[0]
    identity = np.eye(n, dtype=np.float64)
    mu1 = 1.0
    mu2 = 1.0
    s = np.zeros((n, n), dtype=np.float64)
    d = np.zeros((n, n), dtype=np.float64)
    y1 = np.zeros((n, n), dtype=np.float64)
    y2 = np.zeros((n, n), dtype=np.float64)
    f1 = np.zeros((n, n), dtype=np.float64)
    f2 = np.zeros((n, n), dtype=np.float64)
    for _ in range(max_iter):
        s_prev = s.copy()
        left_s = (2 * mu1 * f1 - d.T - y1).T
        s = np.linalg.solve((2 * l_matrix + 2 * mu1 * identity).T, left_s).T
        f1 = y1 / (2 * mu1) + s
        f1[ml > 0] = 0.0
        f1 = np.clip(f1, 0.0, 1.0)
        f1 = f1 + ml
        f1 = (f1 + f1.T) / 2.0
        d_prev = d.copy()
        left_d = (2 * mu2 * f2 - s.T - y2).T
        d = np.linalg.solve((2 * l_matrix + 2 * mu2 * identity).T, left_d).T
        f2 = y2 / (2 * mu2) + d
        f2[cl > 0] = 0.0
        f2 = np.clip(f2, 0.0, 1.0)
        f2 = f2 + cl
        f2 = (f2 + f2.T) / 2.0
        y1 = y1 + mu1 * (s - f1)
        y2 = y2 + mu2 * (d - f2)
        mu1 = min(mu1 * 1.1, 1e6)
        mu2 = min(mu2 * 1.1, 1e6)
        errors = np.array(
            [
                np.linalg.norm(s - s_prev, ord="fro"),
                np.linalg.norm(d - d_prev, ord="fro"),
                np.linalg.norm(s - f1, ord="fro"),
                np.linalg.norm(d - f2, ord="fro"),
            ]
        )
        if np.max(errors) < 1e-3:
            break
    return s, d


def compute_w(s, d, w):
    s = np.clip(s, 0.0, 1.0)
    s = (s + s.T) / 2.0
    d = np.clip(d, 0.0, 1.0)
    d = (d + d.T) / 2.0
    w1 = 1.0 - (1.0 - s + d) * (1.0 - w)
    w2 = (1.0 + s - d) * w
    flag = s - d
    w1[flag < 0] = 0.0
    w2[flag >= 0] = 0.0
    return w1 + w2


def get_cls_result(ca, cls_num, method="average"):
    ca = np.clip(ca, 0.0, 1.0)
    ca = np.maximum(ca, ca.T)
    matrix = ca.copy()
    np.fill_diagonal(matrix, 0.0)
    similarity = squareform(matrix, checks=False)
    distance = 1.0 - similarity
    z = linkage(distance, method=method)
    return fcluster(z, t=cls_num, criterion="maxclust").astype(np.int64)


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


def run_sdgca(dataset_path, data_name=None, seed=19, m=40, cnt_times=20, nwca_para=0.09, eta=0.75, theta=0.65, method="average"):
    members, gt = load_dataset(dataset_path)
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
        ca = base_cls_segs.T @ base_cls_segs / m
        nwca = compute_nwca(base_cls_segs, compute_neci(bcs, base_cls_segs, nwca_para), m)
        if eta > 1:
            result = get_cls_result(nwca, cls_nums, method=method)
        else:
            hc = ca.copy()
            hc[hc < eta] = 0.0
            l_matrix = np.diag(np.sum(hc, axis=1)) - hc
            mla = ca.copy()
            mla[mla < theta] = 0.0
            ml = compute_s(nwca, mla)
            cl = compute_d(bcs, base_cls_segs)
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
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--lambda_", type=float, default=0.09)
    parser.add_argument("--eta", type=float, default=0.75)
    parser.add_argument("--theta", type=float, default=0.65)
    parser.add_argument("--method", default="average", choices=["average", "complete", "single", "ward"])
    args = parser.parse_args()
    dataset_path = Path(args.root) / f"{args.dataset}.mat"
    if not dataset_path.exists():
        dataset_path = Path(args.root) / f"{args.dataset}.npz"
    result = run_sdgca(
        dataset_path=dataset_path,
        data_name=args.dataset,
        seed=args.seed,
        m=args.m,
        cnt_times=args.runs,
        nwca_para=args.lambda_,
        eta=args.eta,
        theta=args.theta,
        method=args.method,
    )
    print("           mean    variance")
    print(f"NMI       {result['nmi_mean']:.3f}     {result['nmi_std']:.3f}")
    print(f"ARI       {result['ari_mean']:.3f}     {result['ari_std']:.3f}")
    print(f"F-score   {result['f_mean']:.3f}     {result['f_std']:.3f}")


if __name__ == "__main__":
    main()
