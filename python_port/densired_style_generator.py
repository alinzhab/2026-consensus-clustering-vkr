import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist


def normalize_features(x):
    x = np.asarray(x, dtype=np.float64)
    mins = np.min(x, axis=0, keepdims=True)
    spans = np.max(x, axis=0, keepdims=True) - mins
    spans[spans == 0] = 1.0
    return (x - mins) / spans


def random_unit_vector(dim, rng):
    vec = rng.normal(size=dim)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return random_unit_vector(dim, rng)
    return vec / norm


def choose_anchor(cluster_cores, branch_prob, star_prob, rng):
    if len(cluster_cores) == 1:
        return 0
    u = rng.random()
    if u < star_prob:
        return 0
    if u < star_prob + branch_prob:
        return int(rng.integers(0, len(cluster_cores)))
    return len(cluster_cores) - 1


def is_far_enough(candidate, cores_by_cluster, cluster_id, cluster_radius, min_separation):
    for other_cluster_id, other_cores in cores_by_cluster.items():
        if other_cluster_id == cluster_id or len(other_cores) == 0:
            continue
        stacked = np.vstack(other_cores)
        dists = np.linalg.norm(stacked - candidate, axis=1)
        if np.min(dists) < min_separation * cluster_radius:
            return False
    return True


def grow_cluster_skeleton(dim, cluster_id, core_count, domain_size, base_radius, step, density_factor, momentum, branch_prob, star_prob, min_separation, rng, cores_by_cluster):
    cluster_cores = []
    directions = []
    for core_idx in range(core_count):
        accepted = False
        for _ in range(200):
            if core_idx == 0:
                candidate = rng.uniform(0, domain_size, size=dim)
                direction = random_unit_vector(dim, rng)
            else:
                anchor_idx = choose_anchor(cluster_cores, branch_prob, star_prob, rng)
                anchor = cluster_cores[anchor_idx]
                prev_direction = directions[anchor_idx]
                fresh_direction = random_unit_vector(dim, rng)
                direction = momentum * prev_direction + (1.0 - momentum) * fresh_direction
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0:
                    direction = fresh_direction
                else:
                    direction = direction / direction_norm
                length = step * density_factor * max(0.6, min(1.5, rng.normal(1.0, 0.2)))
                candidate = anchor + direction * length
            if is_far_enough(candidate, cores_by_cluster, cluster_id, base_radius * density_factor, min_separation):
                cluster_cores.append(candidate)
                directions.append(direction)
                accepted = True
                break
        if not accepted:
            cluster_cores.append(rng.uniform(0, domain_size, size=dim))
            directions.append(random_unit_vector(dim, rng))
    return cluster_cores


def sample_uniform_ball(center, radius, count, rng):
    dim = center.shape[0]
    u = rng.normal(size=(count, dim))
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    r = rng.random(count) ** (1.0 / dim)
    return center + radius * (u / norms) * r[:, None]


def sample_gaussian_ball(center, radius, count, rng):
    cov = np.eye(center.shape[0]) * (radius**2)
    return rng.multivariate_normal(center, cov, size=count)


def generate_points_from_skeleton(cores_by_cluster, data_num, cluster_ratios, density_factors, base_radius, noise_ratio, distribution, rng):
    dim = len(next(iter(cores_by_cluster.values()))[0])
    counts = np.floor(np.asarray(cluster_ratios) * (1.0 - noise_ratio) * data_num).astype(int)
    while counts.sum() < int(round((1.0 - noise_ratio) * data_num)):
        counts[np.argmin(counts)] += 1
    data = []
    labels = []
    for cluster_id, cluster_count in enumerate(counts):
        cores = np.vstack(cores_by_cluster[cluster_id])
        assignments = rng.integers(0, len(cores), size=cluster_count)
        for core_idx in range(len(cores)):
            core_points = np.sum(assignments == core_idx)
            if core_points == 0:
                continue
            radius = base_radius * density_factors[cluster_id]
            if distribution == "gaussian":
                pts = sample_gaussian_ball(cores[core_idx], radius, core_points, rng)
            else:
                pts = sample_uniform_ball(cores[core_idx], radius, core_points, rng)
            data.append(pts)
            labels.extend([cluster_id + 1] * len(pts))
    x = np.vstack(data) if data else np.empty((0, dim), dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if noise_ratio > 0:
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        span = maxs - mins
        noise_count = data_num - len(x)
        noise = rng.uniform(mins - 0.15 * span, maxs + 0.15 * span, size=(noise_count, dim))
        nearest = np.argmin(cdist(noise, x), axis=1)
        nearest_labels = labels[nearest]
        x = np.vstack([x, noise])
        labels = np.concatenate([labels, nearest_labels])
    return normalize_features(x), labels


def run_kmeans_labels(x, k, rng):
    _, labels = kmeans2(x, k, minit="points", iter=30, seed=rng)
    return labels.astype(np.int64) + 1


def run_hierarchical_labels(x, k, method):
    tree = linkage(x, method=method, metric="euclidean")
    return fcluster(tree, t=k, criterion="maxclust").astype(np.int64)


def build_base_clusterings(x, gt, n_clusterings, rng):
    n, dim = x.shape
    k_true = np.unique(gt).size
    members = np.zeros((n, n_clusterings), dtype=np.int64)
    hierarchical_methods = ["average", "complete", "single"]
    for j in range(n_clusterings):
        feature_count = int(rng.integers(max(2, dim // 2), dim + 1))
        feat_idx = rng.choice(dim, size=feature_count, replace=False)
        x_sub = x[:, feat_idx]
        k = int(max(2, k_true + rng.integers(-2, 3)))
        if j % 2 == 0:
            members[:, j] = run_kmeans_labels(x_sub, k, rng)
        else:
            method = hierarchical_methods[j % len(hierarchical_methods)]
            members[:, j] = run_hierarchical_labels(x_sub, k, method)
    return members


def generate_densired_style_dataset(name, dim, clunum, core_num, data_num, seed, domain_size, radius, step, noise_ratio, density_factors, momentum, branch, star, distribution="uniform"):
    rng = np.random.default_rng(seed)
    cluster_ratios = np.full(clunum, 1.0 / clunum, dtype=np.float64)
    if np.isscalar(density_factors):
        density_factors = [density_factors] * clunum
    if np.isscalar(momentum):
        momentum = [momentum] * clunum
    if np.isscalar(branch):
        branch = [branch] * clunum
    if np.isscalar(star):
        star = [star] * clunum
    if np.isscalar(core_num):
        core_counts = [core_num // clunum] * clunum
        for i in range(core_num % clunum):
            core_counts[i] += 1
    else:
        core_counts = list(core_num)
    cores_by_cluster = {}
    for cluster_id in range(clunum):
        cores_by_cluster[cluster_id] = grow_cluster_skeleton(
            dim=dim,
            cluster_id=cluster_id,
            core_count=core_counts[cluster_id],
            domain_size=domain_size,
            base_radius=radius,
            step=step,
            density_factor=density_factors[cluster_id],
            momentum=momentum[cluster_id],
            branch_prob=branch[cluster_id],
            star_prob=star[cluster_id],
            min_separation=2.2,
            rng=rng,
            cores_by_cluster=cores_by_cluster,
        )
    x, gt = generate_points_from_skeleton(
        cores_by_cluster=cores_by_cluster,
        data_num=data_num,
        cluster_ratios=cluster_ratios,
        density_factors=density_factors,
        base_radius=radius,
        noise_ratio=noise_ratio,
        distribution=distribution,
        rng=rng,
    )
    members = build_base_clusterings(x, gt, n_clusterings=30, rng=rng)
    meta = {
        "name": name,
        "dim": dim,
        "clunum": clunum,
        "core_num": core_num,
        "data_num": data_num,
        "seed": seed,
        "domain_size": domain_size,
        "radius": radius,
        "step": step,
        "noise_ratio": noise_ratio,
        "density_factors": density_factors,
        "momentum": momentum,
        "branch": branch,
        "star": star,
        "distribution": distribution,
    }
    return x, gt, members, meta


def get_hard_presets():
    return {
        "densired_compact_hard": {
            "dim": 2,
            "clunum": 10,
            "core_num": 120,
            "data_num": 3000,
            "seed": 6,
            "domain_size": 20,
            "radius": 0.035,
            "step": 0.05,
            "noise_ratio": 0.10,
            "density_factors": [1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1],
            "momentum": 0.0,
            "branch": 0.0,
            "star": 0.0,
            "distribution": "uniform",
        },
        "densired_stretched_hard": {
            "dim": 10,
            "clunum": 10,
            "core_num": 140,
            "data_num": 3500,
            "seed": 6,
            "domain_size": 20,
            "radius": 0.03,
            "step": 0.065,
            "noise_ratio": 0.10,
            "density_factors": [1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1],
            "momentum": 0.8,
            "branch": 0.10,
            "star": 1.0,
            "distribution": "uniform",
        },
        "densired_mix_hard": {
            "dim": 2,
            "clunum": 10,
            "core_num": 160,
            "data_num": 3500,
            "seed": 6,
            "domain_size": 20,
            "radius": 0.035,
            "step": 0.06,
            "noise_ratio": 0.10,
            "density_factors": [1, 1, 0.5, 0.3, 2, 1.2, 0.9, 0.6, 1.4, 1.1],
            "momentum": [0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6, 0.45, 0.7],
            "branch": [0, 0.05, 0.1, 0, 0, 0.1, 0.02, 0, 0, 0.25],
            "star": [0.0, 0.3, 1.0, 0.0, 0.15, 0.0, 0.2, 0.0, 0.0, 0.4],
            "distribution": "uniform",
        },
    }


def save_dataset(path, x, gt, members, meta):
    np.savez(path, X=x, gt=gt, members=members, meta=json.dumps(meta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(get_hard_presets().keys()) + ["all"], default="all")
    parser.add_argument("--output", default=Path(__file__).resolve().parents[1] / "datasets")
    args = parser.parse_args()
    presets = get_hard_presets()
    selected = presets.items() if args.preset == "all" else [(args.preset, presets[args.preset])]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in selected:
        x, gt, members, meta = generate_densired_style_dataset(name=name, **config)
        path = output_dir / f"{name}.npz"
        save_dataset(path, x, gt, members, meta)
        print(path.name, x.shape, members.shape, np.unique(gt).size)


if __name__ == "__main__":
    main()
