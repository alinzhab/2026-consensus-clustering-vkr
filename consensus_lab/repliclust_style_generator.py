import argparse
import json
from pathlib import Path

import numpy as np

from base_clusterings import build_base_clusterings as build_base_clusterings_from_features


def normalize_features(x):
    mins = np.min(x, axis=0, keepdims=True)
    spans = np.max(x, axis=0, keepdims=True) - mins
    spans[spans == 0] = 1.0
    return (x - mins) / spans


def random_orthogonal_matrix(dim, rng):
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def sample_maxmin_values(k, ref, maxmin, rng):
    if k == 1:
        return np.array([ref], dtype=np.float64)
    low = ref / np.sqrt(maxmin)
    high = ref * np.sqrt(maxmin)
    vals = rng.uniform(low, high, size=k)
    vals = vals / np.exp(np.mean(np.log(vals))) * ref
    vals = np.clip(vals, low, high)
    vals[0] = low
    vals[-1] = high
    vals = vals / np.exp(np.mean(np.log(vals))) * ref
    return vals


def build_axis_lengths(dim, radius, aspect, rng):
    raw = rng.uniform(1.0 / aspect, aspect, size=dim)
    raw = raw / np.exp(np.mean(np.log(raw)))
    return radius * raw


def choose_cluster_distributions(distributions, distribution_proportions, k, rng):
    probs = np.asarray(distribution_proportions, dtype=np.float64)
    probs = probs / probs.sum()
    idx = rng.choice(len(distributions), size=k, replace=True, p=probs)
    return [distributions[i] for i in idx]


def approximate_overlap(center_i, center_j, axes_i, axes_j):
    diff = center_j - center_i
    direction = diff / max(np.linalg.norm(diff), 1e-12)
    spread_i = np.sqrt(np.sum((axes_i @ direction) ** 2))
    spread_j = np.sqrt(np.sum((axes_j @ direction) ** 2))
    separation = np.linalg.norm(diff) / max(spread_i + spread_j, 1e-12)
    return 2.0 / (1.0 + np.exp(2.3 * (separation - 1.6)))


def place_centers(k, dim, axes_bank, min_overlap, max_overlap, domain_scale, rng):
    centers = []
    max_tries = 5000
    for i in range(k):
        placed = False
        for _ in range(max_tries):
            candidate = rng.uniform(-domain_scale, domain_scale, size=dim)
            if not centers:
                centers.append(candidate)
                placed = True
                break
            overlaps = []
            for j, center_j in enumerate(centers):
                overlap = approximate_overlap(candidate, center_j, axes_bank[i], axes_bank[j])
                overlaps.append(overlap)
            overlaps = np.asarray(overlaps)
            if np.all(overlaps <= max_overlap) and np.any(overlaps >= min_overlap):
                centers.append(candidate)
                placed = True
                break
        if not placed:
            centers.append(rng.uniform(-domain_scale, domain_scale, size=dim))
    return np.vstack(centers)


def sample_cluster_sizes(n_samples, k, imbalance_ratio, rng):
    weights = sample_maxmin_values(k, ref=1.0, maxmin=imbalance_ratio, rng=rng)
    weights = weights / weights.sum()
    counts = np.floor(weights * n_samples).astype(int)
    while counts.sum() < n_samples:
        counts[np.argmax(weights - counts / max(n_samples, 1))] += 1
    return counts


def sample_radii(distribution, n, rng):
    if distribution == "normal":
        return np.abs(rng.normal(size=n))
    if distribution == "lognormal":
        return rng.lognormal(mean=0.0, sigma=0.55, size=n)
    if distribution == "exponential":
        return rng.exponential(scale=1.0, size=n)
    if distribution == "student_t":
        return np.abs(rng.standard_t(df=3, size=n))
    if distribution == "uniform":
        return rng.uniform(0.0, 1.5, size=n)
    return np.abs(rng.normal(size=n))


def sample_cluster_points(center, rotation, axis_lengths, distribution, n, rng):
    dim = len(center)
    directions = rng.normal(size=(n, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = directions / norms
    radii = sample_radii(distribution, n, rng)[:, None]
    local = directions * radii * axis_lengths[None, :]
    return local @ rotation.T + center


def generate_archetype_dataset(
    name,
    n_clusters,
    dim,
    n_samples,
    aspect_ref,
    aspect_maxmin,
    radius_ref,
    radius_maxmin,
    min_overlap,
    max_overlap,
    imbalance_ratio,
    distributions,
    distribution_proportions,
    seed,
    base_clusterings=30,
    base_k_min=None,
    base_k_max=None,
    base_strategy="mixed",
):
    rng = np.random.default_rng(seed)
    aspects = sample_maxmin_values(n_clusters, aspect_ref, aspect_maxmin, rng)
    radii = sample_maxmin_values(n_clusters, radius_ref, radius_maxmin, rng)
    rotations = []
    axis_lengths = []
    for i in range(n_clusters):
        rotations.append(random_orthogonal_matrix(dim, rng))
        axis_lengths.append(build_axis_lengths(dim, radii[i], aspects[i], rng))
    centers = place_centers(
        k=n_clusters,
        dim=dim,
        axes_bank=[np.diag(a) @ r.T for a, r in zip(axis_lengths, rotations)],
        min_overlap=min_overlap,
        max_overlap=max_overlap,
        domain_scale=4.0 * np.max(radii) * np.sqrt(dim) * n_clusters / 4.0,
        rng=rng,
    )
    dists = choose_cluster_distributions(distributions, distribution_proportions, n_clusters, rng)
    counts = sample_cluster_sizes(n_samples, n_clusters, imbalance_ratio, rng)
    x_parts = []
    gt_parts = []
    for i in range(n_clusters):
        pts = sample_cluster_points(centers[i], rotations[i], axis_lengths[i], dists[i], counts[i], rng)
        x_parts.append(pts)
        gt_parts.append(np.full(counts[i], i + 1, dtype=np.int64))
    x = np.vstack(x_parts)
    gt = np.concatenate(gt_parts)
    x = normalize_features(x)
    base_k_min = max(2, n_clusters - 2) if base_k_min is None else int(base_k_min)
    base_k_max = n_clusters + 2 if base_k_max is None else int(base_k_max)
    members, base_info = build_base_clusterings_from_features(
        x,
        n_clusterings=base_clusterings,
        k_min=base_k_min,
        k_max=base_k_max,
        rng=rng,
        strategy=base_strategy,
        return_info=True,
    )
    meta = {
        "name": name,
        "n_clusters": n_clusters,
        "dim": dim,
        "n_samples": n_samples,
        "aspect_ref": aspect_ref,
        "aspect_maxmin": aspect_maxmin,
        "radius_ref": radius_ref,
        "radius_maxmin": radius_maxmin,
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "imbalance_ratio": imbalance_ratio,
        "distributions": distributions,
        "distribution_proportions": distribution_proportions,
        "seed": seed,
        "base_clusterings": base_clusterings,
        "base_k_min": base_k_min,
        "base_k_max": base_k_max,
        "base_strategy": base_strategy,
        "base_info": base_info,
    }
    return x, gt, members, meta


def get_presets():
    return {
        "repliclust_oblong_overlap": {
            "n_clusters": 6,
            "dim": 2,
            "n_samples": 2500,
            "aspect_ref": 4.0,
            "aspect_maxmin": 6.0,
            "radius_ref": 1.0,
            "radius_maxmin": 2.0,
            "min_overlap": 0.08,
            "max_overlap": 0.18,
            "imbalance_ratio": 2.5,
            "distributions": ["normal", "student_t", "lognormal"],
            "distribution_proportions": [0.5, 0.3, 0.2],
            "seed": 11,
        },
        "repliclust_heterogeneous_hard": {
            "n_clusters": 8,
            "dim": 5,
            "n_samples": 3200,
            "aspect_ref": 3.5,
            "aspect_maxmin": 8.0,
            "radius_ref": 1.0,
            "radius_maxmin": 3.0,
            "min_overlap": 0.05,
            "max_overlap": 0.22,
            "imbalance_ratio": 3.5,
            "distributions": ["normal", "student_t", "exponential", "lognormal"],
            "distribution_proportions": [0.35, 0.25, 0.2, 0.2],
            "seed": 17,
        },
        "repliclust_highdim_hard": {
            "n_clusters": 10,
            "dim": 12,
            "n_samples": 4000,
            "aspect_ref": 2.5,
            "aspect_maxmin": 5.0,
            "radius_ref": 1.0,
            "radius_maxmin": 2.5,
            "min_overlap": 0.04,
            "max_overlap": 0.16,
            "imbalance_ratio": 4.0,
            "distributions": ["normal", "student_t", "uniform"],
            "distribution_proportions": [0.5, 0.35, 0.15],
            "seed": 23,
        },
    }


def save_dataset(path, x, gt, members, meta):
    np.savez(path, X=x, gt=gt, members=members, meta=json.dumps(meta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(get_presets().keys()) + ["all"], default="all")
    parser.add_argument("--output", default=Path(__file__).resolve().parents[1] / "datasets")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    presets = get_presets()
    selected = presets.items() if args.preset == "all" else [(args.preset, presets[args.preset])]
    for name, cfg in selected:
        x, gt, members, meta = generate_archetype_dataset(name=name, **cfg)
        out_path = output_dir / f"{name}.npz"
        save_dataset(out_path, x, gt, members, meta)
        print(out_path.name, x.shape, members.shape, len(np.unique(gt)))


if __name__ == "__main__":
    main()
