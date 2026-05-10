from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from base_clusterings import build_base_clusterings


def normalize_features(x: np.ndarray) -> np.ndarray:
    mins = np.min(x, axis=0, keepdims=True)
    spans = np.max(x, axis=0, keepdims=True) - mins
    spans[spans == 0] = 1.0
    return (x - mins) / spans


def as_list(value: Any, size: int, dtype=float) -> list:
    if np.isscalar(value):
        return [dtype(value)] * size
    values = list(value)
    if len(values) != size:
        raise ValueError(f"Expected {size} values, got {len(values)}")
    return [dtype(v) for v in values]


def random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        v = rng.normal(size=dim)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm


def choose_restart_anchor(n_cores: int, branch_prob: float, star_prob: float, rng: np.random.Generator) -> int:
    if n_cores <= 1:
        return 0

    u = rng.random()
    if u < star_prob:
        return 0
    if u < star_prob + branch_prob:
        return int(rng.integers(0, n_cores))
    return n_cores - 1


def is_separated_from_other_clusters(
    candidate: np.ndarray,
    cores_by_cluster: dict[int, list[np.ndarray]],
    cluster_id: int,
    epsilon: float,
    separation_factor: float,
) -> bool:
    min_allowed = 2.0 * epsilon * separation_factor

    for other_id, other_cores in cores_by_cluster.items():
        if other_id == cluster_id or not other_cores:
            continue

        other = np.vstack(other_cores)
        if np.min(np.linalg.norm(other - candidate, axis=1)) < min_allowed:
            return False

    return True


def core_components(cores: np.ndarray, epsilon: float) -> int:
    n = len(cores)
    if n <= 1:
        return 1

    dist = cdist(cores, cores)
    adj = dist <= epsilon

    visited = np.zeros(n, dtype=bool)
    components = 0

    for start in range(n):
        if visited[start]:
            continue

        components += 1
        stack = [start]
        visited[start] = True

        while stack:
            v = stack.pop()
            neighbors = np.where(adj[v] & ~visited)[0]
            visited[neighbors] = True
            stack.extend(neighbors.tolist())

    return components


def grow_cluster_skeleton(
    dim: int,
    cluster_id: int,
    core_count: int,
    domain_size: float,
    epsilon: float,
    step_ratio: float,
    momentum: float,
    branch_prob: float,
    star_prob: float,
    separation_factor: float,
    rng: np.random.Generator,
    cores_by_cluster: dict[int, list[np.ndarray]],
    max_attempts_per_core: int = 300,
    max_restarts: int = 50,
) -> list[np.ndarray]:
    step_max = epsilon * step_ratio

    for _ in range(max_restarts):
        cores: list[np.ndarray] = []
        directions: list[np.ndarray] = []

        for core_idx in range(core_count):
            accepted = False

            for _ in range(max_attempts_per_core):
                if core_idx == 0:
                    candidate = rng.uniform(0.0, domain_size, size=dim)
                    direction = random_unit_vector(dim, rng)
                else:
                    anchor_idx = choose_restart_anchor(len(cores), branch_prob, star_prob, rng)
                    anchor = cores[anchor_idx]
                    prev_direction = directions[anchor_idx]
                    fresh_direction = random_unit_vector(dim, rng)

                    direction = momentum * prev_direction + (1.0 - momentum) * fresh_direction
                    norm = np.linalg.norm(direction)
                    direction = fresh_direction if norm == 0 else direction / norm

                    step = step_max * rng.uniform(0.55, 0.95)
                    candidate = anchor + direction * step

                if is_separated_from_other_clusters(
                    candidate,
                    cores_by_cluster,
                    cluster_id,
                    epsilon,
                    separation_factor,
                ):
                    cores.append(candidate)
                    directions.append(direction)
                    accepted = True
                    break

            if not accepted:
                break

        if len(cores) == core_count:
            arr = np.vstack(cores)
            if core_components(arr, epsilon) == 1:
                return cores

    raise RuntimeError(
        f"Could not generate cluster {cluster_id}. "
        f"Try increasing domain_size or decreasing epsilon/core_count/clunum."
    )


def sample_uniform_ball(center: np.ndarray, radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    dim = center.shape[0]
    dirs = rng.normal(size=(count, dim))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    radii = rng.random(count) ** (1.0 / dim)
    return center + radius * dirs / norms * radii[:, None]


def sample_points_around_core(
    center: np.ndarray,
    ball_radius: float,
    count: int,
    rng: np.random.Generator,
    point_distribution: str = "uniform",
) -> np.ndarray:
    """Выборка «облака» вокруг ядра скелета: равномерно в шаре или изотропный Gaussian."""
    mode = point_distribution.strip().lower()
    if mode == "uniform":
        return sample_uniform_ball(center, ball_radius, count, rng)
    if mode in {"gaussian", "normal"}:
        dim = int(center.shape[0])
        sigma = ball_radius / max(float(np.sqrt(dim)), 1.0)
        return center + rng.normal(scale=sigma, size=(count, dim))
    raise ValueError(f"Unknown point_distribution: {point_distribution!r} (expect 'uniform' or 'gaussian')")


def allocate_points_per_core(total: int, n_cores: int, min_pts: int, rng: np.random.Generator) -> np.ndarray:
    min_required = n_cores * min_pts
    total = max(total, min_required)

    counts = np.full(n_cores, min_pts, dtype=np.int64)
    rest = total - min_required

    if rest > 0:
        counts += rng.multinomial(rest, np.full(n_cores, 1.0 / n_cores))

    return counts


def sample_noise(
    x: np.ndarray,
    count: int,
    epsilon: float,
    rng: np.random.Generator,
    max_batches: int = 100,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, x.shape[1]), dtype=np.float64)

    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    span = maxs - mins
    low = mins - 0.25 * span
    high = maxs + 0.25 * span

    accepted = []
    batch_size = max(256, count * 4)

    for _ in range(max_batches):
        cand = rng.uniform(low, high, size=(batch_size, x.shape[1]))
        d = cdist(cand, x)
        mask = np.min(d, axis=1) > epsilon

        for p in cand[mask]:
            accepted.append(p)
            if len(accepted) >= count:
                return np.asarray(accepted, dtype=np.float64)

    return rng.uniform(low, high, size=(count, x.shape[1]))


def instantiate_points(
    cores_by_cluster: dict[int, list[np.ndarray]],
    data_num: int,
    epsilon: float,
    min_pts: int,
    density_factors: list[float],
    noise_ratio: float,
    rng: np.random.Generator,
    point_distribution: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    clunum = len(cores_by_cluster)
    cluster_total = int(round(data_num * (1.0 - noise_ratio)))
    base_per_cluster = cluster_total // clunum
    remainder = cluster_total % clunum

    data = []
    labels = []

    for cluster_id in range(clunum):
        cores = np.vstack(cores_by_cluster[cluster_id])
        target = base_per_cluster + (1 if cluster_id < remainder else 0)

        counts = allocate_points_per_core(
            total=target,
            n_cores=len(cores),
            min_pts=min_pts,
            rng=rng,
        )

        radius = 0.5 * epsilon * density_factors[cluster_id]

        for core, count in zip(cores, counts):
            pts = sample_points_around_core(
                core, radius, int(count), rng, point_distribution=point_distribution
            )
            data.append(pts)
            labels.extend([cluster_id + 1] * len(pts))

    x = np.vstack(data)
    gt = np.asarray(labels, dtype=np.int64)

    noise_count = data_num - len(x)
    if noise_ratio > 0 and noise_count > 0:
        noise = sample_noise(x, noise_count, epsilon, rng)
        x = np.vstack([x, noise])
        gt = np.concatenate([gt, np.zeros(len(noise), dtype=np.int64)])

    return x, gt


def validate_eps_connectivity(x: np.ndarray, gt: np.ndarray, epsilon: float) -> dict:
    result = {
        "epsilon": float(epsilon),
        "noise_count": int(np.sum(gt == 0)),
        "clusters": {},
    }

    for label in sorted(int(v) for v in np.unique(gt) if v > 0):
        pts = x[gt == label]

        if len(pts) <= 1:
            result["clusters"][label] = {
                "n_points": int(len(pts)),
                "components": 1,
            }
            continue

        dist = cdist(pts, pts)
        adj = dist <= epsilon

        visited = np.zeros(len(pts), dtype=bool)
        components = 0

        for start in range(len(pts)):
            if visited[start]:
                continue

            components += 1
            stack = [start]
            visited[start] = True

            while stack:
                v = stack.pop()
                neighbors = np.where(adj[v] & ~visited)[0]
                visited[neighbors] = True
                stack.extend(neighbors.tolist())

        result["clusters"][label] = {
            "n_points": int(len(pts)),
            "components": int(components),
        }

    return result


def generate_densired_dataset(
    name: str,
    dim: int = 2,
    clunum: int = 8,
    core_num: int | list[int] = 120,
    data_num: int = 3000,
    seed: int = 19,
    domain_size: float = 30.0,
    epsilon: float = 0.55,
    step_ratio: float = 0.75,
    min_pts: int = 8,
    noise_ratio: float = 0.05,
    density_factors: float | list[float] = 1.0,
    momentum: float | list[float] = 0.5,
    branch: float | list[float] = 0.05,
    star: float | list[float] = 0.0,
    separation_factor: float = 1.15,
    normalize: bool = True,
    point_distribution: str = "uniform",
    base_clusterings: int = 30,
    base_k_min: int | None = None,
    base_k_max: int | None = None,
    base_strategy: str = "mixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    pd = str(point_distribution).strip().lower() or "uniform"
    if pd in {"normal"}:
        pd = "gaussian"
    if pd not in {"uniform", "gaussian"}:
        warnings.warn(
            f"Unknown point_distribution={point_distribution!r}, using 'uniform'.",
            RuntimeWarning,
            stacklevel=2,
        )
        pd = "uniform"

    density_factors = as_list(density_factors, clunum, float)
    momentum = as_list(momentum, clunum, float)
    branch = as_list(branch, clunum, float)
    star = as_list(star, clunum, float)

    if np.isscalar(core_num):
        total_cores = int(core_num)
        core_counts = [total_cores // clunum] * clunum
        for i in range(total_cores % clunum):
            core_counts[i] += 1
    else:
        core_counts = as_list(core_num, clunum, int)

    cores_by_cluster: dict[int, list[np.ndarray]] = {}

    for cluster_id in range(clunum):
        cores_by_cluster[cluster_id] = grow_cluster_skeleton(
            dim=dim,
            cluster_id=cluster_id,
            core_count=core_counts[cluster_id],
            domain_size=domain_size,
            epsilon=epsilon,
            step_ratio=step_ratio,
            momentum=momentum[cluster_id],
            branch_prob=branch[cluster_id],
            star_prob=star[cluster_id],
            separation_factor=separation_factor,
            rng=rng,
            cores_by_cluster=cores_by_cluster,
        )

    x_raw, gt = instantiate_points(
        cores_by_cluster=cores_by_cluster,
        data_num=data_num,
        epsilon=epsilon,
        min_pts=min_pts,
        density_factors=density_factors,
        noise_ratio=noise_ratio,
        rng=rng,
        point_distribution=pd,
    )

    diagnostics_raw = validate_eps_connectivity(x_raw, gt, epsilon)

    x = normalize_features(x_raw) if normalize else x_raw

    perm = rng.permutation(x.shape[0])
    x = x[perm]
    gt = gt[perm]

    base_k_min = max(2, clunum - 2) if base_k_min is None else int(base_k_min)
    base_k_max = clunum + 2 if base_k_max is None else int(base_k_max)

    members, base_info = build_base_clusterings(
        x,
        n_clusterings=base_clusterings,
        k_min=base_k_min,
        k_max=base_k_max,
        rng=rng,
        strategy=base_strategy,
        return_info=True,
    )

    core_diagnostics = {}
    for cluster_id, cores in cores_by_cluster.items():
        arr = np.vstack(cores)
        core_diagnostics[cluster_id + 1] = {
            "core_count": int(len(arr)),
            "core_components": int(core_components(arr, epsilon)),
        }

    meta = {
        "name": name,
        "generator": "densired_full_practical",
        "note": (
            "Practical DENSIRED-style implementation based on skeleton construction "
            "and hypersphere instantiation. Not a bitwise reproduction of the authors' code."
        ),
        "dim": int(dim),
        "clunum": int(clunum),
        "core_num": core_num,
        "core_counts": core_counts,
        "data_num_target": int(data_num),
        "data_num_actual": int(x.shape[0]),
        "seed": int(seed),
        "domain_size": float(domain_size),
        "epsilon_raw_space": float(epsilon),
        "step_ratio": float(step_ratio),
        "min_pts": int(min_pts),
        "noise_ratio": float(noise_ratio),
        "noise_label": 0,
        "density_factors": density_factors,
        "momentum": momentum,
        "branch": branch,
        "star": star,
        "separation_factor": float(separation_factor),
        "normalized": bool(normalize),
        "core_diagnostics": core_diagnostics,
        "density_diagnostics_raw_space": diagnostics_raw,
        "base_clusterings": int(base_clusterings),
        "base_k_min": int(base_k_min),
        "base_k_max": int(base_k_max),
        "base_strategy": base_strategy,
        "base_info": base_info,
        "point_distribution": pd,
    }

    return x, gt, members, meta


def generate_densired_style_dataset(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """API для Flask и скриптов экспериментов.

    Поддерживает устаревшие имена параметров из веб‑формы и пайплайнов:

    - ``radius`` → ``epsilon`` (масштаб ε‑связности и размера локальных облаков);
    - ``step`` → ``step_ratio``, как ``step / epsilon`` (с усечением в разумных пределах);
    - ``distribution`` — ``uniform`` | ``gaussian`` для выборки точек вокруг ядер.

    Если заданы и канонические ``epsilon`` / ``step_ratio``, они имеют приоритет над
    ``radius`` / ``step``.
    """
    params = dict(kwargs)
    legacy_radius = params.pop("radius", None)
    legacy_step = params.pop("step", None)
    dist_raw = params.pop("distribution", "uniform")
    point_distribution = str(dist_raw).strip().lower() or "uniform"
    if point_distribution in {"normal"}:
        point_distribution = "gaussian"

    if legacy_radius is not None and "epsilon" not in params:
        params["epsilon"] = float(legacy_radius)

    epsilon_eff = float(params.get("epsilon", 0.55))
    if legacy_step is not None and "step_ratio" not in params:
        denom = epsilon_eff if epsilon_eff > 0 else 1.0
        params["step_ratio"] = float(np.clip(float(legacy_step) / denom, 0.12, 2.5))

    params["point_distribution"] = point_distribution
    return generate_densired_dataset(**params)


def get_presets() -> dict[str, dict]:
    return {
        "densired_compact": {
            "dim": 2,
            "clunum": 8,
            "core_num": 120,
            "data_num": 3000,
            "seed": 6,
            "domain_size": 30.0,
            "epsilon": 0.55,
            "step_ratio": 0.70,
            "min_pts": 8,
            "noise_ratio": 0.05,
            "density_factors": [1.0, 1.0, 0.8, 0.7, 1.2, 1.1, 0.9, 1.3],
            "momentum": 0.0,
            "branch": 0.0,
            "star": 0.0,
            "separation_factor": 1.15,
        },
        "densired_stretched": {
            "dim": 10,
            "clunum": 8,
            "core_num": 140,
            "data_num": 3500,
            "seed": 6,
            "domain_size": 45.0,
            "epsilon": 0.65,
            "step_ratio": 0.75,
            "min_pts": 8,
            "noise_ratio": 0.05,
            "density_factors": [1.0, 1.0, 0.7, 0.6, 1.4, 1.2, 0.9, 1.3],
            "momentum": 0.85,
            "branch": 0.05,
            "star": 0.0,
            "separation_factor": 1.20,
        },
        "densired_branching": {
            "dim": 2,
            "clunum": 8,
            "core_num": 160,
            "data_num": 3500,
            "seed": 6,
            "domain_size": 35.0,
            "epsilon": 0.55,
            "step_ratio": 0.70,
            "min_pts": 8,
            "noise_ratio": 0.08,
            "density_factors": [1.0, 1.0, 0.7, 0.6, 1.4, 1.2, 0.9, 1.3],
            "momentum": [0.5, 0.75, 0.8, 0.3, 0.5, 0.4, 0.2, 0.6],
            "branch": [0.05, 0.1, 0.15, 0.05, 0.05, 0.15, 0.1, 0.2],
            "star": [0.0, 0.2, 0.6, 0.0, 0.15, 0.0, 0.2, 0.4],
            "separation_factor": 1.20,
        },
    }


def save_dataset(path: str | Path, x: np.ndarray, gt: np.ndarray, members: np.ndarray, meta: dict) -> None:
    path = Path(path)
    np.savez(
        path,
        X=x,
        gt=gt,
        members=members,
        meta=json.dumps(meta, ensure_ascii=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(get_presets().keys()) + ["all"], default="all")
    parser.add_argument("--output", default=Path(__file__).resolve().parents[1] / "datasets")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = get_presets()
    selected = presets.items() if args.preset == "all" else [(args.preset, presets[args.preset])]

    for name, config in selected:
        x, gt, members, meta = generate_densired_dataset(name=name, **config)
        out_path = output_dir / f"{name}.npz"
        save_dataset(out_path, x, gt, members, meta)
        print(out_path.name, x.shape, members.shape, "classes:", np.unique(gt).size)


if __name__ == "__main__":
    main()