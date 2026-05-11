from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from densired_style_generator import generate_densired_style_dataset, save_dataset as save_densired_dataset
from repliclust_style_generator import generate_archetype_dataset, save_dataset as save_repliclust_dataset
from simple_dataset_generator import generate_simple_gaussian_dataset, save_dataset as save_simple_dataset

def generate(output_dir: Path, base_clusterings: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simple_configs = [('analysis_simple_separated', dict(n_samples=3000, n_clusters=6, dim=2, cluster_std=0.45, separation=5.0, imbalance_ratio=1.0, seed=101)), ('analysis_simple_overlap', dict(n_samples=3500, n_clusters=6, dim=2, cluster_std=1.2, separation=2.2, imbalance_ratio=1.5, seed=102)), ('analysis_highdim', dict(n_samples=3500, n_clusters=8, dim=20, cluster_std=0.9, separation=4.0, imbalance_ratio=2.0, seed=103)), ('analysis_imbalanced', dict(n_samples=3500, n_clusters=8, dim=5, cluster_std=0.8, separation=3.5, imbalance_ratio=5.0, seed=104))]
    for name, cfg in simple_configs:
        path = output_dir / f'{name}.npz'
        if path.exists():
            print(f'SKIP {path.name}')
            continue
        x, gt, members, meta = generate_simple_gaussian_dataset(name=name, base_clusterings=base_clusterings, base_k_min=2, base_k_max=max(8, cfg['n_clusters'] + 3), base_strategy='mixed', **cfg)
        save_simple_dataset(path, x, gt, members, meta)
        print(f'OK   {path.name:<35} X={x.shape} members={members.shape} classes={np.unique(gt).size}')
    densired_configs = [('analysis_densired_compact', dict(dim=2, clunum=8, core_num=120, data_num=3200, seed=201, domain_size=20, radius=0.035, step=0.05, noise_ratio=0.08, density_factors=[1.0, 0.6, 1.4, 0.8, 1.8, 0.5, 1.2, 0.9], momentum=0.0, branch=0.0, star=0.0, distribution='uniform')), ('analysis_densired_stretched', dict(dim=8, clunum=8, core_num=140, data_num=3800, seed=202, domain_size=20, radius=0.03, step=0.065, noise_ratio=0.1, density_factors=[1.0, 0.6, 1.4, 0.8, 1.8, 0.5, 1.2, 0.9], momentum=0.75, branch=0.1, star=0.45, distribution='uniform'))]
    for name, cfg in densired_configs:
        path = output_dir / f'{name}.npz'
        if path.exists():
            print(f'SKIP {path.name}')
            continue
        x, gt, members, meta = generate_densired_style_dataset(name=name, base_clusterings=base_clusterings, base_k_min=2, base_k_max=cfg['clunum'] + 3, base_strategy='mixed', **cfg)
        save_densired_dataset(path, x, gt, members, meta)
        print(f'OK   {path.name:<35} X={x.shape} members={members.shape} classes={np.unique(gt).size}')
    repliclust_configs = [('analysis_repliclust_oblong', dict(n_clusters=6, dim=2, n_samples=3000, aspect_ref=4.0, aspect_maxmin=6.0, radius_ref=1.0, radius_maxmin=2.0, min_overlap=0.08, max_overlap=0.18, imbalance_ratio=2.5, distributions=['normal', 'student_t', 'lognormal'], distribution_proportions=[0.5, 0.3, 0.2], seed=301)), ('analysis_repliclust_heterogeneous', dict(n_clusters=8, dim=6, n_samples=4000, aspect_ref=3.5, aspect_maxmin=8.0, radius_ref=1.0, radius_maxmin=3.0, min_overlap=0.05, max_overlap=0.22, imbalance_ratio=3.5, distributions=['normal', 'student_t', 'exponential', 'lognormal'], distribution_proportions=[0.35, 0.25, 0.2, 0.2], seed=302))]
    for name, cfg in repliclust_configs:
        path = output_dir / f'{name}.npz'
        if path.exists():
            print(f'SKIP {path.name}')
            continue
        x, gt, members, meta = generate_archetype_dataset(name=name, base_clusterings=base_clusterings, base_k_min=2, base_k_max=cfg['n_clusters'] + 3, base_strategy='mixed', **cfg)
        save_repliclust_dataset(path, x, gt, members, meta)
        print(f'OK   {path.name:<35} X={x.shape} members={members.shape} classes={np.unique(gt).size}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=Path(__file__).resolve().parents[1] / 'datasets')
    parser.add_argument('--base-clusterings', type=int, default=40)
    args = parser.parse_args()
    generate(Path(args.output), args.base_clusterings)
if __name__ == '__main__':
    main()
