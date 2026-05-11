from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSENSUS_LAB = PROJECT_ROOT / 'consensus_lab'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
sys.path.insert(0, str(CONSENSUS_LAB))
from densired_style_generator import generate_densired_style_dataset, save_dataset as save_densired_dataset
from repliclust_style_generator import generate_archetype_dataset, save_dataset as save_repliclust_dataset
from simple_dataset_generator import generate_simple_gaussian_dataset, save_dataset as save_simple_dataset
BASE_CLUSTERINGS = 28

def _save_simple(name: str, dataset_type: str, **kwargs) -> None:
    path = DATASETS_DIR / f'{name}.npz'
    if path.exists():
        print(f'SKIP {path.name}')
        return
    x, gt, members, meta = generate_simple_gaussian_dataset(name=name, base_clusterings=BASE_CLUSTERINGS, base_k_min=2, base_k_max=max(8, int(kwargs['n_clusters']) + 3), base_strategy='mixed', **kwargs)
    meta['dataset_type'] = dataset_type
    save_simple_dataset(path, x, gt, members, meta)
    print(f'OK   {path.name:<34} type={dataset_type:<16} X={x.shape} members={members.shape} classes={np.unique(gt).size}')

def _save_repliclust(name: str, dataset_type: str, **kwargs) -> None:
    path = DATASETS_DIR / f'{name}.npz'
    if path.exists():
        print(f'SKIP {path.name}')
        return
    x, gt, members, meta = generate_archetype_dataset(name=name, base_clusterings=BASE_CLUSTERINGS, base_k_min=2, base_k_max=int(kwargs['n_clusters']) + 3, base_strategy='mixed', **kwargs)
    meta['dataset_type'] = dataset_type
    save_repliclust_dataset(path, x, gt, members, meta)
    print(f'OK   {path.name:<34} type={dataset_type:<16} X={x.shape} members={members.shape} classes={np.unique(gt).size}')

def _save_densired(name: str, dataset_type: str, **kwargs) -> None:
    path = DATASETS_DIR / f'{name}.npz'
    if path.exists():
        print(f'SKIP {path.name}')
        return
    x, gt, members, meta = generate_densired_style_dataset(name=name, base_clusterings=BASE_CLUSTERINGS, base_k_min=2, base_k_max=int(kwargs['clunum']) + 3, base_strategy='mixed', **kwargs)
    meta['dataset_type'] = dataset_type
    save_densired_dataset(path, x, gt, members, meta)
    print(f'OK   {path.name:<34} type={dataset_type:<16} X={x.shape} members={members.shape} classes={np.unique(gt).size}')

def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    _save_simple('design_mini_compact', 'compact', n_samples=420, n_clusters=5, dim=2, cluster_std=0.35, separation=5.0, imbalance_ratio=1.0, seed=4001)
    _save_simple('design_mini_overlap', 'overlapping', n_samples=450, n_clusters=5, dim=2, cluster_std=1.1, separation=2.1, imbalance_ratio=1.5, seed=4002)
    _save_simple('design_mini_imbalanced', 'imbalanced', n_samples=460, n_clusters=6, dim=5, cluster_std=0.75, separation=3.4, imbalance_ratio=5.0, seed=4003)
    _save_simple('design_mini_highdim', 'high_dimensional', n_samples=440, n_clusters=6, dim=20, cluster_std=0.9, separation=4.0, imbalance_ratio=1.8, seed=4004)
    _save_repliclust('design_mini_elongated', 'elongated', n_clusters=5, dim=2, n_samples=450, aspect_ref=5.5, aspect_maxmin=8.0, radius_ref=1.0, radius_maxmin=2.4, min_overlap=0.04, max_overlap=0.16, imbalance_ratio=2.0, distributions=['normal', 'student_t'], distribution_proportions=[0.6, 0.4], seed=4005)
    _save_densired('design_mini_density_varied', 'density_varied', dim=2, clunum=6, core_num=60, data_num=500, seed=4006, domain_size=18.0, radius=0.038, step=0.055, noise_ratio=0.1, density_factors=[0.45, 0.7, 1.0, 1.4, 1.8, 2.2], momentum=0.25, branch=0.05, star=0.1, distribution='uniform')
    _save_repliclust('design_mini_mixed_complex', 'mixed_complex', n_clusters=6, dim=5, n_samples=520, aspect_ref=4.0, aspect_maxmin=7.0, radius_ref=1.0, radius_maxmin=2.6, min_overlap=0.07, max_overlap=0.23, imbalance_ratio=3.0, distributions=['normal', 'student_t', 'exponential', 'lognormal'], distribution_proportions=[0.35, 0.25, 0.2, 0.2], seed=4007)
    _save_simple('design_compact_easy_5k', 'compact', n_samples=1200, n_clusters=5, dim=2, cluster_std=0.35, separation=5.5, imbalance_ratio=1.0, seed=4101)
    _save_simple('design_compact_easy_8k', 'compact', n_samples=1500, n_clusters=8, dim=4, cluster_std=0.4, separation=5.0, imbalance_ratio=1.2, seed=4102)
    _save_simple('design_overlap_moderate', 'overlapping', n_samples=1400, n_clusters=6, dim=2, cluster_std=1.15, separation=2.2, imbalance_ratio=1.4, seed=4201)
    _save_repliclust('design_overlap_oblong', 'overlapping', n_clusters=6, dim=2, n_samples=1500, aspect_ref=3.0, aspect_maxmin=5.0, radius_ref=1.0, radius_maxmin=2.0, min_overlap=0.1, max_overlap=0.26, imbalance_ratio=2.0, distributions=['normal', 'student_t', 'lognormal'], distribution_proportions=[0.5, 0.3, 0.2], seed=4202)
    _save_simple('design_imbalanced_6x', 'imbalanced', n_samples=1400, n_clusters=7, dim=5, cluster_std=0.7, separation=3.6, imbalance_ratio=6.0, seed=4301)
    _save_repliclust('design_imbalanced_oblong', 'imbalanced', n_clusters=8, dim=4, n_samples=1600, aspect_ref=2.5, aspect_maxmin=6.0, radius_ref=1.0, radius_maxmin=2.4, min_overlap=0.04, max_overlap=0.16, imbalance_ratio=6.0, distributions=['normal', 'student_t', 'exponential'], distribution_proportions=[0.45, 0.35, 0.2], seed=4302)
    _save_simple('design_highdim_20d', 'high_dimensional', n_samples=1400, n_clusters=7, dim=20, cluster_std=0.85, separation=4.0, imbalance_ratio=1.8, seed=4401)
    _save_simple('design_highdim_40d', 'high_dimensional', n_samples=1300, n_clusters=8, dim=40, cluster_std=0.95, separation=4.2, imbalance_ratio=2.0, seed=4402)
    _save_repliclust('design_elongated_2d', 'elongated', n_clusters=6, dim=2, n_samples=1500, aspect_ref=6.0, aspect_maxmin=9.0, radius_ref=1.0, radius_maxmin=2.5, min_overlap=0.04, max_overlap=0.16, imbalance_ratio=2.2, distributions=['normal', 'student_t'], distribution_proportions=[0.6, 0.4], seed=4501)
    _save_densired('design_elongated_density', 'elongated', dim=5, clunum=7, core_num=105, data_num=1700, seed=4502, domain_size=20.0, radius=0.032, step=0.067, noise_ratio=0.1, density_factors=[1.0, 0.8, 1.4, 0.6, 1.6, 0.9, 1.2], momentum=0.78, branch=0.1, star=0.35, distribution='uniform')
    _save_densired('design_density_varied_low_noise', 'density_varied', dim=2, clunum=7, core_num=98, data_num=1600, seed=4601, domain_size=20.0, radius=0.035, step=0.052, noise_ratio=0.05, density_factors=[0.45, 0.65, 0.85, 1.0, 1.3, 1.7, 2.1], momentum=0.25, branch=0.04, star=0.08, distribution='uniform')
    _save_densired('design_density_varied_noisy', 'density_varied', dim=3, clunum=8, core_num=120, data_num=1800, seed=4602, domain_size=20.0, radius=0.033, step=0.055, noise_ratio=0.16, density_factors=[0.4, 0.55, 0.75, 1.0, 1.25, 1.55, 1.9, 2.3], momentum=0.3, branch=0.06, star=0.12, distribution='gaussian')
    _save_repliclust('design_mixed_complex_6d', 'mixed_complex', n_clusters=8, dim=6, n_samples=1700, aspect_ref=4.0, aspect_maxmin=8.0, radius_ref=1.0, radius_maxmin=3.0, min_overlap=0.07, max_overlap=0.24, imbalance_ratio=3.5, distributions=['normal', 'student_t', 'exponential', 'lognormal'], distribution_proportions=[0.3, 0.3, 0.2, 0.2], seed=4701)
    _save_densired('design_mixed_complex_branchy', 'mixed_complex', dim=6, clunum=8, core_num=128, data_num=1800, seed=4702, domain_size=22.0, radius=0.031, step=0.066, noise_ratio=0.15, density_factors=[0.55, 0.7, 1.0, 1.3, 1.7, 0.85, 1.45, 2.0], momentum=0.7, branch=0.18, star=0.3, distribution='uniform')
if __name__ == '__main__':
    main()
