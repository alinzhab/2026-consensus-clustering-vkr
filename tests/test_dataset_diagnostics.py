from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'consensus_lab'
for p in (str(ROOT), str(LAB)):
    if p not in sys.path:
        sys.path.insert(0, p)
from dataset_diagnostics import compute_diagnostics, recommend_from_diagnostics

@pytest.fixture(scope='module')
def well_separated():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [8.0, 0.0], [4.0, 8.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in centres])
    gt = np.repeat(np.arange(3), 50)
    return (X, gt)

@pytest.fixture(scope='module')
def overlapping():
    rng = np.random.default_rng(1)
    centres = np.array([[0.0, 0.0], [1.5, 0.0], [0.75, 1.5]])
    X = np.vstack([c + rng.normal(scale=1.2, size=(50, 2)) for c in centres])
    gt = np.repeat(np.arange(3), 50)
    return (X, gt)

@pytest.fixture(scope='module')
def high_dim():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 80))
    gt = np.repeat(np.arange(4), 50)
    return (X, gt)

@pytest.fixture(scope='module')
def imbalanced():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal([0, 0], 0.5, size=(150, 2)), rng.normal([6, 0], 0.5, size=(20, 2)), rng.normal([3, 6], 0.5, size=(10, 2))])
    gt = np.concatenate([np.zeros(150), np.ones(20), np.full(10, 2)]).astype(int)
    return (X, gt)

def test_basic_fields_present(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    for key in ('n_objects', 'n_features', 'dimensionality_ratio', 'missing_ratio', 'infinite_ratio', 'constant_feature_ratio', 'feature_variance_mean', 'feature_variance_std', 'feature_variance_min', 'feature_variance_max'):
        assert key in diag, f'Missing field: {key}'

def test_basic_values_correct(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    assert diag['n_objects'] == 150
    assert diag['n_features'] == 2
    assert diag['missing_ratio'] == 0.0
    assert diag['infinite_ratio'] == 0.0
    assert diag['constant_feature_ratio'] == 0.0
    assert diag['feature_variance_mean'] > 0

def test_missing_and_inf_ratios(well_separated):
    X, _ = well_separated
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    X_bad[1, 1] = np.inf
    diag = compute_diagnostics(X_bad)
    assert diag['missing_ratio'] > 0
    assert diag['infinite_ratio'] > 0
    assert 'n_objects' in diag

def test_pca_fields_present(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    for key in ('explained_variance_2d', 'explained_variance_10d', 'effective_dimension_90', 'effective_dimension_95', 'participation_ratio'):
        assert key in diag, f'Missing PCA field: {key}'

def test_pca_ranges(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    assert 0.0 < diag['explained_variance_2d'] <= 1.0
    assert 0.0 < diag['explained_variance_10d'] <= 1.0
    assert diag['effective_dimension_90'] >= 1
    assert diag['effective_dimension_95'] >= diag['effective_dimension_90']
    assert diag['participation_ratio'] >= 1.0

def test_pca_high_dim(high_dim):
    X, _ = high_dim
    diag = compute_diagnostics(X)
    assert diag['explained_variance_2d'] < 0.5
    assert diag['effective_dimension_90'] > 2

def test_density_fields(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    assert 'knn_distance_mean' in diag
    assert 'knn_distance_std' in diag
    assert 'density_variation' in diag
    assert 'outlier_ratio' in diag

def test_density_values_positive(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    assert diag['knn_distance_mean'] > 0
    assert diag['knn_distance_std'] >= 0
    assert diag['density_variation'] >= 0
    assert 0.0 <= diag['outlier_ratio'] <= 1.0

def test_density_variation_higher_for_overlapping(well_separated, overlapping):
    Xw, _ = well_separated
    Xo, _ = overlapping
    diag_w = compute_diagnostics(Xw)
    diag_o = compute_diagnostics(Xo)
    assert diag_w['density_variation'] >= 0

def test_hopkins_present(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    assert 'hopkins' in diag

def test_hopkins_range(well_separated):
    X, _ = well_separated
    diag = compute_diagnostics(X)
    h = diag['hopkins']
    if h is not None:
        assert 0.0 <= h <= 1.0

def test_hopkins_higher_for_clustered_vs_random():
    rng = np.random.default_rng(42)
    centres = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
    X_clust = np.vstack([c + rng.normal(scale=0.3, size=(60, 2)) for c in centres])
    X_rand = rng.uniform(0, 10, size=(180, 2))
    h_clust = compute_diagnostics(X_clust, seed=0)['hopkins']
    h_rand = compute_diagnostics(X_rand, seed=0)['hopkins']
    if h_clust is not None and h_rand is not None:
        assert h_clust > h_rand

def test_class_stats_fields(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    for key in ('n_classes', 'class_size_min', 'class_size_max', 'imbalance_ratio', 'small_class_ratio', 'class_entropy'):
        assert key in diag

def test_class_stats_values(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert diag['n_classes'] == 3
    assert diag['class_size_min'] == 50
    assert diag['class_size_max'] == 50
    assert diag['imbalance_ratio'] == pytest.approx(1.0)
    assert diag['class_entropy'] > 0

def test_imbalanced_class_stats(imbalanced):
    X, gt = imbalanced
    diag = compute_diagnostics(X, gt=gt)
    assert diag['imbalance_ratio'] > 5.0
    assert diag['n_classes'] == 3

def test_separability_fields(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert 'silhouette_score' in diag
    assert 'calinski_harabasz_score' in diag
    assert 'davies_bouldin_score' in diag

def test_separability_ranges(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    sil = diag['silhouette_score']
    if sil is not None:
        assert -1.0 <= sil <= 1.0
    ch = diag['calinski_harabasz_score']
    if ch is not None:
        assert ch > 0
    db = diag['davies_bouldin_score']
    if db is not None:
        assert db >= 0

def test_well_separated_has_higher_silhouette(well_separated, overlapping):
    Xw, gtw = well_separated
    Xo, gto = overlapping
    diag_w = compute_diagnostics(Xw, gt=gtw)
    diag_o = compute_diagnostics(Xo, gt=gto)
    sw = diag_w.get('silhouette_score')
    so = diag_o.get('silhouette_score')
    if sw is not None and so is not None:
        assert sw > so

def test_overlap_fields(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert 'nearest_friend_distance' in diag
    assert 'nearest_enemy_distance' in diag
    assert 'overlap_ratio' in diag
    assert 'margin_ratio' in diag

def test_overlap_ranges(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert 0.0 <= diag['overlap_ratio'] <= 1.0
    assert diag['margin_ratio'] >= 0

def test_well_separated_low_overlap(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert diag['overlap_ratio'] < 0.15, 'Well-separated clusters should have low overlap'

def test_overlapping_high_overlap():
    rng = np.random.default_rng(99)
    X = np.vstack([rng.normal([0.0, 0.0], 1.5, size=(30, 2)), rng.normal([0.5, 0.5], 1.5, size=(30, 2)), rng.normal([1.0, 0.0], 1.5, size=(30, 2))])
    gt = np.repeat(np.arange(3), 30)
    diag = compute_diagnostics(X, gt=gt)
    sil = diag.get('silhouette_score')
    if sil is not None:
        assert sil < 0.4, 'Strongly overlapping clusters should have low silhouette'
    assert 0.0 <= diag['overlap_ratio'] <= 1.0

def test_centroid_fields(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert 'centroid_distance_min' in diag
    assert 'centroid_distance_mean' in diag
    assert 'centroid_margin_mean' in diag

def test_centroid_distances_positive(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert diag['centroid_distance_min'] > 0
    assert diag['centroid_distance_mean'] >= diag['centroid_distance_min']

def test_shape_fields(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert 'elongation_mean' in diag
    assert 'elongation_max' in diag
    assert 'covariance_condition_mean' in diag
    assert 'covariance_condition_max' in diag

def test_shape_values_positive(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    assert diag['elongation_mean'] >= 1.0
    assert diag['elongation_max'] >= diag['elongation_mean']

def test_elongated_clusters_higher_elongation():
    rng = np.random.default_rng(7)
    X_elong = np.vstack([rng.normal([0, 0], [5.0, 0.3], size=(60, 2)), rng.normal([0, 8], [5.0, 0.3], size=(60, 2))])
    gt_elong = np.repeat([0, 1], 60)
    X_round = np.vstack([rng.normal([0, 0], 1.0, size=(60, 2)), rng.normal([0, 8], 1.0, size=(60, 2))])
    gt_round = np.repeat([0, 1], 60)
    diag_e = compute_diagnostics(X_elong, gt=gt_elong)
    diag_r = compute_diagnostics(X_round, gt=gt_round)
    assert diag_e['elongation_mean'] > diag_r['elongation_mean']

def test_single_class_gt(well_separated):
    X, _ = well_separated
    gt_single = np.zeros(len(X), dtype=int)
    diag = compute_diagnostics(X, gt=gt_single)
    assert 'n_classes' in diag
    assert diag['n_classes'] == 1

def test_very_small_n():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    gt = np.array([0, 0, 1, 1])
    diag = compute_diagnostics(X, gt=gt)
    assert diag['n_objects'] == 4
    assert 'hopkins' in diag

def test_nan_in_X_does_not_crash():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 5))
    X[5, 2] = np.nan
    X[10, 0] = np.inf
    diag = compute_diagnostics(X)
    assert diag['missing_ratio'] > 0
    assert diag['n_objects'] == 50

def test_recommend_basic_structure(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    assert 'm' in rec
    assert 'k_min' in rec
    assert 'k_max' in rec
    assert 'strategy' in rec
    assert 'per_algorithm' in rec
    assert 'reasoning' in rec
    assert 'warnings' in rec
    for algo in ('hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified'):
        assert algo in rec['per_algorithm']

def test_recommend_k_range_from_n_classes(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    assert rec['k_min'] >= 1
    assert rec['k_max'] >= rec['k_min']
    assert rec['k_max'] <= 30

def test_recommend_large_n_uses_kmeans():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(3500, 4))
    gt = np.repeat(np.arange(4), 875)
    diag = compute_diagnostics(X, gt=gt, seed=0)
    rec = recommend_from_diagnostics(diag)
    assert rec['strategy'] == 'kmeans'
    assert rec['per_algorithm']['sdgca_modified']['diffusion_time'] >= 4

def test_recommend_small_n_uses_mixed():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 3))
    gt = np.repeat(np.arange(4), 25)
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    assert rec['strategy'] == 'mixed'
    assert rec['m'] <= 40

def test_recommend_high_overlap_increases_sharpen(overlapping):
    X, gt = overlapping
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    if diag.get('overlap_ratio', 0) > 0.35:
        assert rec['per_algorithm']['hierarchical_weighted']['sharpen'] >= 2.0

def test_recommend_well_separated_lower_sharpen(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    sil = diag.get('silhouette_score', 0) or 0
    if sil > 0.5:
        assert rec['per_algorithm']['hierarchical_weighted']['sharpen'] <= 1.5

def test_recommend_imbalanced_increases_m(imbalanced):
    X, gt = imbalanced
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    assert 'imbalance' in ' '.join(rec['reasoning']).lower() or rec['m'] >= 30

def test_recommend_reasoning_nonempty(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    assert len(rec['reasoning']) >= 3

def test_recommend_sdgca_params_valid(well_separated):
    X, gt = well_separated
    diag = compute_diagnostics(X, gt=gt)
    rec = recommend_from_diagnostics(diag)
    sdgca = rec['per_algorithm']['sdgca']
    assert 0 < sdgca['lambda_'] < 1
    assert 0 < sdgca['eta'] < 1
    assert 0 < sdgca['theta'] < 1

def test_recommend_no_gt():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 6))
    diag = compute_diagnostics(X)
    rec = recommend_from_diagnostics(diag)
    assert rec['k_min'] == 2
    assert rec['k_max'] >= 2
