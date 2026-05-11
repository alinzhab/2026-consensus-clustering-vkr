from __future__ import annotations
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from algorithms_base import AlgorithmRegistry, ConsensusResult

def _make_simple_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 6.0]])
    pts_per_cluster = 40
    x = np.vstack([c + rng.normal(scale=0.4, size=(pts_per_cluster, 2)) for c in centres])
    gt = np.repeat(np.arange(1, 4), pts_per_cluster)
    from base_clusterings import build_base_clusterings
    members = build_base_clusterings(x, n_clusterings=25, k_min=2, k_max=5, rng=0, strategy='kmeans')
    out = tmp_path / 'smoke.npz'
    np.savez(out, X=x, gt=gt, members=members.astype(np.int64))
    return out

@pytest.fixture(scope='module')
def smoke_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp('smoke_data')
    return _make_simple_dataset(tmp)

@pytest.mark.parametrize('algo_name', ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified'])
def test_algorithm_runs_and_returns_valid_metrics(smoke_dataset: Path, algo_name: str) -> None:
    algo = AlgorithmRegistry.get(algo_name)
    result = algo.run(dataset_path=smoke_dataset, m=10, runs=2, method='average', seed=19)
    assert isinstance(result, ConsensusResult)
    for attr in ('nmi_mean', 'ari_mean', 'f_mean'):
        value = getattr(result, attr)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.001
    for attr in ('nmi_std', 'ari_std', 'f_std'):
        value = getattr(result, attr)
        assert isinstance(value, float)
        assert value >= 0.0

@pytest.mark.parametrize('algo_name', ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified'])
def test_algorithm_recovers_well_separated_clusters(smoke_dataset: Path, algo_name: str) -> None:
    algo = AlgorithmRegistry.get(algo_name)
    result = algo.run(dataset_path=smoke_dataset, m=15, runs=3, method='average', seed=19)
    assert result.nmi_mean > 0.7, f'{algo_name}: NMI={result.nmi_mean:.3f} ниже регресс-порога 0.7'

def test_registry_lists_all_four_algorithms() -> None:
    names = AlgorithmRegistry.names()
    assert set(names) == {'hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified'}

def test_registry_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        AlgorithmRegistry.get('does_not_exist')

def test_consensus_result_serialisable(smoke_dataset: Path) -> None:
    import json
    algo = AlgorithmRegistry.get('hierarchical_baseline')
    result = algo.run(dataset_path=smoke_dataset, m=10, runs=2)
    payload = result.as_dict()
    base_keys = {'data_name', 'algorithm', 'method', 'm', 'runs', 'seed', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean', 'f_std'}
    json.dumps({k: payload[k] for k in base_keys})
