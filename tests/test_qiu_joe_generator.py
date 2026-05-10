import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consensus_lab"))

from qiu_joe_style_generator import generate_qiu_joe_style_dataset, save_dataset


def test_qiu_joe_generator_returns_consensus_ready_arrays():
    x, gt, members, meta = generate_qiu_joe_style_dataset(
        name="qiu_smoke",
        n_samples=60,
        n_clusters=3,
        dim=2,
        overlap_level="medium",
        separation=1.0,
        shape_ratio=2.0,
        volume_mean=1.0,
        imbalance_ratio=1.5,
        orientation="axis_aligned",
        noise_ratio=0.05,
        seed=123,
        base_clusterings=4,
        base_k_min=2,
        base_k_max=4,
        base_strategy="kmeans",
    )

    assert x.shape == (60, 2)
    assert gt.shape == (60,)
    assert members.shape == (60, 4)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(members))
    assert np.unique(gt).size == 4
    assert meta["generator"] == "qiu_joe_cluster_generation_style"
    assert meta["n_samples_actual"] == 60
    assert meta["n_outliers"] == 3


def test_qiu_joe_save_dataset_writes_project_format(tmp_path):
    x, gt, members, meta = generate_qiu_joe_style_dataset(
        name="qiu_save_smoke",
        n_samples=40,
        n_clusters=2,
        dim=2,
        overlap_level="low",
        shape_ratio=1.5,
        imbalance_ratio=1.0,
        orientation="axis_aligned",
        seed=7,
        base_clusterings=3,
        base_k_min=2,
        base_k_max=3,
        base_strategy="kmeans",
    )
    output_path = tmp_path / "qiu_save_smoke.npz"

    save_dataset(output_path, x, gt, members, meta)

    data = np.load(output_path, allow_pickle=True)
    assert {"X", "gt", "members", "meta"}.issubset(set(data.files))
    assert data["X"].shape == x.shape
    assert data["gt"].shape == gt.shape
    assert data["members"].shape == members.shape
    assert json.loads(str(data["meta"]))["name"] == "qiu_save_smoke"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overlap_level": "bad"},
        {"orientation": "bad"},
        {"n_clusters": 1},
        {"n_samples": 5, "n_clusters": 3},
        {"shape_ratio": 0.5},
        {"volume_mean": 0.0},
        {"noise_ratio": 0.8},
    ],
)
def test_qiu_joe_generator_rejects_invalid_parameters(kwargs):
    params = {
        "n_samples": 40,
        "n_clusters": 2,
        "dim": 2,
        "base_clusterings": 2,
        "base_strategy": "kmeans",
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        generate_qiu_joe_style_dataset(**params)
