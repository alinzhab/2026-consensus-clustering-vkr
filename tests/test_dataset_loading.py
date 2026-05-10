"""Тесты загрузки датасетов и вспомогательных функций.

Проверяем:
  - _load_table_arrays: CSV с заголовком, без заголовка, TSV, строковые метки
  - _load_json_arrays: полный и минимальный JSON-датасет
  - summarize_dataset: NPZ, CSV; поля shape/n_classes/keys
  - analyze_dataset_structure: вычисленные флаги и k_estimate
  - recommend_params: правила m, k_min/k_max, strategy, diffusion_time
  - build_consensus_ready_dataset: датасет X+gt → members строится автоматически
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _load_json_arrays,
    _load_table_arrays,
    analyze_dataset_structure,
    build_consensus_ready_dataset,
    recommend_params,
    summarize_dataset,
)


# ---------------------------------------------------------------------------
# Вспомогательные генераторы временных файлов
# ---------------------------------------------------------------------------


def _csv_with_header(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text(
        "f1,f2,f3,gt\n"
        "1.0,2.0,3.0,0\n"
        "4.0,5.0,6.0,1\n"
        "7.0,8.0,9.0,0\n"
        "10.0,11.0,12.0,1\n",
        encoding="utf-8",
    )
    return p


def _csv_no_header(tmp_path: Path) -> Path:
    p = tmp_path / "nohdr.csv"
    p.write_text(
        "1.0,2.0,0\n"
        "3.0,4.0,1\n"
        "5.0,6.0,0\n",
        encoding="utf-8",
    )
    return p


def _tsv_string_labels(tmp_path: Path) -> Path:
    p = tmp_path / "labels.tsv"
    p.write_text(
        "a\tb\tclass\n"
        "1.0\t2.0\tcat\n"
        "3.0\t4.0\tdog\n"
        "5.0\t6.0\tcat\n",
        encoding="utf-8",
    )
    return p


def _json_full(tmp_path: Path, n: int = 30) -> Path:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3)).tolist()
    gt = (rng.integers(0, 3, size=n)).tolist()
    members = rng.integers(1, 4, size=(n, 10)).tolist()
    p = tmp_path / "full.json"
    p.write_text(
        json.dumps({"X": x, "gt": gt, "members": members, "meta": {"note": "test"}}),
        encoding="utf-8",
    )
    return p


def _json_minimal(tmp_path: Path, n: int = 20) -> Path:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 2)).tolist()
    gt = rng.integers(0, 2, size=n).tolist()
    p = tmp_path / "minimal.json"
    p.write_text(json.dumps({"X": x, "gt": gt}), encoding="utf-8")
    return p


def _npz_x_gt_only(tmp_path: Path) -> Path:
    """NPZ без members — для теста build_consensus_ready_dataset."""
    rng = np.random.default_rng(2)
    centres = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]])
    x = np.vstack([c + rng.normal(scale=0.4, size=(20, 2)) for c in centres])
    gt = np.repeat(np.arange(3), 20)
    p = tmp_path / "x_only.npz"
    np.savez(p, X=x, gt=gt)
    return p


# ---------------------------------------------------------------------------
# _load_table_arrays
# ---------------------------------------------------------------------------


def test_csv_with_header_shape(tmp_path):
    x, gt, members, meta = _load_table_arrays(_csv_with_header(tmp_path))
    assert x.shape == (4, 3)
    assert gt.shape == (4,)
    assert members is None
    assert meta["label_column"] == "gt"


def test_csv_no_header_last_col_as_label(tmp_path):
    x, gt, members, meta = _load_table_arrays(_csv_no_header(tmp_path))
    assert x.shape == (3, 2)
    assert set(gt.tolist()) == {0, 1}
    assert meta["label_column"] == "last_column"


def test_tsv_string_labels_encoded(tmp_path):
    x, gt, members, meta = _load_table_arrays(_tsv_string_labels(tmp_path))
    assert x.shape == (3, 2)
    assert gt.dtype == np.int64
    assert set(gt.tolist()) == {0, 1}
    assert meta["label_mapping"] is not None


def test_csv_minimum_two_columns_required(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("1\n2\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one feature"):
        _load_table_arrays(p)


# ---------------------------------------------------------------------------
# _load_json_arrays
# ---------------------------------------------------------------------------


def test_json_full_loads_all_fields(tmp_path):
    x, gt, members, meta = _load_json_arrays(_json_full(tmp_path))
    assert x.shape == (30, 3)
    assert gt.shape == (30,)
    assert members.shape == (30, 10)
    assert meta.get("note") == "test"
    assert meta["source_format"] == ".json"


def test_json_minimal_no_members(tmp_path):
    x, gt, members, meta = _load_json_arrays(_json_minimal(tmp_path))
    assert x.shape == (20, 2)
    assert gt.shape == (20,)
    assert members is None


def test_json_bad_structure_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        _load_json_arrays(p)


# ---------------------------------------------------------------------------
# summarize_dataset
# ---------------------------------------------------------------------------


def test_summarize_npz(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 4))
    gt = np.repeat(np.arange(5), 10)
    members = rng.integers(1, 6, size=(50, 15)).astype(np.int64)
    p = tmp_path / "ds.npz"
    np.savez(p, X=x, gt=gt, members=members)
    summary = summarize_dataset(p)
    assert summary["x_shape"] == (50, 4)
    assert summary["gt_shape"] == (50,)
    assert summary["members_shape"] == (50, 15)
    assert summary["n_classes"] == 5
    assert "error" not in summary


def test_summarize_csv(tmp_path):
    summary = summarize_dataset(_csv_with_header(tmp_path))
    assert summary["x_shape"] == (4, 3)
    assert summary["n_classes"] == 2
    assert summary["suffix"] == ".csv"


# ---------------------------------------------------------------------------
# analyze_dataset_structure
# ---------------------------------------------------------------------------


def test_analyze_sets_n_samples_and_features(tmp_path):
    rng = np.random.default_rng(4)
    p = tmp_path / "a.npz"
    np.savez(p, X=rng.normal(size=(100, 10)), gt=np.repeat(np.arange(4), 25))
    info = analyze_dataset_structure(p)
    # compute_diagnostics использует n_objects вместо n_samples
    assert info["n_objects"] == 100
    assert info["n_features"] == 10
    assert info["n_classes"] == 4
    # n_objects <= 2000 → не «большой»
    assert info["n_objects"] <= 2000
    # n_features <= 50 → не высокоразмерный
    assert info["n_features"] <= 50


def test_analyze_large_high_dim(tmp_path):
    rng = np.random.default_rng(5)
    p = tmp_path / "b.npz"
    np.savez(p, X=rng.normal(size=(3000, 60)), gt=np.repeat(np.arange(12), 250))
    info = analyze_dataset_structure(p)
    assert info["n_objects"] > 2000       # is_large
    assert info["n_features"] > 50        # is_high_dim
    assert info["n_classes"] > 10         # is_many_classes


def test_analyze_has_members_flag(tmp_path):
    rng = np.random.default_rng(6)
    p = tmp_path / "c.npz"
    np.savez(
        p,
        X=rng.normal(size=(40, 2)),
        gt=np.repeat(np.arange(2), 20),
        members=rng.integers(1, 3, size=(40, 10)).astype(np.int64),
    )
    info = analyze_dataset_structure(p)
    assert info["has_members"]
    assert info["members_m"] == 10
    assert info["is_runnable"]


# ---------------------------------------------------------------------------
# recommend_params
# ---------------------------------------------------------------------------


def test_recommend_small_known_classes():
    rec = recommend_params(n_samples=150, n_features=4, n_classes=3)
    assert rec["k_min"] == 1 or rec["k_min"] >= 2
    assert rec["k_max"] >= rec["k_min"]
    assert rec["k_max"] <= 20
    assert rec["m"] == 20
    assert rec["strategy"] == "mixed"
    assert rec["per_algorithm"]["sdgca_modified"]["diffusion_time"] == 3


def test_recommend_large_dataset():
    rec = recommend_params(n_samples=5000, n_features=20, n_classes=8)
    assert rec["strategy"] == "kmeans"
    assert rec["m"] >= 40
    assert rec["per_algorithm"]["sdgca_modified"]["diffusion_time"] == 5


def test_recommend_many_classes_sharpen():
    rec = recommend_params(n_samples=500, n_features=10, n_classes=12)
    assert rec["per_algorithm"]["hierarchical_weighted"]["sharpen"] == 2.0


def test_recommend_few_classes_sharpen():
    rec = recommend_params(n_samples=200, n_features=5, n_classes=3)
    assert rec["per_algorithm"]["hierarchical_weighted"]["sharpen"] == 1.5


def test_recommend_uses_existing_members_m():
    rec = recommend_params(n_samples=1000, n_features=5, n_classes=4, has_members=True, members_m=35)
    assert rec["m"] == 35


def test_recommend_high_dim_warning():
    rec = recommend_params(n_samples=300, n_features=100, n_classes=5)
    assert any("размерность" in w.lower() or "d >" in w for w in rec["warnings"])


def test_recommend_unknown_classes():
    rec = recommend_params(n_samples=400, n_features=5, n_classes=None)
    assert rec["k_min"] == 2
    assert rec["k_max"] == 10
    # Предупреждение об отсутствующем gt/n_classes
    assert any(
        "неизвестно" in w.lower() or "отсутствует" in w.lower() or "gt" in w.lower()
        for w in rec["warnings"]
    )


# ---------------------------------------------------------------------------
# build_consensus_ready_dataset
# ---------------------------------------------------------------------------


def test_build_from_x_gt_creates_members(tmp_path):
    p = _npz_x_gt_only(tmp_path)
    ready_path, members = build_consensus_ready_dataset(
        p, n_clusterings=10, k_min=2, k_max=4, strategy="kmeans"
    )
    assert ready_path.exists()
    assert ready_path.name.endswith("_consensus_ready.npz")
    assert members.shape == (60, 10)
    data = np.load(ready_path, allow_pickle=True)
    assert "X" in data.files
    assert "gt" in data.files
    assert "members" in data.files


def test_build_raises_without_gt(tmp_path):
    rng = np.random.default_rng(7)
    p = tmp_path / "no_gt.npz"
    np.savez(p, X=rng.normal(size=(30, 2)))
    with pytest.raises(ValueError, match="gt"):
        build_consensus_ready_dataset(p, 10, 2, 4, "kmeans")


def test_build_raises_without_x_and_members(tmp_path):
    p = tmp_path / "gt_only.npz"
    np.savez(p, gt=np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="X"):
        build_consensus_ready_dataset(p, 10, 2, 4, "kmeans")
