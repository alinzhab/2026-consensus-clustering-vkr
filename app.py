import csv
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from scipy.io import loadmat
from werkzeug.utils import secure_filename

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = DATASETS_DIR / "uploaded"
CONSENSUS_LAB_DIR = BASE_DIR / "consensus_lab"
SQLITE_DB_PATH = BASE_DIR / "app.db"
sys.path.insert(0, str(CONSENSUS_LAB_DIR))

from densired_style_generator import generate_densired_style_dataset, save_dataset as save_densired_dataset
from qiu_joe_style_generator import generate_qiu_joe_style_dataset, save_dataset as save_qiu_joe_dataset
from repliclust_style_generator import generate_archetype_dataset, save_dataset as save_repliclust_dataset
from base_clusterings import build_base_clusterings
from simple_dataset_generator import generate_simple_gaussian_dataset, save_dataset as save_simple_dataset
from sdgca_modified import resolve_params


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

ALGORITHM_LABELS = {
    "hierarchical_baseline": "Иерархическая базовая версия",
    "hierarchical_weighted": "Иерархическая взвешенная версия",
    "sdgca": "SDGCA",
    "sdgca_modified": "SDGCA, модифицированная версия",
}

METHOD_LABELS = {
    "average": "average",
    "complete": "complete",
    "single": "single",
    "ward": "ward",
}


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_database_url(url):
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


DATABASE_URL = normalize_database_url(os.environ.get("DATABASE_URL", "")).strip()
USE_POSTGRES = DATABASE_URL.startswith("postgresql://")
DEMO_MODE = env_flag("DEMO_MODE", False)


def get_db_connection():
    if USE_POSTGRES:
        if psycopg is None:
            raise RuntimeError("psycopg is not installed, but DATABASE_URL points to PostgreSQL")
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _db_execute(sql, params=None):
    """Выполнить SQL-запрос, адаптируя плейсхолдеры под движок БД."""
    if USE_POSTGRES:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
    else:
        sql_sqlite = sql.replace("%s", "?").replace("EXCLUDED.", "excluded.")
        with get_db_connection() as conn:
            conn.execute(sql_sqlite, params or ())


def _db_query(sql):
    """Выполнить SELECT-запрос, вернуть список dict-строк."""
    with get_db_connection() as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
        else:
            return conn.execute(sql).fetchall()


_CREATE_DATASETS_TABLE = """
CREATE TABLE IF NOT EXISTS datasets (
    id {pk} PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL UNIQUE,
    suffix TEXT,
    size_kb {real_type},
    source TEXT,
    keys_json TEXT,
    x_shape_json TEXT,
    gt_shape_json TEXT,
    members_shape_json TEXT,
    n_classes INTEGER,
    meta_json TEXT,
    error TEXT,
    updated_at TEXT NOT NULL
)
"""

_CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    id {pk} PRIMARY KEY,
    created_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    algorithm_label TEXT,
    method TEXT,
    method_label TEXT,
    seed INTEGER,
    m INTEGER,
    runs INTEGER,
    nmi_mean {real_type},
    nmi_std {real_type},
    ari_mean {real_type},
    ari_std {real_type},
    f_mean {real_type},
    f_std {real_type},
    params_json TEXT,
    result_file TEXT UNIQUE
)
"""


def init_db():
    pk = "SERIAL" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real_type = "DOUBLE PRECISION" if USE_POSTGRES else "REAL"
    fmt = {"pk": pk, "real_type": real_type}
    if USE_POSTGRES:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_DATASETS_TABLE.format(**fmt))
                cur.execute(_CREATE_RESULTS_TABLE.format(**fmt))
            conn.commit()
    else:
        with get_db_connection() as conn:
            conn.executescript(
                _CREATE_DATASETS_TABLE.format(**fmt) + ";" + _CREATE_RESULTS_TABLE.format(**fmt)
            )


def ensure_dirs():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


SUPPORTED_DATASET_SUFFIXES = {".npz", ".mat", ".csv", ".tsv", ".txt", ".json"}
RUNNABLE_DATASET_SUFFIXES = {".npz", ".mat"}
TABLE_DATASET_SUFFIXES = {".csv", ".tsv", ".txt"}


def allowed_dataset(filename):
    return Path(filename).suffix.lower() in SUPPORTED_DATASET_SUFFIXES


def _json_safe(value):
    """Convert numpy/scalar containers to values accepted by json.dumps."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _jsonify_or_none(value):
    return None if value is None else json.dumps(_json_safe(value), ensure_ascii=False)


def _parse_json_field(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


TABLE_LABEL_COLUMNS = {"gt", "label", "labels", "target", "class", "y"}


def _is_float_token(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _split_table_line(line, delimiter):
    if delimiter is None:
        return line.strip().split()
    return [item.strip() for item in line.strip().split(delimiter)]


def _table_delimiter(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix == ".tsv":
        return "\t"
    return None


def _encode_labels(values):
    labels = []
    for value in values:
        text = str(value).strip()
        try:
            numeric = float(text)
            labels.append(int(numeric) if numeric.is_integer() else numeric)
        except ValueError:
            labels.append(text)

    if all(isinstance(label, int) for label in labels):
        return np.asarray(labels, dtype=np.int64), None

    mapping = {label: idx for idx, label in enumerate(sorted(set(labels), key=str))}
    encoded = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    return encoded, {str(key): int(value) for key, value in mapping.items()}


def _load_table_arrays(path: Path):
    delimiter = _table_delimiter(path)
    text = path.read_text(encoding="utf-8-sig")
    if delimiter is None:
        rows_raw = [_split_table_line(line, delimiter) for line in text.splitlines() if line.strip()]
    else:
        rows_raw = [
            [item.strip() for item in row]
            for row in csv.reader(text.splitlines(), delimiter=delimiter)
            if row and any(item.strip() for item in row)
        ]
    if not rows_raw:
        raise ValueError("Table file is empty")

    first = rows_raw[0]
    has_header = not (all(_is_float_token(item) for item in first) or (len(first) >= 2 and all(_is_float_token(item) for item in first[:-1])))
    header = first if has_header else None
    data_rows = rows_raw[1:] if has_header else rows_raw
    rows = []
    for parts in data_rows:
        if len(parts) != len(data_rows[0]):
            raise ValueError("All table rows must contain the same number of columns")
        rows.append(parts)
    if not rows:
        raise ValueError("Table file does not contain numeric rows")

    if len(rows[0]) < 2:
        raise ValueError("Table dataset must contain at least one feature column and one label column")

    if header:
        lowered = [name.strip().lower() for name in header]
        label_idx = next((idx for idx, name in enumerate(lowered) if name in TABLE_LABEL_COLUMNS), len(rows[0]) - 1)
    else:
        label_idx = len(rows[0]) - 1

    feature_rows = []
    label_values = []
    for row in rows:
        label_values.append(row[label_idx])
        feature_rows.append([float(value) for idx, value in enumerate(row) if idx != label_idx])

    x = np.asarray(feature_rows, dtype=np.float64)
    gt, label_mapping = _encode_labels(label_values)
    meta = {
        "source_format": path.suffix.lower(),
        "label_column": header[label_idx] if header else "last_column",
        "feature_columns": [name for idx, name in enumerate(header) if idx != label_idx] if header else None,
        "label_mapping": label_mapping,
    }
    return x, gt, None, meta


def _load_json_arrays(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON dataset must be an object with X/gt/members fields")
    x = np.asarray(payload["X"], dtype=np.float64) if "X" in payload else None
    gt = np.asarray(payload["gt"]).reshape(-1).astype(np.int64) if "gt" in payload else None
    members = np.asarray(payload["members"], dtype=np.int64) if "members" in payload else None
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    meta = {**meta, "source_format": ".json"}
    return x, gt, members, meta


def summarize_dataset(path: Path):
    summary = {
        "name": path.name,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }
    try:
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            summary["keys"] = list(data.files)
            if "X" in data:
                summary["x_shape"] = tuple(data["X"].shape)
            if "gt" in data:
                summary["gt_shape"] = tuple(data["gt"].shape)
                summary["n_classes"] = int(np.unique(data["gt"]).size)
            if "members" in data:
                summary["members_shape"] = tuple(data["members"].shape)
            if "meta" in data:
                meta_raw = data["meta"]
                if np.isscalar(meta_raw):
                    meta_raw = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode("utf-8", errors="ignore")
                if isinstance(meta_raw, str):
                    summary["meta"] = json.loads(meta_raw)
        elif path.suffix.lower() == ".mat":
            data = loadmat(path)
            keys = [key for key in data.keys() if not key.startswith("__")]
            summary["keys"] = keys
            for x_key in ("X", "x", "data", "fea", "features"):
                if x_key in data:
                    summary["x_shape"] = tuple(np.asarray(data[x_key]).shape)
                    break
            if "gt" in data:
                gt = np.asarray(data["gt"]).reshape(-1)
                summary["gt_shape"] = tuple(gt.shape)
                summary["n_classes"] = int(np.unique(gt).size)
            if "members" in data:
                summary["members_shape"] = tuple(np.asarray(data["members"]).shape)
        elif path.suffix.lower() in TABLE_DATASET_SUFFIXES:
            x, gt, _, meta = _load_table_arrays(path)
            summary["keys"] = ["X", "gt"]
            summary["x_shape"] = tuple(x.shape)
            summary["gt_shape"] = tuple(gt.shape)
            summary["n_classes"] = int(np.unique(gt).size)
            summary["meta"] = meta
        elif path.suffix.lower() == ".json":
            x, gt, members, meta = _load_json_arrays(path)
            keys = []
            if x is not None:
                keys.append("X")
                summary["x_shape"] = tuple(x.shape)
            if gt is not None:
                keys.append("gt")
                summary["gt_shape"] = tuple(gt.shape)
                summary["n_classes"] = int(np.unique(gt).size)
            if members is not None:
                keys.append("members")
                summary["members_shape"] = tuple(members.shape)
            summary["keys"] = keys
            summary["meta"] = meta
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def analyze_dataset_structure(path: Path) -> dict:
    """Полная диагностика датасета: структурные метрики + флаги совместимости.

    Загружает X и gt из файла, вызывает compute_diagnostics и добавляет
    поля совместимости с веб-интерфейсом (is_runnable, has_members и т.д.).
    """
    from dataset_diagnostics import compute_diagnostics

    summary = summarize_dataset(path)
    x_shape = summary.get("x_shape")
    gt_shape = summary.get("gt_shape")
    members_shape = summary.get("members_shape")

    has_members = members_shape is not None
    members_m = members_shape[1] if (has_members and len(members_shape) > 1) else None

    # Совместимость-флаги (быстро, без загрузки массивов)
    compat = {
        "has_members": has_members,
        "members_m": members_m,
        "is_runnable": has_members and summary.get("suffix") in RUNNABLE_DATASET_SUFFIXES,
    }

    # Полная диагностика (требует загрузки X/gt)
    diagnostics: dict = {}
    if not summary.get("error") and x_shape is not None:
        try:
            x, gt, _, _ = _load_dataset_arrays(path)
            if x is not None:
                diagnostics = compute_diagnostics(x, gt=gt, seed=0)
        except Exception as exc:
            diagnostics = {"diagnostics_error": str(exc)}

    return {**summary, **compat, **diagnostics}


def recommend_params(
    n_samples: "int | None" = None,
    n_features: "int | None" = None,
    n_classes: "int | None" = None,
    has_members: bool = False,
    members_m: "int | None" = None,
    diagnostics: "dict | None" = None,
) -> dict:
    """Рекомендовать параметры алгоритмов.

    Если передан полный словарь diagnostics (из compute_diagnostics),
    делегирует recommend_from_diagnostics для детальных рекомендаций.
    Иначе — простые правила по n/d/k (обратная совместимость).
    """
    from dataset_diagnostics import recommend_from_diagnostics

    if diagnostics:
        return recommend_from_diagnostics(diagnostics)

    # Простые правила (fallback: когда X не загружен)
    is_large = (n_samples or 0) > 2500
    if n_classes is not None:
        k_min = max(2, n_classes - 2)
        k_max = min(n_classes + 4, 20)
    else:
        k_min, k_max = 2, 10

    if has_members and members_m is not None:
        m = members_m
    elif (n_samples or 0) < 300:   m = 20
    elif (n_samples or 0) < 1000:  m = 30
    elif (n_samples or 0) < 5000:  m = 40
    else:                           m = 50

    strategy = "kmeans" if is_large else "mixed"
    sharpen  = 2.0 if (n_classes is not None and n_classes > 6) else 1.5
    dt       = 5 if is_large else 3

    warnings: list[str] = []
    if (n_features or 0) > 50:
        warnings.append("Высокая размерность (d > 50): рекомендуется zscore + feature subsampling.")
    if is_large:
        warnings.append("n > 2500: strategy=kmeans для построения ансамбля.")
    if n_classes is None:
        warnings.append("gt отсутствует: k_min/k_max по умолчанию.")

    return {
        "m": m, "k_min": k_min, "k_max": k_max,
        "strategy": strategy, "warnings": warnings,
        "per_algorithm": {
            "hierarchical_baseline": {"method": "average"},
            "hierarchical_weighted": {"method": "average", "sharpen": sharpen},
            "sdgca": {"method": "average", "lambda_": 0.09, "eta": 0.75, "theta": 0.65},
            "sdgca_modified": {"method": "average", "lambda_": 0.09, "eta": 0.75, "theta": 0.65, "diffusion_time": dt},
        },
    }


def _load_dataset_arrays(path: Path):
    path = Path(path)
    meta = {}
    members = None
    x = None
    gt = None
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "X" in data.files:
            x = np.asarray(data["X"], dtype=np.float64)
        if "gt" in data.files:
            gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        if "members" in data.files:
            members = np.asarray(data["members"], dtype=np.int64)
        if "meta" in data.files:
            meta_raw = data["meta"]
            if np.isscalar(meta_raw):
                meta_raw = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8", errors="ignore")
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {"raw_meta": meta_raw}
    elif path.suffix.lower() == ".mat":
        data = loadmat(path)
        for key in ("X", "x", "data", "fea", "features"):
            if key in data:
                x = np.asarray(data[key], dtype=np.float64)
                break
        if "gt" in data:
            gt = np.asarray(data["gt"]).reshape(-1).astype(np.int64)
        if "members" in data:
            members = np.asarray(data["members"], dtype=np.int64)
    elif path.suffix.lower() in TABLE_DATASET_SUFFIXES:
        x, gt, members, meta = _load_table_arrays(path)
    elif path.suffix.lower() == ".json":
        x, gt, members, meta = _load_json_arrays(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    return x, gt, members, meta


def build_consensus_ready_dataset(path: Path, n_clusterings, k_min, k_max, strategy):
    x, gt, members, meta = _load_dataset_arrays(path)
    built_members_from_x = False
    if gt is None:
        raise ValueError("Для оценки качества в датасете должен быть вектор gt.")
    if x is not None and x.shape[0] != gt.shape[0]:
        raise ValueError("Число строк в X должно совпадать с длиной gt.")
    if members is None:
        if x is None:
            raise ValueError("Чтобы построить базовые кластеризации, нужна матрица признаков X.")
        members, base_info = build_base_clusterings(
            x,
            n_clusterings=n_clusterings,
            k_min=k_min,
            k_max=k_max,
            rng=19,
            strategy=strategy,
            return_info=True,
        )
        built_members_from_x = True
    else:
        base_info = meta.get("base_info")
    meta = {
        **meta,
        "source_file": path.name,
        "base_clusterings": int(n_clusterings),
        "base_k_min": int(k_min),
        "base_k_max": int(k_max),
        "base_strategy": strategy,
        "base_info": base_info,
        "members_built_from_x": built_members_from_x,
    }
    output_path = path.with_name(f"{path.stem}_consensus_ready.npz")
    payload = {"gt": gt, "members": members, "meta": json.dumps(meta, ensure_ascii=False)}
    if x is not None:
        payload["X"] = x
    np.savez(output_path, **payload)
    return output_path, members


_UPSERT_DATASET_SQL = """
INSERT INTO datasets (
    name, path, suffix, size_kb, source, keys_json, x_shape_json, gt_shape_json,
    members_shape_json, n_classes, meta_json, error, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT(path) DO UPDATE SET
    name = EXCLUDED.name,
    suffix = EXCLUDED.suffix,
    size_kb = EXCLUDED.size_kb,
    source = EXCLUDED.source,
    keys_json = EXCLUDED.keys_json,
    x_shape_json = EXCLUDED.x_shape_json,
    gt_shape_json = EXCLUDED.gt_shape_json,
    members_shape_json = EXCLUDED.members_shape_json,
    n_classes = EXCLUDED.n_classes,
    meta_json = EXCLUDED.meta_json,
    error = EXCLUDED.error,
    updated_at = EXCLUDED.updated_at
"""


def upsert_dataset_record(summary):
    updated_at = datetime.now().isoformat(timespec="seconds")
    source = "uploaded" if UPLOADS_DIR in Path(summary["path"]).parents else "built_in"
    params = (
        summary["name"],
        summary["path"],
        summary.get("suffix"),
        summary.get("size_kb"),
        source,
        _jsonify_or_none(summary.get("keys")),
        _jsonify_or_none(summary.get("x_shape")),
        _jsonify_or_none(summary.get("gt_shape")),
        _jsonify_or_none(summary.get("members_shape")),
        summary.get("n_classes"),
        _jsonify_or_none(summary.get("meta")),
        summary.get("error"),
        updated_at,
    )
    _db_execute(_UPSERT_DATASET_SQL, params)


def sync_dataset_registry():
    files = []
    for suffix in sorted(SUPPORTED_DATASET_SUFFIXES):
        pattern = f"*{suffix}"
        files.extend(sorted(DATASETS_DIR.glob(pattern)))
        files.extend(sorted(UPLOADS_DIR.glob(pattern)))
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        upsert_dataset_record(summarize_dataset(path))


def list_datasets():
    sync_dataset_registry()
    rows = _db_query("""
        SELECT name, path, suffix, size_kb, source, keys_json, x_shape_json, gt_shape_json,
               members_shape_json, n_classes, meta_json, error
        FROM datasets
        ORDER BY lower(name), name
    """)

    items = []
    for row in rows:
        x_shape = _parse_json_field(row["x_shape_json"])
        gt_shape = _parse_json_field(row["gt_shape_json"])
        members_shape = _parse_json_field(row["members_shape_json"])
        items.append(
            {
                "name": row["name"],
                "path": row["path"],
                "suffix": row["suffix"],
                "size_kb": row["size_kb"],
                "source": row["source"],
                "keys": _parse_json_field(row["keys_json"]),
                "x_shape": tuple(x_shape) if x_shape else None,
                "gt_shape": tuple(gt_shape) if gt_shape else None,
                "members_shape": tuple(members_shape) if members_shape else None,
                "n_classes": row["n_classes"],
                "meta": _parse_json_field(row["meta_json"]),
                "error": row["error"],
            }
        )
    return items


_INSERT_RESULT_SQL = """
INSERT INTO results (
    created_at, dataset, algorithm, algorithm_label, method, method_label,
    seed, m, runs, nmi_mean, nmi_std, ari_mean, ari_std, f_mean, f_std,
    params_json, result_file
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _result_params_tuple(payload, result_file_str):
    """Собрать кортеж параметров для INSERT INTO results."""
    algo_params = {}
    for key in ("sharpen", "lambda_", "eta", "theta", "params"):
        if key in payload:
            algo_params[key] = payload[key]
    payload.setdefault("algorithm_label", ALGORITHM_LABELS.get(payload.get("algorithm"), payload.get("algorithm")))
    payload.setdefault("method_label", METHOD_LABELS.get(payload.get("method"), payload.get("method")))
    return (
        payload.get("created_at"),
        payload.get("dataset"),
        payload.get("algorithm"),
        payload.get("algorithm_label"),
        payload.get("method"),
        payload.get("method_label"),
        payload.get("seed"),
        payload.get("m"),
        payload.get("runs"),
        payload.get("nmi_mean"),
        payload.get("nmi_std"),
        payload.get("ari_mean"),
        payload.get("ari_std"),
        payload.get("f_mean"),
        payload.get("f_std"),
        _jsonify_or_none(algo_params),
        result_file_str,
    )


def import_existing_results():
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    existing = {
        row["result_file"]
        for row in _db_query("SELECT result_file FROM results WHERE result_file IS NOT NULL")
    }
    for path in result_files:
        path_str = str(path)
        if path_str in existing:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if any(not payload.get(f) for f in ("created_at", "dataset", "algorithm")):
            continue
        _db_execute(
            _INSERT_RESULT_SQL + "ON CONFLICT(result_file) DO NOTHING",
            _result_params_tuple(payload, path_str),
        )


def list_results():
    import_existing_results()
    rows = _db_query("""
        SELECT created_at, dataset, algorithm, algorithm_label, method, method_label,
               seed, m, runs, nmi_mean, nmi_std, ari_mean, ari_std, f_mean, f_std, params_json
        FROM results
        ORDER BY created_at DESC
        LIMIT 50
    """)
    items = []
    for row in rows:
        payload = dict(row)
        payload["params"] = _parse_json_field(payload.pop("params_json", None))
        payload["algorithm_label"] = payload.get("algorithm_label") or ALGORITHM_LABELS.get(payload.get("algorithm"), payload.get("algorithm"))
        payload["method_label"] = payload.get("method_label") or METHOD_LABELS.get(payload.get("method"), payload.get("method"))
        items.append(payload)
    return items


def build_base_context(active_page):
    datasets = list_datasets()
    results = list_results()
    runnable_datasets = [
        dataset
        for dataset in datasets
        if dataset.get("members_shape") and dataset.get("suffix") in RUNNABLE_DATASET_SUFFIXES
    ]
    return {
        "active_page": active_page,
        "datasets": datasets,
        "runnable_datasets": runnable_datasets,
        "results": results,
        "dataset_count": len(datasets),
        "result_count": len(results),
        "demo_mode": DEMO_MODE,
        "demo_message": "Публичная версия сайта работает в демонстрационном режиме: запуск алгоритмов, генерация новых датасетов и загрузка пользовательских файлов выполняются только локально.",
    }


def find_dataset_path(dataset_name):
    candidates = [
        DATASETS_DIR / f"{dataset_name}.npz",
        DATASETS_DIR / f"{dataset_name}.mat",
        UPLOADS_DIR / f"{dataset_name}.npz",
        UPLOADS_DIR / f"{dataset_name}.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset not found: {dataset_name}")


_UPSERT_RESULT_CONFLICT = """
ON CONFLICT(result_file) DO UPDATE SET
    created_at = EXCLUDED.created_at,
    dataset = EXCLUDED.dataset,
    algorithm = EXCLUDED.algorithm,
    algorithm_label = EXCLUDED.algorithm_label,
    method = EXCLUDED.method,
    method_label = EXCLUDED.method_label,
    seed = EXCLUDED.seed,
    m = EXCLUDED.m,
    runs = EXCLUDED.runs,
    nmi_mean = EXCLUDED.nmi_mean,
    nmi_std = EXCLUDED.nmi_std,
    ari_mean = EXCLUDED.ari_mean,
    ari_std = EXCLUDED.ari_std,
    f_mean = EXCLUDED.f_mean,
    f_std = EXCLUDED.f_std,
    params_json = EXCLUDED.params_json
"""


def save_result_record(payload):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = _json_safe(payload)
    safe_name = payload["dataset"].replace(" ", "_")
    output_path = RESULTS_DIR / f"{timestamp}_{payload['algorithm']}_{safe_name}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _db_execute(
        _INSERT_RESULT_SQL + _UPSERT_RESULT_CONFLICT,
        _result_params_tuple(payload, str(output_path)),
    )


def _optional_float(value):
    value = (value or "").strip()
    return None if value == "" else float(value)


@app.route("/", methods=["GET"])
def index():
    ensure_dirs()
    context = build_base_context("home")
    context["recent_results"] = context["results"][:5]
    return render_template("home.html", **context)


@app.route("/datasets", methods=["GET", "POST"])
def datasets_page():
    ensure_dirs()
    upload_message = None
    if request.method == "POST":
        if DEMO_MODE:
            upload_message = "В демонстрационной версии загрузка датасетов отключена. Для добавления собственных файлов используйте локальный запуск приложения."
            context = build_base_context("data")
            context["upload_message"] = upload_message
            return render_template("datasets.html", **context)
        file = request.files.get("dataset_file")
        upload_message = "Файл не выбран."
        if file and file.filename:
            if not allowed_dataset(file.filename):
                upload_message = "Можно загружать файлы форматов .npz, .mat, .csv, .tsv, .txt и .json."
            else:
                filename = secure_filename(file.filename)
                target = UPLOADS_DIR / filename
                file.save(target)
                summary = summarize_dataset(target)
                upsert_dataset_record(summary)
                if summary.get("members_shape") and summary.get("suffix") in RUNNABLE_DATASET_SUFFIXES:
                    upload_message = f"Датасет {filename} успешно загружен и готов к запуску консенсусных алгоритмов."
                elif summary.get("gt_shape") and (summary.get("x_shape") or summary.get("members_shape")):
                    ready_path, members = build_consensus_ready_dataset(
                        target,
                        int(request.form.get("upload_base_clusterings", 30)),
                        int(request.form.get("upload_base_k_min", 2)),
                        int(request.form.get("upload_base_k_max", 8)),
                        request.form.get("upload_base_strategy", "mixed"),
                    )
                    upsert_dataset_record(summarize_dataset(ready_path))
                    if summary.get("members_shape"):
                        upload_message = f"Датасет {filename} преобразован в готовую версию {ready_path.name}."
                    else:
                        upload_message = (
                            f"Обычный датасет {filename} загружен. "
                            f"Построены базовые кластеризации members {tuple(members.shape)} "
                            f"и сохранена готовая версия {ready_path.name}."
                        )
                else:
                    upload_message = (
                        f"Файл {filename} загружен, но для запуска нужны либо готовое поле members, "
                        "либо обычные данные X и gt для построения базовых кластеризаций."
                    )
    context = build_base_context("data")
    context["upload_message"] = upload_message
    return render_template("datasets.html", **context)


@app.route("/generate", methods=["GET", "POST"])
def generate_page():
    ensure_dirs()
    generation_result = None
    if request.method == "POST":
        if DEMO_MODE:
            generation_result = {"error": "В демонстрационной версии генерация датасетов отключена. Полная генерация доступна только при локальном запуске приложения."}
            context = build_base_context("data")
            context["generation_result"] = generation_result
            return render_template("generate.html", **context)
        generator_type = request.form.get("generator_type", "densired")
        dataset_name = request.form.get("dataset_name", "").strip() or f"generated_{generator_type}"
        try:
            if generator_type == "simple":
                params = {
                    "name": dataset_name,
                    "n_samples": int(request.form.get("simple_n_samples", 1000)),
                    "n_clusters": int(request.form.get("simple_n_clusters", 4)),
                    "dim": int(request.form.get("simple_dim", 2)),
                    "cluster_std": float(request.form.get("simple_cluster_std", 0.6)),
                    "separation": float(request.form.get("simple_separation", 4.0)),
                    "imbalance_ratio": float(request.form.get("simple_imbalance_ratio", 1.0)),
                    "seed": int(request.form.get("simple_seed", 19)),
                    "base_clusterings": int(request.form.get("simple_base_clusterings", 30)),
                    "base_k_min": int(request.form.get("simple_base_k_min", 2)),
                    "base_k_max": int(request.form.get("simple_base_k_max", 8)),
                    "base_strategy": request.form.get("simple_base_strategy", "mixed"),
                }
                x, gt, members, meta = generate_simple_gaussian_dataset(**params)
                output_path = DATASETS_DIR / f"{dataset_name}.npz"
                save_simple_dataset(output_path, x, gt, members, meta)
            elif generator_type == "densired":
                params = {
                    "name": dataset_name,
                    "dim": int(request.form.get("densired_dim", 2)),
                    "clunum": int(request.form.get("densired_clunum", 6)),
                    "core_num": int(request.form.get("densired_core_num", 80)),
                    "data_num": int(request.form.get("densired_data_num", 2000)),
                    "seed": int(request.form.get("densired_seed", 19)),
                    "domain_size": float(request.form.get("densired_domain_size", 20)),
                    "radius": float(request.form.get("densired_radius", 0.038)),
                    "step": float(request.form.get("densired_step", 0.055)),
                    "noise_ratio": float(request.form.get("densired_noise_ratio", 0.1)),
                    "density_factors": [
                        float(x.strip())
                        for x in request.form.get("densired_density_factors", "1,1,0.8,1.2,0.6,1.4").split(",")
                        if x.strip()
                    ],
                    "momentum": float(request.form.get("densired_momentum", 0.25)),
                    "branch": float(request.form.get("densired_branch", 0.05)),
                    "star": float(request.form.get("densired_star", 0.1)),
                    "distribution": request.form.get("densired_distribution", "uniform"),
                    "base_clusterings": int(request.form.get("densired_base_clusterings", 30)),
                    "base_k_min": int(request.form.get("densired_base_k_min", 2)),
                    "base_k_max": int(request.form.get("densired_base_k_max", 8)),
                    "base_strategy": request.form.get("densired_base_strategy", "mixed"),
                }
                if len(params["density_factors"]) != params["clunum"]:
                    params["density_factors"] = [1.0] * params["clunum"]
                x, gt, members, meta = generate_densired_style_dataset(**params)
                output_path = DATASETS_DIR / f"{dataset_name}.npz"
                save_densired_dataset(output_path, x, gt, members, meta)
            elif generator_type == "repliclust":
                distributions = [
                    x.strip()
                    for x in request.form.get("repliclust_distributions", "normal,student_t,lognormal").split(",")
                    if x.strip()
                ]
                proportions = [
                    float(x.strip())
                    for x in request.form.get("repliclust_distribution_proportions", "0.5,0.3,0.2").split(",")
                    if x.strip()
                ]
                if len(distributions) != len(proportions):
                    raise ValueError("Число распределений и число пропорций должно совпадать.")
                params = {
                    "name": dataset_name,
                    "n_clusters": int(request.form.get("repliclust_n_clusters", 6)),
                    "dim": int(request.form.get("repliclust_dim", 2)),
                    "n_samples": int(request.form.get("repliclust_n_samples", 2500)),
                    "aspect_ref": float(request.form.get("repliclust_aspect_ref", 4.0)),
                    "aspect_maxmin": float(request.form.get("repliclust_aspect_maxmin", 6.0)),
                    "radius_ref": float(request.form.get("repliclust_radius_ref", 1.0)),
                    "radius_maxmin": float(request.form.get("repliclust_radius_maxmin", 2.0)),
                    "min_overlap": float(request.form.get("repliclust_min_overlap", 0.08)),
                    "max_overlap": float(request.form.get("repliclust_max_overlap", 0.18)),
                    "imbalance_ratio": float(request.form.get("repliclust_imbalance_ratio", 2.5)),
                    "distributions": distributions,
                    "distribution_proportions": proportions,
                    "seed": int(request.form.get("repliclust_seed", 19)),
                    "base_clusterings": int(request.form.get("repliclust_base_clusterings", 30)),
                    "base_k_min": int(request.form.get("repliclust_base_k_min", 2)),
                    "base_k_max": int(request.form.get("repliclust_base_k_max", 8)),
                    "base_strategy": request.form.get("repliclust_base_strategy", "mixed"),
                }
                x, gt, members, meta = generate_archetype_dataset(**params)
                output_path = DATASETS_DIR / f"{dataset_name}.npz"
                save_repliclust_dataset(output_path, x, gt, members, meta)
            elif generator_type == "qiu_joe":
                params = {
                    "name": dataset_name,
                    "n_samples": int(request.form.get("qiu_n_samples", 2000)),
                    "n_clusters": int(request.form.get("qiu_n_clusters", 6)),
                    "dim": int(request.form.get("qiu_dim", 10)),
                    "overlap_level": request.form.get("qiu_overlap_level", "medium"),
                    "separation": float(request.form.get("qiu_separation", 1.0)),
                    "shape_ratio": float(request.form.get("qiu_shape_ratio", 6.0)),
                    "volume_mean": float(request.form.get("qiu_volume_mean", 1.0)),
                    "imbalance_ratio": float(request.form.get("qiu_imbalance_ratio", 2.0)),
                    "orientation": request.form.get("qiu_orientation", "random"),
                    "noise_ratio": float(request.form.get("qiu_noise_ratio", 0.0)),
                    "seed": int(request.form.get("qiu_seed", 19)),
                    "base_clusterings": int(request.form.get("qiu_base_clusterings", 30)),
                    "base_k_min": int(request.form.get("qiu_base_k_min", 2)),
                    "base_k_max": int(request.form.get("qiu_base_k_max", 8)),
                    "base_strategy": request.form.get("qiu_base_strategy", "mixed"),
                }
                x, gt, members, meta = generate_qiu_joe_style_dataset(**params)
                output_path = DATASETS_DIR / f"{dataset_name}.npz"
                save_qiu_joe_dataset(output_path, x, gt, members, meta)
            else:
                raise ValueError("Неизвестный тип генератора.")

            upsert_dataset_record(summarize_dataset(output_path))
            generation_result = {
                "message": f"Датасет {output_path.name} успешно сгенерирован и сохранён.",
                "file": output_path.name,
                "dataset_id": output_path.stem,
                "x_shape": tuple(x.shape),
                "gt_shape": tuple(gt.shape),
                "members_shape": tuple(members.shape),
                "meta_pretty": json.dumps(meta, ensure_ascii=False, indent=2),
            }
        except Exception as exc:
            generation_result = {"error": str(exc)}
    context = build_base_context("data")
    context["generation_result"] = generation_result
    return render_template("generate.html", **context)


def _extract_algorithm_kwargs(algorithm_name, dataset_name, form):
    """Извлечь специфичные для алгоритма параметры из формы."""
    if algorithm_name == "hierarchical_weighted":
        return {"sharpen": float(form.get("sharpen", 1.5))}
    if algorithm_name == "sdgca":
        return {
            "nwca_para": float(form.get("lambda_", 0.09)),
            "eta": float(form.get("eta", 0.75)),
            "theta": float(form.get("theta", 0.65)),
        }
    if algorithm_name == "sdgca_modified":
        params = resolve_params(
            dataset_name,
            _optional_float(form.get("lambda_mod")),
            _optional_float(form.get("eta_mod")),
            _optional_float(form.get("theta_mod")),
        )
        return {
            "nwca_para": params["lambda_"],
            "eta": params["eta"],
            "theta": params["theta"],
            "diffusion_time": params["diffusion_time"],
        }
    return {}


@app.route("/test", methods=["GET", "POST"])
def test_page():
    ensure_dirs()
    run_results = []
    selected_dataset = request.args.get("dataset", "")
    if request.method == "POST":
        dataset_name = request.form.get("selected_dataset")
        selected_dataset = dataset_name or selected_dataset
        algorithm_names = request.form.getlist("algorithms")
        if not algorithm_names:
            algorithm_names = [request.form.get("algorithm", "hierarchical_baseline")]
        seed = int(request.form.get("seed", 19))
        m = int(request.form.get("m", 20))
        runs = int(request.form.get("runs", 5))
        method = request.form.get("method", "average")
        from algorithms_base import AlgorithmRegistry
        dataset_path = find_dataset_path(dataset_name)
        for algorithm_name in algorithm_names:
            try:
                algo_kwargs = _extract_algorithm_kwargs(algorithm_name, dataset_name, request.form)
                algo = AlgorithmRegistry.get(algorithm_name)
                cr = algo.run(dataset_path, m=m, runs=runs, method=method, seed=seed, **algo_kwargs)
                rec = {
                    "dataset": dataset_name,
                    "algorithm": algorithm_name,
                    "algorithm_label": ALGORITHM_LABELS.get(algorithm_name, algorithm_name),
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "seed": seed,
                    "m": m,
                    "runs": runs,
                    "nmi_mean": round(cr.nmi_mean, 6),
                    "nmi_std": round(cr.nmi_std, 6),
                    "ari_mean": round(cr.ari_mean, 6),
                    "ari_std": round(cr.ari_std, 6),
                    "f_mean": round(cr.f_mean, 6),
                    "f_std": round(cr.f_std, 6),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "error": None,
                }
                if cr.extra:
                    rec.update(_json_safe({k: v for k, v in cr.extra.items() if k != "_extra_per_run"}))
                save_result_record(rec)
            except Exception as exc:
                rec = {
                    "algorithm": algorithm_name,
                    "algorithm_label": ALGORITHM_LABELS.get(algorithm_name, algorithm_name),
                    "error": str(exc),
                }
            run_results.append(rec)

    context = build_base_context("lab")
    context["run_results"] = run_results
    context["selected_dataset"] = selected_dataset
    return render_template("test.html", **context)


@app.route("/results", methods=["GET"])
def results_page():
    ensure_dirs()
    context = build_base_context("lab")
    return render_template("results.html", **context)


@app.route("/api/results-json", methods=["GET"])
def api_results_json():
    """JSON-endpoint for analytics charts. Returns DB results + QD experiment TSV."""
    from flask import jsonify
    import csv as csv_module

    items = list_results()
    try:
        sys.path.insert(0, str(CONSENSUS_LAB_DIR))
        from ensemble_selection import get_dataset_type, DATASET_TYPE_LABELS
        for item in items:
            ds_name = item.get("dataset", "")
            item["dataset_type"] = get_dataset_type(ds_name)
            item["dataset_type_label"] = DATASET_TYPE_LABELS.get(
                item["dataset_type"], item["dataset_type"]
            )
    except Exception:
        for item in items:
            item["dataset_type"] = "unknown"
            item["dataset_type_label"] = "Неизвестный тип"

    qd_rows = []
    designed_combined = RESULTS_DIR / "designed_qd_experiment_combined.tsv"
    qd_tsv_files = [RESULTS_DIR / "qd_selection_analysis.tsv"]
    if designed_combined.exists():
        qd_tsv_files.append(designed_combined)
    else:
        qd_tsv_files.append(RESULTS_DIR / "designed_qd_experiment_latest.tsv")
    for qd_tsv in qd_tsv_files:
        if not qd_tsv.exists():
            continue
        try:
            with qd_tsv.open(encoding="utf-8") as f:
                reader = csv_module.DictReader(f, delimiter="\t")
                for row in reader:
                    for key in ("nmi_mean", "nmi_std", "ari_mean", "ari_std",
                                "f_mean", "f_std", "runtime_sec"):
                        try:
                            row[key] = float(row[key]) if row.get(key) else None
                        except ValueError:
                            row[key] = None
                    row["source_file"] = qd_tsv.name
                    qd_rows.append(row)
        except Exception:
            pass

    # --- ablation results ---
    ablation_rows = []
    ablation_tsv = RESULTS_DIR / "sdgca_modified_ablation_latest.tsv"
    if ablation_tsv.exists():
        try:
            with ablation_tsv.open(encoding="utf-8") as f:
                reader = csv_module.DictReader(f, delimiter="\t")
                for row in reader:
                    for key in (
                        "nmi_mean",
                        "nmi_std",
                        "ari_mean",
                        "ari_std",
                        "f_mean",
                        "f_std",
                        "runtime_sec",
                        "memory_peak_mb",
                        "n_clusters_ensemble",
                    ):
                        try:
                            row[key] = (
                                float(row[key])
                                if row.get(key) not in ("", None)
                                else None
                            )
                        except ValueError:
                            row[key] = None
                    for key in ("diffusion_time_used",):
                        val = row.get(key, "")
                        try:
                            row[key] = float(val) if val not in ("", None) else None
                        except (ValueError, TypeError):
                            row[key] = None
                    ablation_rows.append(row)
        except Exception:
            pass

    return jsonify({"db_results": items, "qd_results": qd_rows, "ablation_results": ablation_rows})


@app.route("/analytics", methods=["GET"])
def analytics_page():
    """Interactive analytics: algorithm comparison charts by metric and dataset type."""
    ensure_dirs()
    context = build_base_context("analytics")
    return render_template("analytics.html", **context)


@app.route("/api/single-vs-consensus-json", methods=["GET"])
def api_single_vs_consensus_json():
    """JSON endpoint for the massive single-vs-consensus experiment."""
    from flask import jsonify
    import csv as csv_module

    def read_tsv(path, numeric_fields=()):
        rows = []
        if not path.exists():
            return rows
        try:
            with path.open(encoding="utf-8") as f:
                reader = csv_module.DictReader(f, delimiter="\t")
                for row in reader:
                    for field in numeric_fields:
                        try:
                            row[field] = (
                                float(row[field])
                                if row.get(field) not in ("", None)
                                else None
                            )
                        except (ValueError, TypeError):
                            row[field] = None
                    rows.append(row)
        except Exception:
            return []
        return rows

    benchmark_rows = read_tsv(
        RESULTS_DIR / "single_vs_consensus_benchmark.tsv",
        numeric_fields=(
            "n_samples",
            "n_clusters",
            "dim",
            "fold_id",
            "m",
            "seed",
            "NMI",
            "ARI",
            "F-score",
            "runtime_sec",
            "memory_peak_mb",
        ),
    )
    by_type_rows = read_tsv(
        RESULTS_DIR / "single_vs_consensus_by_type.tsv",
        numeric_fields=(
            "datasets",
            "mean_single_nmi",
            "mean_consensus_nmi",
            "mean_delta_nmi",
            "win_rate",
            "mean_runtime_ratio",
        ),
    )
    stat_rows = read_tsv(
        RESULTS_DIR / "single_vs_consensus_stat_tests.tsv",
        numeric_fields=("statistic", "p_value", "n"),
    )
    failure_rows = read_tsv(
        RESULTS_DIR / "failure_cases.tsv",
        numeric_fields=("best_nmi", "consensus_delta_vs_single"),
    )

    selected_profile = None
    selected_path = RESULTS_DIR / "selected_consensus_profile.json"
    if selected_path.exists():
        try:
            selected_profile = json.loads(selected_path.read_text(encoding="utf-8"))
        except Exception:
            selected_profile = None

    return jsonify(
        {
            "benchmark_rows": benchmark_rows,
            "by_type_rows": by_type_rows,
            "stat_rows": stat_rows,
            "failure_rows": failure_rows,
            "selected_profile": selected_profile,
        }
    )


@app.route("/massive-analytics", methods=["GET"])
def massive_analytics_page():
    """Interactive charts for dataset-level single-vs-consensus experiments."""
    ensure_dirs()
    context = {
        "active_page": "analytics",
        "demo_mode": DEMO_MODE,
        "demo_message": "Публичная версия сайта работает в демонстрационном режиме: запуск алгоритмов, генерация новых датасетов и загрузка пользовательских файлов выполняются только локально.",
    }
    return render_template("massive_analytics.html", **context)


@app.route("/api/dataset-analysis/<dataset_name>", methods=["GET"])
def api_dataset_analysis(dataset_name):
    """Полная диагностика датасета + рекомендации параметров.

    Возвращает:
      analysis: все метрики из compute_diagnostics + флаги совместимости
      recommendations: рекомендованные параметры с обоснованием
    """
    from flask import jsonify

    candidates = []
    for suffix in sorted(SUPPORTED_DATASET_SUFFIXES):
        candidates.append(DATASETS_DIR / f"{dataset_name}{suffix}")
        candidates.append(UPLOADS_DIR / f"{dataset_name}{suffix}")
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return jsonify({"error": f"Датасет не найден: {dataset_name}"}), 404

    try:
        analysis = analyze_dataset_structure(path)
        # Используем полную диагностику для рекомендаций если она вычислена
        recommendations = recommend_params(diagnostics=analysis)
        # Сериализуем: numpy-типы → Python
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(i) for i in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        return jsonify({"analysis": _clean(analysis), "recommendations": _clean(recommendations)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ai-agent/dataset/<dataset_name>", methods=["GET"])
def api_ai_agent_dataset(dataset_name):
    """AI interpretation using Groq LLM via ai_agent/client.py."""
    from flask import jsonify

    candidates = []
    for suffix in sorted(SUPPORTED_DATASET_SUFFIXES):
        candidates.append(DATASETS_DIR / f"{dataset_name}{suffix}")
        candidates.append(UPLOADS_DIR / f"{dataset_name}{suffix}")
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return jsonify({"error": f"Датасет не найден: {dataset_name}"}), 404

    try:
        analysis = analyze_dataset_structure(path)
        recommendations = recommend_params(diagnostics=analysis)

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(i) for i in obj]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_analysis = _clean(analysis)
        clean_recs = _clean(recommendations)

        prompt = (
            "Ты эксперт по кластеризации данных. Проанализируй датасет и верни ответ ТОЛЬКО в виде JSON, "
            "без каких-либо пояснений вне JSON.\n\n"
            f"Датасет: {dataset_name}\n"
            f"Объектов: {clean_analysis.get('n_objects') or clean_analysis.get('n_samples')}, "
            f"Признаков: {clean_analysis.get('n_features')}, "
            f"Классов: {clean_analysis.get('n_classes')}\n"
            f"Силуэт: {clean_analysis.get('silhouette_score')}, "
            f"Перекрытие: {clean_analysis.get('overlap_ratio')}, "
            f"Дисбаланс: {clean_analysis.get('imbalance_ratio')}, "
            f"Вариация плотности: {clean_analysis.get('density_variation')}, "
            f"Вытянутость: {clean_analysis.get('elongation_max')}, "
            f"Выбросы: {clean_analysis.get('outlier_ratio')}\n"
            f"Есть members: {clean_analysis.get('has_members')}\n"
            f"Рекомендованные параметры: m={clean_recs.get('m')}, "
            f"k_min={clean_recs.get('k_min')}, k_max={clean_recs.get('k_max')}, "
            f"strategy={clean_recs.get('strategy')}\n\n"
            'Верни JSON строго следующей структуры (все поля на русском языке):\n'
            '{\n'
            '  "summary": "краткий вывод о датасете и лучшем алгоритме (2-3 предложения)",\n'
            '  "findings": ["наблюдение 1", "наблюдение 2", "наблюдение 3"],\n'
            '  "warnings": ["предупреждение если есть, иначе пустой массив"],\n'
            '  "recommended_parameters": {"m": 20, "k_min": 2, "k_max": 8, "strategy": "mixed", "algorithm": "название"},\n'
            '  "evidence": {"silhouette_score": "0.xxx", "overlap_ratio": "0.xxx", "imbalance_ratio": "0.xxx"}\n'
            '}'
        )

        import re as _re
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR))
        try:
            from ai_agent.client import ask_llm
            llm_raw = ask_llm(prompt)
            json_match = _re.search(r'\{[\s\S]*\}', llm_raw)
            if json_match:
                agent = json.loads(json_match.group())
            else:
                agent = {"summary": llm_raw, "findings": [], "warnings": [],
                         "recommended_parameters": clean_recs, "evidence": {}}
        except Exception as llm_exc:
            agent = {
                "summary": (
                    f"ИИ-агент недоступен: {str(llm_exc)[:120]}. "
                    "Рекомендации вычислены по диагностике датасета."
                ),
                "findings": [],
                "warnings": [],
                "recommended_parameters": clean_recs,
                "evidence": {},
            }

        return jsonify({
            "dataset": dataset_name,
            "agent": agent,
            "analysis": clean_analysis,
            "recommendations": clean_recs,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/assistant", methods=["GET"])
def assistant_page():
    """AI assistant page with Groq-powered chat."""
    ensure_dirs()
    context = build_base_context("assistant")
    return render_template("assistant.html", **context)


@app.route("/api/ai-agent/chat", methods=["POST"])
def api_ai_agent_chat():
    """Free-form chat endpoint using Groq LLM with conversation history."""
    from flask import jsonify
    import sys as _sys
    _sys.path.insert(0, str(BASE_DIR))

    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Пустое сообщение"}), 400

    ctx_dataset = data.get("dataset", "")
    # history: list of {role, content} from the frontend (prior turns only)
    history = data.get("history", [])
    # Keep at most last 10 turns to stay within token budget
    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in history[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    # --- Build dataset context ---
    dataset_context = {"dataset": ctx_dataset or None}
    if ctx_dataset:
        try:
            dataset_path = find_dataset_path(ctx_dataset)
            analysis = _json_safe(analyze_dataset_structure(dataset_path))
            recommendations = _json_safe(recommend_params(diagnostics=analysis))
            useful_analysis_keys = [
                "n_objects", "n_samples", "n_features", "n_classes", "has_members",
                "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score",
                "density_variation", "elongation_max", "imbalance_ratio", "overlap_ratio",
                "outlier_ratio", "effective_dimension_90", "explained_variance_2d",
                "explained_variance_10d",
            ]
            useful_recommendation_keys = [
                "m", "k_min", "k_max", "strategy", "preprocessing", "per_algorithm", "warnings",
            ]
            density_variation = analysis.get("density_variation") or 0
            elongation_max = analysis.get("elongation_max") or 0
            overlap_ratio = analysis.get("overlap_ratio") or 0
            if density_variation > 1.0 or elongation_max > 8.0 or overlap_ratio > 0.25:
                primary_start = "sdgca_modified"
                secondary_start = "hierarchical_weighted"
                recommendation_reason = (
                    "датасет имеет сложную структуру: неоднородная плотность, вытянутость "
                    "или перекрытие; стоит проверить модифицированный SDGCA и сравнить с "
                    "взвешенной иерархической версией"
                )
            else:
                primary_start = "hierarchical_weighted"
                secondary_start = "hierarchical_baseline"
                recommendation_reason = (
                    "структура выглядит достаточно устойчивой; разумно начать с взвешенной "
                    "иерархической консенсус кластеризации и сравнить её с базовой версией"
                )
            dataset_context.update(
                {
                    "diagnostics": {key: analysis.get(key) for key in useful_analysis_keys},
                    "recommendations": {key: recommendations.get(key) for key in useful_recommendation_keys},
                    "system_recommendation": {
                        "primary_start": primary_start,
                        "secondary_start": secondary_start,
                        "reason": recommendation_reason,
                        "m": recommendations.get("m"),
                        "k_min": recommendations.get("k_min"),
                        "k_max": recommendations.get("k_max"),
                        "strategy": recommendations.get("strategy"),
                        "primary_params": (recommendations.get("per_algorithm") or {}).get(primary_start),
                        "secondary_params": (recommendations.get("per_algorithm") or {}).get(secondary_start),
                    },
                }
            )
        except Exception as ctx_exc:
            dataset_context["context_error"] = str(ctx_exc)

    # --- Pull last 5 experiment results from DB ---
    try:
        db_results = list_results()[:5]
        recent_results_summary = [
            {
                "dataset": r.get("dataset"),
                "algorithm": r.get("algorithm_label") or r.get("algorithm"),
                "nmi_mean": r.get("nmi_mean"),
                "ari_mean": r.get("ari_mean"),
                "f_mean": r.get("f_mean"),
                "m": r.get("m"),
                "runs": r.get("runs"),
                "created_at": r.get("created_at"),
            }
            for r in db_results
        ]
    except Exception:
        recent_results_summary = []

    # --- System message with full project context ---
    system_content = (
        "Ты ИИ-агент внутри веб-системы ВКР на тему:\n"
        "«Разработка системы тестирования иерархических алгоритмов консенсус кластеризации».\n\n"
        "ЖЁСТКИЕ ПРАВИЛА:\n"
        "1. Отвечай только на русском языке, строго в рамках проекта.\n"
        "2. Алгоритмы системы (ТОЛЬКО ЭТИ ЧЕТЫРЕ):\n"
        "   - hierarchical_baseline: строит обычную co-association matrix и применяет иерархическую кластеризацию.\n"
        "   - hierarchical_weighted: взвешенная co-association matrix по качеству базовых разбиений.\n"
        "   - sdgca: SDGCA — Similarity and Dissimilarity Guided Co-association Matrix Construction.\n"
        "   - sdgca_modified: модифицированный SDGCA с нечёткими энтропийными весами и графовой диффузией.\n"
        "3. ЗАПРЕЩЕНО рекомендовать k-means, DBSCAN, HDBSCAN, GMM как основной ответ "
        "   (можно упоминать только как внешние baseline, если пользователь прямо спрашивает).\n"
        "4. Не придумывай числа. Если датасет выбран — используй только данные из 'dataset_context'. "
        "   Если датасет не выбран — честно скажи, что нужно выбрать датасет.\n"
        "5. Если есть 'recent_results' — анализируй РЕАЛЬНЫЕ числа из экспериментов, "
        "   сравнивай алгоритмы между собой, указывай, какой показал лучший NMI/ARI/F-score.\n"
        "6. gt используется ТОЛЬКО для оценки NMI/ARI/F-score, не для кластеризации.\n"
        "7. Давай практические советы: что выбрать на странице 'Лаборатория', "
        "   какие параметры m, method, runs попробовать, как интерпретировать результат.\n"
        "8. Помни историю диалога — используй контекст предыдущих вопросов.\n\n"
        f"КОНТЕКСТ ДАТАСЕТА:\n{json.dumps(_json_safe(dataset_context), ensure_ascii=False, indent=2)}\n\n"
        f"ПОСЛЕДНИЕ ЭКСПЕРИМЕНТЫ В БД:\n{json.dumps(recent_results_summary, ensure_ascii=False, indent=2)}"
    )

    # --- Assemble messages: system + history + current user message ---
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    try:
        from ai_agent.client import ask_llm_with_messages
        reply = ask_llm_with_messages(messages)
        return jsonify({"response": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    ensure_dirs()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5000)))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
