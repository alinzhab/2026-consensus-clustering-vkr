from __future__ import annotations
import argparse
import gzip
import io
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'consensus_lab'))
from base_clusterings import build_base_clusterings
OUT_DIR = ROOT / 'datasets' / 'real'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260510
N_CLUSTERINGS = 30
TIMEOUT = 90

def _normalize_labels(y) -> np.ndarray:
    _, inv = np.unique(y, return_inverse=True)
    return (inv + 1).astype(np.int64)

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd

def _save(name: str, X: np.ndarray, gt_raw, meta: dict, force: bool=False) -> bool:
    path = OUT_DIR / f'{name}.npz'
    if path.exists() and (not force):
        print(f'  skip {name}.npz (already exists)')
        return False
    X = np.asarray(X, dtype=np.float64)
    X = X[:, X.std(axis=0) > 0]
    X = _zscore(X)
    gt = _normalize_labels(np.asarray(gt_raw))
    k = int(np.unique(gt).size)
    members, info = build_base_clusterings(X, n_clusterings=N_CLUSTERINGS, k_min=max(2, k - 2), k_max=k + 3, rng=SEED, return_info=True)
    meta.update({'n': int(X.shape[0]), 'd': int(X.shape[1]), 'K': k, 'ensemble': info})
    np.savez_compressed(path, X=X, gt=gt, members=members, meta=np.array(json.dumps(meta, ensure_ascii=False), dtype=object))
    counts = [int(np.sum(gt == c)) for c in np.unique(gt)]
    print(f"  saved {path.name}: n={X.shape[0]} d={X.shape[1]} K={k} sizes={counts[:8]}{('...' if k > 8 else '')}")
    return True

def _get(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    return r.content

def _head_ok(url: str) -> bool:
    try:
        return requests.head(url, timeout=10, allow_redirects=True).ok
    except Exception:
        return False

def _parse_gct(content: bytes, label_row_keyword: str | None=None) -> tuple[np.ndarray, list[str]]:
    lines = content.decode('utf-8', errors='replace').splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith('Name')))
    header = lines[start].split('\t')
    sample_ids = header[2:]
    rows = []
    for line in lines[start + 1:]:
        parts = line.split('\t')
        try:
            rows.append([float(v) for v in parts[2:2 + len(sample_ids)]])
        except (ValueError, IndexError):
            pass
    X = np.array(rows, dtype=np.float64).T
    return (X, sample_ids)

def _parse_cls(content: bytes) -> np.ndarray:
    lines = [l.strip() for l in content.decode().splitlines() if l.strip()]
    return np.array(lines[2].split(), dtype=int)

def _parse_geo_gds(gds_id: str, max_genes: int=4000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import re
    prefix = f'GDS{int(gds_id[3:]) // 1000}nnn'
    url = f'https://ftp.ncbi.nlm.nih.gov/geo/datasets/{prefix}/{gds_id}/soft/{gds_id}.soft.gz'
    print(f'    fetching {url} ...')
    raw = _get(url)
    text = gzip.decompress(raw).decode('utf-8', errors='replace')
    sample_class: dict[str, str] = {}
    for desc_m, ids_m in zip(re.finditer('!subset_description\\s*=\\s*(.+)', text), re.finditer('!subset_sample_id\\s*=\\s*(.+)', text)):
        class_name = desc_m.group(1).strip()
        for sid in ids_m.group(1).strip().split(','):
            sample_class[sid.strip()] = class_name
    if not sample_class:
        for block_m in re.finditer('\\^SUBSET\\s*=.*?(?=\\^SUBSET|\\^DATASET|\\Z)', text, re.DOTALL):
            block = block_m.group(0)
            desc = re.search('!subset_description\\s*=\\s*(.+)', block)
            ids = re.search('!subset_sample_id\\s*=\\s*(.+)', block)
            if desc and ids:
                class_name = desc.group(1).strip()
                for sid in ids.group(1).strip().split(','):
                    sample_class[sid.strip()] = class_name
    tstart = text.find('!dataset_table_begin')
    tend = text.find('!dataset_table_end')
    if tstart < 0 or tend < 0:
        raise ValueError(f'No expression table in {gds_id}')
    table_lines = text[tstart:tend].splitlines()
    hdr = table_lines[1].split('\t')
    col_ids = [c.strip() for c in hdr[2:]]
    n_samp = len(col_ids)
    rows = []
    for line in table_lines[2:]:
        parts = line.split('\t')
        try:
            vals = [float(p) for p in parts[2:2 + n_samp]]
            if len(vals) == n_samp:
                rows.append(vals)
        except (ValueError, IndexError):
            pass
    if not rows:
        raise ValueError(f'No numeric rows parsed from {gds_id}')
    X = np.array(rows, dtype=np.float64).T
    if X.shape[1] > max_genes:
        idx = np.argsort(X.var(0))[::-1][:max_genes]
        X = X[:, idx]
    titles = [sample_class.get(sid, 'unknown') for sid in col_ids[:X.shape[0]]]
    n_unique = len(set(titles))
    if n_unique < 2:
        raise ValueError(f'{gds_id}: only {n_unique} class(es) found in SUBSET blocks (sample_class has {len(sample_class)} entries, col_ids[:5]={col_ids[:5]})')
    gt = _normalize_labels(pd.Categorical(titles).codes)
    return (X, gt, titles)

def dl_golub(force: bool) -> None:
    print('Golub Leukemia 1999 (ALL/AML)...')
    raw = _get('http://hastie.su.domains/CASI_files/DATA/leukemia_big.csv')
    df = pd.read_csv(io.BytesIO(raw))
    sample_names = df.columns.tolist()
    gt_raw = np.array([0 if 'ALL' in s else 1 for s in sample_names])
    X = df.values.T.astype(np.float64)
    _save('leukemia_golub', X, gt_raw, {'source': 'Golub et al. Science 1999', 'description': 'Acute leukemia ALL(47) vs AML(25), 7128 genes Affymetrix'}, force)

def dl_wdbc(force: bool) -> None:
    print('WDBC Breast Cancer (sklearn)...')
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer()
    _save('wdbc_breast_cancer', ds.data, ds.target, {'source': 'UCI WDBC / sklearn', 'description': 'Breast tumour malignant vs benign, 30 morphological features'}, force)

def dl_prostate(force: bool) -> None:
    print('Prostate cancer Singh 2002 (GDS2545)...')
    try:
        X, gt, titles = _parse_geo_gds('GDS2545', max_genes=3000)
        _save('prostate_singh', X, gt, {'source': 'Singh et al. Cell 2002 / GEO GDS2545', 'description': 'Prostate cancer tumour vs normal, 12600→3000 top-var genes'}, force)
    except Exception as e:
        print(f'  SKIP: {e}')

def dl_golub_geo(force: bool) -> None:
    print('Golub Leukemia GEO (GDS3955)...')
    try:
        X, gt, titles = _parse_geo_gds('GDS3955', max_genes=3000)
        _save('leukemia_geo', X, gt, {'source': 'GEO GDS3955 / Golub 1999', 'description': 'Leukemia ALL vs AML from GEO, top-3000 variable genes'}, force)
    except Exception as e:
        print(f'  SKIP: {e}')

def dl_bladder(force: bool) -> None:
    print('Bladder cancer (GDS1479)...')
    try:
        X, gt, titles = _parse_geo_gds('GDS1479', max_genes=2000)
        _save('bladder_cancer', X, gt, {'source': 'GEO GDS1479', 'description': 'Bladder cancer subtypes, microarray'}, force)
    except Exception as e:
        print(f'  SKIP: {e}')

def dl_breast_geo(force: bool) -> None:
    print('Breast cancer PAM50 (GDS4685)...')
    try:
        X, gt, titles = _parse_geo_gds('GDS4685', max_genes=2000)
        _save('breast_cancer_pam50', X, gt, {'source': 'GEO GDS4685', 'description': 'Breast cancer PAM50 molecular subtypes, microarray'}, force)
    except Exception as e:
        print(f'  SKIP: {e}')

def dl_ovarian(force: bool) -> None:
    print('Ovarian cancer (GDS3592)...')
    try:
        X, gt, titles = _parse_geo_gds('GDS3592', max_genes=2000)
        _save('ovarian_cancer', X, gt, {'source': 'GEO GDS3592', 'description': 'Ovarian cancer subtypes, microarray'}, force)
    except Exception as e:
        print(f'  SKIP: {e}')

def dl_uci(name: str, dataset_id: int, description: str, force: bool, drop_cols: list[str] | None=None) -> None:
    from ucimlrepo import fetch_ucirepo
    print(f'{name} (UCI id={dataset_id})...')
    ds = fetch_ucirepo(id=dataset_id)
    X_df = ds.data.features.copy()
    y_df = ds.data.targets
    if drop_cols:
        X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], inplace=True, errors='ignore')
    for col in X_df.select_dtypes(['object', 'category']).columns:
        X_df[col] = pd.Categorical(X_df[col]).codes.astype(float).replace(-1, np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(np.float64)
    if isinstance(y_df, pd.DataFrame):
        gt_raw = y_df.iloc[:, 0].values
    else:
        gt_raw = y_df.values.ravel()
    gt_codes = pd.Categorical(gt_raw).codes.astype(np.int64)
    mask = gt_codes >= 0
    X, gt_codes = (X[mask], gt_codes[mask])
    _save(name, X, gt_codes, {'source': f'UCI ML Repository id={dataset_id}', 'description': description}, force)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    tasks = [('Golub Leukemia CSV', lambda: dl_golub(force)), ('Golub Leukemia GEO', lambda: dl_golub_geo(force)), ('Prostate Singh 2002', lambda: dl_prostate(force)), ('Breast cancer PAM50', lambda: dl_breast_geo(force)), ('Ovarian cancer', lambda: dl_ovarian(force)), ('Bladder cancer', lambda: dl_bladder(force)), ('WDBC Breast Cancer', lambda: dl_wdbc(force)), ('Dermatology', lambda: dl_uci('dermatology', 33, 'Dermatology: 6 skin diseases, 34 clinical features', force)), ('Hepatitis', lambda: dl_uci('hepatitis', 46, 'Hepatitis: survive vs die, 19 clinical features', force)), ('Lymphography', lambda: dl_uci('lymphography', 63, 'Lymphography: 4 node classes, 18 features', force)), ('Heart Disease', lambda: dl_uci('heart_disease', 45, 'Heart Disease Cleveland: 0=absent 1-4=present, 13 features', force)), ('Ionosphere', lambda: dl_uci('ionosphere', 52, 'Ionosphere: radar good vs bad, 34 features', force)), ('Glass Identification', lambda: dl_uci('glass', 42, 'Glass: 6 types by oxide composition, 9 features', force)), ('Thyroid Disease', lambda: dl_uci('thyroid', 102, 'Thyroid: hypothyroid vs normal, clinical features', force)), ('Pima Diabetes', lambda: dl_uci('pima_diabetes', 34, 'Pima Indians diabetes type 2, 8 clinical features', force))]
    print(f'Downloading {len(tasks)} biomedical datasets -> {OUT_DIR}')
    print('=' * 65)
    ok, skip, fail = (0, 0, 0)
    for label, fn in tasks:
        try:
            fn()
            ok += 1
        except Exception as exc:
            print(f'  ERROR [{label}]: {exc}')
            fail += 1
    print()
    print('=' * 65)
    saved = sorted(OUT_DIR.glob('*.npz'))
    print(f'Files in {OUT_DIR} ({len(saved)} total):')
    for p in saved:
        d = np.load(p, allow_pickle=True)
        n = d['gt'].size
        k = len(np.unique(d['gt']))
        dim = d['X'].shape[1]
        print(f'  {p.stem:32s}  n={n:4d}  d={dim:5d}  K={k}')
if __name__ == '__main__':
    main()
