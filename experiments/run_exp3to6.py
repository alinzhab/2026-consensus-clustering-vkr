"""Run experiments 3-6 for the report."""
from __future__ import annotations
import csv, sys, time, traceback, warnings
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'consensus_lab'))

from algorithms_base import AlgorithmRegistry

warnings.filterwarnings('ignore')

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
RUNS = 5
M_DEFAULT = 20
LINKAGES = ['single', 'complete', 'average', 'ward']
M_VALUES = [5, 10, 15, 20, 30, 40, 50]
REAL_DATASETS = ['Aggregation', 'BBC', 'Ecoli', 'GLIOMA', 'Lung', 'orlraws10P']


def find_dataset(name):
    for d in [ROOT / 'datasets', ROOT / 'datasets' / 'real']:
        for ext in ['.mat', '.npz']:
            p = d / (name + ext)
            if p.exists():
                return p
    raise FileNotFoundError(f'Dataset not found: {name}')


def dataset_info(path):
    if path.suffix == '.mat':
        m = scipy.io.loadmat(str(path))
        gt = m.get('gt', m.get('label', m.get('y'))).ravel().astype(np.int64)
        X_raw = None
        for k in ('X', 'x', 'fea', 'data', 'features'):
            if k in m:
                X_raw = m[k]
                break
        d = int(X_raw.shape[1]) if X_raw is not None else -1
    else:
        data = np.load(path, allow_pickle=True)
        gt = data['gt'].ravel().astype(np.int64)
        d = int(data['X'].shape[1]) if 'X' in data.files else -1
    return {'n': int(gt.size), 'k': int(np.unique(gt).size), 'd': d}


def run_algo(algo_name, path, method='average', m=M_DEFAULT, runs=RUNS, seed=SEED,
             selection_strategy='random', qd_alpha=0.5):
    algo = AlgorithmRegistry.get(algo_name)
    res = algo.run(dataset_path=str(path), m=m, runs=runs, method=method, seed=seed,
                   selection_strategy=selection_strategy, qd_alpha=qd_alpha)
    return res.nmi_mean, res.nmi_std, res.ari_mean, res.ari_std, res.f_mean, res.f_std


def _get_pool_size(path):
    if path.suffix == '.mat':
        m = scipy.io.loadmat(str(path))
        return int(m['members'].shape[1])
    data = np.load(path, allow_pickle=True)
    return int(data['members'].shape[1])


# ─── EXP 3: m effect ─────────────────────────────────────────────────────────
def exp3_m_effect():
    out_path = RESULTS_DIR / 'report_exp3_m_effect.tsv'
    print('\n' + '='*60)
    print('EXP 3: Влияние числа базовых разбиений m')
    print('='*60)
    TEST_DS = ['Ecoli', 'Aggregation']
    header = ['dataset', 'algorithm', 'm', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std']
    rows = []
    for ds in TEST_DS:
        try:
            path = find_dataset(ds)
            pool_size = _get_pool_size(path)
            print(f'\n[{ds}] pool_size={pool_size}')
            for algo in ['hierarchical_baseline', 'sdgca']:
                for m in M_VALUES:
                    if m > pool_size:
                        continue
                    try:
                        nmi, nmi_std, ari, ari_std, _, _ = run_algo(
                            algo, path, method='average', m=m, runs=RUNS)
                        rows.append({'dataset': ds, 'algorithm': algo, 'm': m,
                                     'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                                     'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5)})
                        print(f'  {algo} m={m}: NMI={nmi:.4f}')
                    except Exception as e:
                        print(f'  {algo} m={m}: ERROR {e}')
        except Exception as e:
            print(f'  ERROR {ds}: {e}')
            traceback.print_exc()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {out_path}')
    return rows


# ─── EXP 4: QD selection ─────────────────────────────────────────────────────
def exp4_qd_selection():
    out_path = RESULTS_DIR / 'report_exp4_qd.tsv'
    print('\n' + '='*60)
    print('EXP 4: QD-отбор vs случайный отбор')
    print('='*60)
    header = ['dataset', 'algorithm', 'selection', 'qd_alpha',
              'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean']
    rows = []
    for ds in REAL_DATASETS:
        try:
            path = find_dataset(ds)
            print(f'\n[{ds}]')
            for algo in ['hierarchical_baseline', 'sdgca']:
                try:
                    nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                        algo, path, method='average', m=M_DEFAULT, runs=RUNS,
                        selection_strategy='random')
                    rows.append({'dataset': ds, 'algorithm': algo,
                                 'selection': 'random', 'qd_alpha': '-',
                                 'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                                 'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5),
                                 'f_mean': round(f, 5)})
                    print(f'  {algo} [random]: NMI={nmi:.4f}')
                except Exception as e:
                    print(f'  {algo} [random]: ERROR {e}')
                for alpha in [0.25, 0.5, 0.75]:
                    try:
                        nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                            algo, path, method='average', m=M_DEFAULT, runs=RUNS,
                            selection_strategy='qd', qd_alpha=alpha)
                        rows.append({'dataset': ds, 'algorithm': algo,
                                     'selection': 'qd', 'qd_alpha': alpha,
                                     'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                                     'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5),
                                     'f_mean': round(f, 5)})
                        print(f'  {algo} [qd a={alpha}]: NMI={nmi:.4f}')
                    except Exception as e:
                        print(f'  {algo} [qd a={alpha}]: ERROR {e}')
        except Exception as e:
            print(f'  ERROR {ds}: {e}')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {out_path}')
    return rows


# ─── EXP 5: Synthetic datasets ───────────────────────────────────────────────
def exp5_synthetic():
    out_path = RESULTS_DIR / 'report_exp5_synthetic.tsv'
    print('\n' + '='*60)
    print('EXP 5: Синтетические датасеты')
    print('='*60)
    patterns = [
        ('compact',    ['design_compact_easy_5k', 'design_mini_compact', 'analysis_simple_separated']),
        ('overlap',    ['design_overlap_moderate', 'design_mini_overlap', 'analysis_simple_overlap']),
        ('imbalanced', ['design_imbalanced_6x', 'design_mini_imbalanced', 'analysis_imbalanced']),
        ('highdim',    ['design_highdim_20d', 'design_mini_highdim', 'analysis_highdim']),
        ('elongated',  ['design_elongated_2d', 'design_mini_elongated', 'analysis_densired_stretched']),
        ('density',    ['design_density_varied_noisy', 'design_mini_density_varied',
                        'analysis_repliclust_heterogeneous']),
    ]
    header = ['scenario', 'dataset', 'algorithm', 'nmi_mean', 'nmi_std', 'ari_mean', 'f_mean']
    rows = []
    for scenario, candidates in patterns:
        found_path = None
        found_name = None
        for name in candidates:
            try:
                found_path = find_dataset(name)
                found_name = name
                break
            except FileNotFoundError:
                continue
        if found_path is None:
            print(f'  Scenario {scenario}: not found')
            continue
        try:
            info = dataset_info(found_path)
            print(f'\n[{scenario}] {found_name} n={info["n"]} k={info["k"]}')
            for algo in ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified']:
                try:
                    m = min(M_DEFAULT, _get_pool_size(found_path))
                    nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                        algo, found_path, method='average', m=m, runs=RUNS)
                    rows.append({'scenario': scenario, 'dataset': found_name,
                                 'algorithm': algo,
                                 'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                                 'ari_mean': round(ari, 5), 'f_mean': round(f, 5)})
                    print(f'  {algo}: NMI={nmi:.4f}')
                except Exception as e:
                    print(f'  {algo}: ERROR {e}')
        except Exception as e:
            print(f'  ERROR {scenario}: {e}')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {out_path}')
    return rows


# ─── EXP 6: Timing ───────────────────────────────────────────────────────────
def exp6_timing():
    out_path = RESULTS_DIR / 'report_exp6_timing.tsv'
    print('\n' + '='*60)
    print('EXP 6: Время выполнения')
    print('='*60)
    ALGOS = ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified']
    header = ['dataset', 'n', 'algorithm', 'time_sec']
    rows = []
    for ds in REAL_DATASETS:
        try:
            path = find_dataset(ds)
            info = dataset_info(path)
            print(f'\n[{ds}] n={info["n"]}')
            for algo in ALGOS:
                t0 = time.time()
                try:
                    run_algo(algo, path, method='average', m=M_DEFAULT, runs=3)
                    elapsed = round(time.time() - t0, 2)
                    rows.append({'dataset': ds, 'n': info['n'],
                                 'algorithm': algo, 'time_sec': elapsed})
                    print(f'  {algo}: {elapsed:.2f}s')
                except Exception as e:
                    print(f'  {algo}: ERROR {e}')
        except Exception as e:
            print(f'  ERROR {ds}: {e}')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {out_path}')
    return rows


if __name__ == '__main__':
    t_start = time.time()
    print('Running experiments 3-6...')
    exp3_m_effect()
    exp4_qd_selection()
    exp5_synthetic()
    exp6_timing()
    total = (time.time() - t_start) / 60
    print(f'\nAll done in {total:.1f} min')
