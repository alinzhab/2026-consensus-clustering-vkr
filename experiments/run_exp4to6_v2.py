"""Быстрый прогон exp4-6 с датасет-специфичными параметрами SDGCA."""
from __future__ import annotations
import csv, sys, time, warnings
from pathlib import Path
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'consensus_lab'))
from algorithms_base import AlgorithmRegistry
from sdgca_modified import DEFAULT_PARAMS as DP
warnings.filterwarnings('ignore')

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
RUNS = 2
M_DEFAULT = 20
REAL_DATASETS = ['Aggregation', 'BBC', 'Ecoli', 'GLIOMA', 'Lung', 'orlraws10P']


def find_dataset(name):
    for d in [ROOT / 'datasets', ROOT / 'datasets' / 'real']:
        for ext in ['.mat', '.npz']:
            p = d / (name + ext)
            if p.exists():
                return p
    raise FileNotFoundError(name)


def dataset_info(path):
    if path.suffix == '.mat':
        m = scipy.io.loadmat(str(path))
        gt = m.get('gt', m.get('label', m.get('y'))).ravel().astype(np.int64)
    else:
        data = np.load(path, allow_pickle=True)
        gt = data['gt'].ravel().astype(np.int64)
    return {'n': int(gt.size), 'k': int(np.unique(gt).size)}


def _get_pool_size(path):
    if path.suffix == '.mat':
        m = scipy.io.loadmat(str(path))
        return int(m['members'].shape[1])
    data = np.load(path, allow_pickle=True)
    return int(data['members'].shape[1])


def sdgca_kwargs(ds_name):
    """Возвращает датасет-специфичные параметры SDGCA."""
    p = DP.get(ds_name, {'lambda_': 0.09, 'eta': 0.75, 'theta': 0.65})
    return {'nwca_para': p['lambda_'], 'eta': p['eta'], 'theta': p['theta']}


def run_algo(algo_name, ds_name, path, method='average', m=M_DEFAULT, runs=RUNS,
             seed=SEED, selection_strategy='random', qd_alpha=0.5):
    algo = AlgorithmRegistry.get(algo_name)
    extra = {}
    if 'sdgca' in algo_name:
        extra = sdgca_kwargs(ds_name)
    res = algo.run(dataset_path=str(path), m=m, runs=runs, method=method, seed=seed,
                   selection_strategy=selection_strategy, qd_alpha=qd_alpha, **extra)
    return res.nmi_mean, res.nmi_std, res.ari_mean, res.ari_std, res.f_mean


def exp4_qd():
    out_path = RESULTS_DIR / 'report_exp4_qd.tsv'
    print('EXP4: QD selection', flush=True)
    header = ['dataset', 'algorithm', 'selection', 'qd_alpha',
              'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean']
    rows = []
    for ds in REAL_DATASETS:
        path = find_dataset(ds)
        print(f'  [{ds}]', flush=True)
        for algo in ['hierarchical_baseline', 'sdgca']:
            for strat, alpha in [('random', '-'), ('qd', 0.25), ('qd', 0.5), ('qd', 0.75)]:
                a = alpha if isinstance(alpha, float) else 0.5
                try:
                    t0 = time.time()
                    nmi, nmi_std, ari, ari_std, f = run_algo(
                        algo, ds, path, method='average', m=M_DEFAULT, runs=RUNS,
                        selection_strategy=strat, qd_alpha=a)
                    rows.append({'dataset': ds, 'algorithm': algo,
                                 'selection': strat, 'qd_alpha': alpha,
                                 'nmi_mean': round(nmi,5), 'nmi_std': round(nmi_std,5),
                                 'ari_mean': round(ari,5), 'ari_std': round(ari_std,5),
                                 'f_mean': round(f,5)})
                    print(f'    {algo} {strat}/{alpha}: NMI={nmi:.4f} ({time.time()-t0:.1f}s)', flush=True)
                except Exception as e:
                    print(f'    {algo} {strat}/{alpha}: ERR {e}', flush=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    print(f'  Saved {out_path}', flush=True)


def exp5_synthetic():
    out_path = RESULTS_DIR / 'report_exp5_synthetic.tsv'
    print('EXP5: synthetic', flush=True)
    patterns = [
        ('compact',    ['design_compact_easy_5k','design_mini_compact','analysis_simple_separated']),
        ('overlap',    ['design_overlap_moderate','design_mini_overlap','analysis_simple_overlap']),
        ('imbalanced', ['design_imbalanced_6x','design_mini_imbalanced','analysis_imbalanced']),
        ('highdim',    ['design_highdim_20d','design_mini_highdim','analysis_highdim']),
        ('elongated',  ['design_elongated_2d','design_mini_elongated','analysis_densired_stretched']),
        ('density',    ['design_density_varied_noisy','design_mini_density_varied',
                        'analysis_repliclust_heterogeneous']),
    ]
    header = ['scenario','dataset','algorithm','nmi_mean','nmi_std','ari_mean','f_mean']
    rows = []
    for scenario, cands in patterns:
        path = None; nm = None
        for c in cands:
            try: path = find_dataset(c); nm = c; break
            except FileNotFoundError: pass
        if path is None:
            print(f'  {scenario}: not found', flush=True); continue
        info = dataset_info(path)
        print(f'  [{scenario}] {nm} n={info["n"]} k={info["k"]}', flush=True)
        m = min(M_DEFAULT, _get_pool_size(path))
        for algo in ['hierarchical_baseline','hierarchical_weighted','sdgca','sdgca_modified']:
            try:
                t0 = time.time()
                nmi, nmi_std, ari, _, f = run_algo(algo, nm, path, method='average', m=m, runs=RUNS)
                rows.append({'scenario': scenario, 'dataset': nm, 'algorithm': algo,
                             'nmi_mean': round(nmi,5), 'nmi_std': round(nmi_std,5),
                             'ari_mean': round(ari,5), 'f_mean': round(f,5)})
                print(f'    {algo}: NMI={nmi:.4f} ({time.time()-t0:.1f}s)', flush=True)
            except Exception as e:
                print(f'    {algo}: ERR {e}', flush=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    print(f'  Saved {out_path}', flush=True)


def exp6_timing():
    out_path = RESULTS_DIR / 'report_exp6_timing.tsv'
    print('EXP6: timing', flush=True)
    header = ['dataset','n','algorithm','time_sec']
    rows = []
    for ds in REAL_DATASETS:
        path = find_dataset(ds)
        info = dataset_info(path)
        print(f'  [{ds}] n={info["n"]}', flush=True)
        for algo in ['hierarchical_baseline','hierarchical_weighted','sdgca','sdgca_modified']:
            t0 = time.time()
            try:
                run_algo(algo, ds, path, method='average', m=M_DEFAULT, runs=1)
                el = round(time.time()-t0, 2)
                rows.append({'dataset': ds, 'n': info['n'], 'algorithm': algo, 'time_sec': el})
                print(f'    {algo}: {el}s', flush=True)
            except Exception as e:
                print(f'    {algo}: ERR {e}', flush=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    print(f'  Saved {out_path}', flush=True)


if __name__ == '__main__':
    t0 = time.time()
    exp4_qd()
    exp5_synthetic()
    exp6_timing()
    print(f'Готово за {(time.time()-t0)/60:.1f} мин', flush=True)
