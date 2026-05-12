"""
Полный набор экспериментов для отчёта.
Сохраняет результаты в results/report_*.tsv

Запуск:
    python experiments/run_full_report_experiments.py
"""
from __future__ import annotations
import csv
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'consensus_lab'))

from algorithms_base import AlgorithmRegistry
from metrics import compute_nmi, compute_ari, compute_pairwise_f_score

warnings.filterwarnings('ignore')

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ─── настройки ──────────────────────────────────────────────────────────────
SEED = 42
RUNS = 5
M_DEFAULT = 20
LINKAGES = ['single', 'complete', 'average', 'ward']
M_VALUES = [5, 10, 15, 20, 30, 40, 50]
REAL_DATASETS = ['Aggregation', 'BBC', 'Ecoli', 'GLIOMA', 'Lung', 'orlraws10P']
# ────────────────────────────────────────────────────────────────────────────


def find_dataset(name: str) -> Path:
    for d in [ROOT / 'datasets', ROOT / 'datasets' / 'real']:
        for ext in ['.mat', '.npz']:
            p = d / (name + ext)
            if p.exists():
                return p
    raise FileNotFoundError(f'Dataset not found: {name}')


def dataset_info(path: Path) -> dict:
    """Return basic n/k/d info."""
    if path.suffix == '.mat':
        m = scipy.io.loadmat(str(path))
        gt = m.get('gt', m.get('label', m.get('y'))).ravel().astype(np.int64)
        X_raw = None
        for k in ('X', 'x', 'fea', 'data', 'features'):
            if k in m:
                X_raw = m[k]; break
        d = int(X_raw.shape[1]) if X_raw is not None else -1
    else:
        data = np.load(path, allow_pickle=True)
        gt = data['gt'].ravel().astype(np.int64)
        d = int(data['X'].shape[1]) if 'X' in data.files else -1
    return {'n': int(gt.size), 'k': int(np.unique(gt).size), 'd': d}


def run_algo(algo_name: str, path: Path, method: str = 'average',
             m: int = M_DEFAULT, runs: int = RUNS, seed: int = SEED,
             selection_strategy: str = 'random', qd_alpha: float = 0.5) -> tuple[float, float, float, float, float, float]:
    """Returns (nmi_mean, nmi_std, ari_mean, ari_std, f_mean, f_std)."""
    algo = AlgorithmRegistry.get(algo_name)
    res = algo.run(
        dataset_path=str(path), m=m, runs=runs,
        method=method, seed=seed,
        selection_strategy=selection_strategy,
        qd_alpha=qd_alpha,
    )
    return res.nmi_mean, res.nmi_std, res.ari_mean, res.ari_std, res.f_mean, res.f_std


# ═══════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Все алгоритмы × все датасеты × лучшее правило объединения
# ═══════════════════════════════════════════════════════════════════════════
def exp1_main_table():
    out_path = RESULTS_DIR / 'report_exp1_main.tsv'
    print('\n' + '='*60)
    print('EXP 1: Основная таблица (все алгоритмы × все датасеты)')
    print('='*60)

    ALGOS = ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified']
    header = ['dataset', 'n', 'k', 'd',
              'hier_base_nmi', 'hier_base_nmi_std',
              'hier_base_ari', 'hier_base_ari_std',
              'hier_base_f',
              'hier_wtd_nmi', 'hier_wtd_nmi_std',
              'hier_wtd_ari', 'hier_wtd_ari_std',
              'hier_wtd_f',
              'sdgca_nmi', 'sdgca_nmi_std',
              'sdgca_ari', 'sdgca_ari_std',
              'sdgca_f',
              'sdgca_mod_nmi', 'sdgca_mod_nmi_std',
              'sdgca_mod_ari', 'sdgca_mod_ari_std',
              'sdgca_mod_f',
              'best_linkage_hier', 'best_linkage_sdgca']

    rows = []
    for ds in REAL_DATASETS:
        try:
            path = find_dataset(ds)
            info = dataset_info(path)
            print(f'\n[{ds}] n={info["n"]} k={info["k"]} d={info["d"]}')
            row = {'dataset': ds, **info}

            # Для каждого алгоритма перебираем все правила и берём лучшее по NMI
            for algo_name, prefix in [
                ('hierarchical_baseline', 'hier_base'),
                ('hierarchical_weighted', 'hier_wtd'),
                ('sdgca', 'sdgca'),
                ('sdgca_modified', 'sdgca_mod'),
            ]:
                best_nmi, best_std_nmi = -1.0, 0.0
                best_ari, best_std_ari = -1.0, 0.0
                best_f = -1.0
                best_lnk = 'average'
                for lnk in LINKAGES:
                    try:
                        nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                            algo_name, path, method=lnk, m=M_DEFAULT, runs=RUNS)
                        if nmi > best_nmi:
                            best_nmi, best_std_nmi = nmi, nmi_std
                            best_ari, best_std_ari = ari, ari_std
                            best_f = f
                            best_lnk = lnk
                        print(f'  {algo_name} [{lnk}]: NMI={nmi:.4f}')
                    except Exception as e:
                        print(f'  {algo_name} [{lnk}]: ERROR {e}')

                row[f'{prefix}_nmi'] = round(best_nmi, 5)
                row[f'{prefix}_nmi_std'] = round(best_std_nmi, 5)
                row[f'{prefix}_ari'] = round(best_ari, 5)
                row[f'{prefix}_ari_std'] = round(best_std_ari, 5)
                row[f'{prefix}_f'] = round(best_f, 5)
                if 'hier' in prefix:
                    row['best_linkage_hier'] = best_lnk
                else:
                    row['best_linkage_sdgca'] = best_lnk

            rows.append(row)
            print(f'  -> hier_base={row["hier_base_nmi"]:.4f} '
                  f'hier_wtd={row["hier_wtd_nmi"]:.4f} '
                  f'sdgca={row["sdgca_nmi"]:.4f} '
                  f'sdgca_mod={row["sdgca_mod_nmi"]:.4f}')
        except Exception as e:
            print(f'  ERROR on {ds}: {e}')
            traceback.print_exc()

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nСохранено: {out_path}')
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 2: Влияние правила межкластерного объединения
# ═══════════════════════════════════════════════════════════════════════════
def exp2_linkage_effect():
    out_path = RESULTS_DIR / 'report_exp2_linkage.tsv'
    print('\n' + '='*60)
    print('EXP 2: Влияние правила объединения')
    print('='*60)

    ALGOS = ['hierarchical_baseline', 'sdgca']
    header = ['dataset', 'algorithm', 'linkage',
              'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean']
    rows = []

    for ds in REAL_DATASETS:
        try:
            path = find_dataset(ds)
            print(f'\n[{ds}]')
            for algo in ALGOS:
                for lnk in LINKAGES:
                    try:
                        nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                            algo, path, method=lnk, m=M_DEFAULT, runs=RUNS)
                        rows.append({
                            'dataset': ds, 'algorithm': algo, 'linkage': lnk,
                            'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                            'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5),
                            'f_mean': round(f, 5),
                        })
                        print(f'  {algo} [{lnk}]: NMI={nmi:.4f}±{nmi_std:.4f}')
                    except Exception as e:
                        print(f'  {algo} [{lnk}]: ERROR {e}')
        except Exception as e:
            print(f'  ERROR on {ds}: {e}')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nСохранено: {out_path}')
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 3: Влияние числа базовых разбиений m
# ═══════════════════════════════════════════════════════════════════════════
def exp3_m_effect():
    out_path = RESULTS_DIR / 'report_exp3_m_effect.tsv'
    print('\n' + '='*60)
    print('EXP 3: Влияние числа базовых разбиений m')
    print('='*60)

    # Используем два представительных датасета
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
                        rows.append({
                            'dataset': ds, 'algorithm': algo, 'm': m,
                            'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                            'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5),
                        })
                        print(f'  {algo} m={m}: NMI={nmi:.4f}±{nmi_std:.4f}')
                    except Exception as e:
                        print(f'  {algo} m={m}: ERROR {e}')
        except Exception as e:
            print(f'  ERROR on {ds}: {e}')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nСохранено: {out_path}')
    return rows


def _get_pool_size(path: Path) -> int:
    if path.suffix == '.mat':
        m = scipy.io.loadmat(str(path))
        return int(m['members'].shape[1])
    data = np.load(path, allow_pickle=True)
    return int(data['members'].shape[1])


# ═══════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 4: QD-отбор vs случайный отбор
# ═══════════════════════════════════════════════════════════════════════════
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
                # Случайный отбор
                try:
                    nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                        algo, path, method='average', m=M_DEFAULT, runs=RUNS,
                        selection_strategy='random')
                    rows.append({
                        'dataset': ds, 'algorithm': algo,
                        'selection': 'random', 'qd_alpha': '-',
                        'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                        'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5),
                        'f_mean': round(f, 5),
                    })
                    print(f'  {algo} [random]: NMI={nmi:.4f}±{nmi_std:.4f}')
                except Exception as e:
                    print(f'  {algo} [random]: ERROR {e}')

                # QD-отбор с разными alpha
                for alpha in [0.25, 0.5, 0.75]:
                    try:
                        nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                            algo, path, method='average', m=M_DEFAULT, runs=RUNS,
                            selection_strategy='qd', qd_alpha=alpha)
                        rows.append({
                            'dataset': ds, 'algorithm': algo,
                            'selection': 'qd', 'qd_alpha': alpha,
                            'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                            'ari_mean': round(ari, 5), 'ari_std': round(ari_std, 5),
                            'f_mean': round(f, 5),
                        })
                        print(f'  {algo} [qd α={alpha}]: NMI={nmi:.4f}±{nmi_std:.4f}')
                    except Exception as e:
                        print(f'  {algo} [qd α={alpha}]: ERROR {e}')
        except Exception as e:
            print(f'  ERROR on {ds}: {e}')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nСохранено: {out_path}')
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 5: Синтетические датасеты
# ═══════════════════════════════════════════════════════════════════════════
def exp5_synthetic():
    out_path = RESULTS_DIR / 'report_exp5_synthetic.tsv'
    print('\n' + '='*60)
    print('EXP 5: Синтетические датасеты')
    print('='*60)

    # Ищем синтетические датасеты по паттернам имён
    patterns = [
        ('compact',      ['design_compact_easy_5k', 'design_mini_compact', 'analysis_simple_separated']),
        ('overlap',      ['design_overlap_moderate', 'design_mini_overlap', 'analysis_simple_overlap']),
        ('imbalanced',   ['design_imbalanced_6x', 'design_mini_imbalanced', 'analysis_imbalanced']),
        ('highdim',      ['design_highdim_20d', 'design_mini_highdim', 'analysis_highdim']),
        ('elongated',    ['design_elongated_2d', 'design_mini_elongated', 'analysis_densired_stretched']),
        ('density',      ['design_density_varied_noisy', 'design_mini_density_varied', 'analysis_repliclust_heterogeneous']),
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
            print(f'  Сценарий {scenario}: датасет не найден')
            continue

        try:
            info = dataset_info(found_path)
            print(f'\n[{scenario}] {found_name} n={info["n"]} k={info["k"]} d={info["d"]}')
            for algo in ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified']:
                try:
                    nmi, nmi_std, ari, ari_std, f, _ = run_algo(
                        algo, found_path, method='average', m=min(M_DEFAULT, _get_pool_size(found_path)),
                        runs=RUNS)
                    rows.append({
                        'scenario': scenario, 'dataset': found_name,
                        'algorithm': algo,
                        'nmi_mean': round(nmi, 5), 'nmi_std': round(nmi_std, 5),
                        'ari_mean': round(ari, 5), 'f_mean': round(f, 5),
                    })
                    print(f'  {algo}: NMI={nmi:.4f}±{nmi_std:.4f}')
                except Exception as e:
                    print(f'  {algo}: ERROR {e}')
        except Exception as e:
            print(f'  ERROR {scenario}: {e}')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nСохранено: {out_path}')
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 6: Время выполнения
# ═══════════════════════════════════════════════════════════════════════════
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
                    rows.append({
                        'dataset': ds, 'n': info['n'],
                        'algorithm': algo, 'time_sec': elapsed,
                    })
                    print(f'  {algo}: {elapsed:.2f}s')
                except Exception as e:
                    print(f'  {algo}: ERROR {e}')
        except Exception as e:
            print(f'  ERROR on {ds}: {e}')

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nСохранено: {out_path}')
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()
    print('Запуск полного набора экспериментов для отчёта...')
    print(f'Параметры: seed={SEED}, runs={RUNS}, m={M_DEFAULT}')
    print(f'Результаты будут сохранены в: {RESULTS_DIR}')

    results = {}

    results['exp1'] = exp1_main_table()
    results['exp2'] = exp2_linkage_effect()
    results['exp3'] = exp3_m_effect()
    results['exp4'] = exp4_qd_selection()
    results['exp5'] = exp5_synthetic()
    results['exp6'] = exp6_timing()

    # Сохранить сводный JSON
    summary_path = RESULTS_DIR / 'report_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({k: len(v) for k, v in results.items()}, f, indent=2)

    total = (time.time() - t_start) / 60
    print(f'\n{"="*60}')
    print(f'Все эксперименты завершены за {total:.1f} мин.')
    print(f'Файлы сохранены в {RESULTS_DIR}')
    print('  report_exp1_main.tsv       — основная таблица')
    print('  report_exp2_linkage.tsv    — влияние правила объединения')
    print('  report_exp3_m_effect.tsv   — влияние числа m')
    print('  report_exp4_qd.tsv         — QD-отбор vs случайный')
    print('  report_exp5_synthetic.tsv  — синтетические данные')
    print('  report_exp6_timing.tsv     — время выполнения')
