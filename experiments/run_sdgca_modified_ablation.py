from __future__ import annotations
import argparse
import csv
import math
import shutil
import sys
import time
import traceback
import tracemalloc
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSENSUS_LAB = PROJECT_ROOT / 'consensus_lab'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'
sys.path.insert(0, str(CONSENSUS_LAB))
from ensemble_selection import DATASET_TYPE_LABELS, get_dataset_type
from sdgca import run_sdgca
from sdgca_modified import DEFAULT_PARAMS, run_sdgca_modified
GENERIC = {'nwca_para': 0.09, 'eta': 0.75, 'theta': 0.65}
VARIANTS: list[dict] = [{'variant': 'A_baseline_random', 'algorithm': 'sdgca', 'diffusion_time': None, 'selection_strategy': 'random', 'use_tuned_params': False, 'label': 'A: SDGCA baseline, random'}, {'variant': 'B_modified_fixed_random', 'algorithm': 'sdgca_modified', 'diffusion_time': 1.0, 'selection_strategy': 'random', 'use_tuned_params': False, 'label': 'B: SDGCA modified fixed t=1.0, random'}, {'variant': 'C_modified_adaptive_random', 'algorithm': 'sdgca_modified', 'diffusion_time': None, 'selection_strategy': 'random', 'use_tuned_params': False, 'label': 'C: SDGCA modified adaptive t, random'}, {'variant': 'D_baseline_qd', 'algorithm': 'sdgca', 'diffusion_time': None, 'selection_strategy': 'qd', 'use_tuned_params': False, 'label': 'D: SDGCA baseline, QD'}, {'variant': 'E_modified_fixed_qd', 'algorithm': 'sdgca_modified', 'diffusion_time': 1.0, 'selection_strategy': 'qd', 'use_tuned_params': False, 'label': 'E: SDGCA modified fixed t=1.0, QD'}, {'variant': 'F_modified_adaptive_qd', 'algorithm': 'sdgca_modified', 'diffusion_time': None, 'selection_strategy': 'qd', 'use_tuned_params': False, 'label': 'F: SDGCA modified adaptive t, QD'}]
TUNED_VARIANTS: list[dict] = [{'variant': 'G_modified_tuned_random', 'algorithm': 'sdgca_modified', 'diffusion_time': 'from_default_params', 'selection_strategy': 'random', 'use_tuned_params': True, 'label': 'G: SDGCA modified tuned (DEFAULT_PARAMS), random'}, {'variant': 'H_modified_tuned_qd', 'algorithm': 'sdgca_modified', 'diffusion_time': 'from_default_params', 'selection_strategy': 'qd', 'use_tuned_params': True, 'label': 'H: SDGCA modified tuned (DEFAULT_PARAMS), QD'}]
SMOKE_DATASETS = [('design_mini_compact', '.npz'), ('design_mini_overlap', '.npz'), ('Ecoli', '.mat')]
MINI_DATASETS = [('design_mini_compact', '.npz'), ('design_mini_overlap', '.npz'), ('design_mini_imbalanced', '.npz'), ('design_mini_highdim', '.npz'), ('design_mini_elongated', '.npz'), ('design_mini_density_varied', '.npz'), ('design_mini_mixed_complex', '.npz'), ('Ecoli', '.mat'), ('GLIOMA', '.mat'), ('Lung', '.mat')]
SCALING_DATASETS = [('design_mini_compact', '.npz'), ('design_mini_overlap', '.npz'), ('design_mini_elongated', '.npz'), ('design_overlap_moderate', '.npz'), ('analysis_densired_compact', '.npz'), ('analysis_densired_stretched', '.npz')]
FULL_DATASETS = MINI_DATASETS + [('design_compact_easy_5k', '.npz'), ('design_overlap_moderate', '.npz'), ('design_elongated_2d', '.npz'), ('design_density_varied_low_noise', '.npz'), ('design_imbalanced_6x', '.npz'), ('Aggregation', '.mat')]
TSV_COLUMNS = ['experiment', 'created_at', 'dataset', 'dataset_type', 'dataset_type_label', 'variant', 'algorithm', 'selection_strategy', 'diffusion_mode', 'linkage', 'm', 'runs', 'seed', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean', 'f_std', 'runtime_sec', 'memory_peak_mb', 'diffusion_time_used', 'n_objects', 'n_members', 'n_clusters_ensemble']

def find_dataset(name: str, suffix: str) -> Path | None:
    for folder in (DATASETS_DIR, DATASETS_DIR / 'uploaded'):
        path = folder / f'{name}{suffix}'
        if path.exists():
            return path
    return None

def dataset_shape(path: Path) -> tuple[int, int]:
    from scipy.io import loadmat
    if path.suffix.lower() == '.mat':
        data = loadmat(path)
        arr = np.asarray(data['members'])
    else:
        data = np.load(path, allow_pickle=True)
        arr = np.asarray(data['members'])
    return (int(arr.shape[0]), int(arr.shape[1]))

def run_variant(path: Path, dataset_name: str, variant_cfg: dict, linkage: str, m: int, runs: int, seed: int, qd_alpha: float, created_at: str) -> dict | None:
    try:
        n_objects, n_members = dataset_shape(path)
    except Exception as exc:
        print(f'  [ERR shape] {exc}')
        return None
    variant = variant_cfg['variant']
    algorithm = variant_cfg['algorithm']
    selection_strategy = variant_cfg['selection_strategy']
    use_tuned = variant_cfg.get('use_tuned_params', False)
    if use_tuned:
        dp = DEFAULT_PARAMS.get(dataset_name, {'lambda_': 0.09, 'eta': 0.75, 'theta': 0.65, 'diffusion_time': 1.0})
        kwargs_extra = {'nwca_para': dp['lambda_'], 'eta': dp['eta'], 'theta': dp['theta'], 'diffusion_time': dp['diffusion_time']}
        diffusion_mode = 'tuned'
    else:
        kwargs_extra = dict(GENERIC)
        raw_dt = variant_cfg.get('diffusion_time')
        if algorithm == 'sdgca_modified':
            kwargs_extra['diffusion_time'] = raw_dt
            diffusion_mode = 'adaptive' if raw_dt is None else f'fixed_{raw_dt}'
        else:
            diffusion_mode = 'n/a'
    kwargs: dict = {'dataset_path': path, 'data_name': dataset_name, 'seed': seed, 'm': m, 'cnt_times': runs, 'method': linkage, 'selection_strategy': selection_strategy, 'qd_alpha': qd_alpha, **kwargs_extra}
    members = load_members_matrix(path)
    pool_size = n_members
    n_clusters_C = ensemble_total_clusters(members, pool_size, m, seed, selection_strategy, qd_alpha)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        if algorithm == 'sdgca':
            result = run_sdgca(**{k: v for k, v in kwargs.items() if k in ('dataset_path', 'data_name', 'seed', 'm', 'cnt_times', 'method', 'nwca_para', 'eta', 'theta', 'selection_strategy', 'qd_alpha')})
        else:
            result = run_sdgca_modified(**{k: v for k, v in kwargs.items() if k in ('dataset_path', 'data_name', 'seed', 'm', 'cnt_times', 'method', 'nwca_para', 'eta', 'theta', 'diffusion_time', 'selection_strategy', 'qd_alpha')})
    except Exception as exc:
        tracemalloc.stop()
        print(f'  [SKIP {variant}] {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()
        return None
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime_sec = time.perf_counter() - t0
    memory_peak_mb = peak_mem / (1024.0 * 1024.0)
    dtype = get_dataset_type(dataset_name)
    dt_used = result.get('diffusion_time_used', '')
    if dt_used is None:
        dt_used = ''
    return {'experiment': 'sdgca_ablation', 'created_at': created_at, 'dataset': dataset_name, 'dataset_type': dtype, 'dataset_type_label': DATASET_TYPE_LABELS.get(dtype, dtype), 'variant': variant, 'algorithm': algorithm, 'selection_strategy': selection_strategy, 'diffusion_mode': diffusion_mode, 'linkage': linkage, 'm': m, 'runs': runs, 'seed': seed, 'nmi_mean': round(float(result['nmi_mean']), 6), 'nmi_std': round(float(result['nmi_std']), 6), 'ari_mean': round(float(result['ari_mean']), 6), 'ari_std': round(float(result['ari_std']), 6), 'f_mean': round(float(result['f_mean']), 6), 'f_std': round(float(result['f_std']), 6), 'runtime_sec': round(runtime_sec, 3), 'memory_peak_mb': round(memory_peak_mb, 3), 'diffusion_time_used': dt_used, 'n_objects': n_objects, 'n_members': n_members, 'n_clusters_ensemble': n_clusters_C}

def append_rows(path: Path, rows: list[dict], write_header: bool=False) -> None:
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter='\t', extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float('nan')

def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return 'n/a'
    return f'{v:+.4f}' if abs(v) < 10 else f'{v:.2f}'

def _loglog_power_fit(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    xs = [math.log(n) for n, t in points if n > 0 and t > 0]
    ys = [math.log(t) for n, t in points if n > 0 and t > 0]
    k = len(xs)
    if k < 3:
        return None
    mx = sum(xs) / k
    my = sum(ys) / k
    sxx = sum(((x - mx) ** 2 for x in xs))
    if sxx < 1e-18:
        return None
    sxy = sum(((x - mx) * (y - my) for x, y in zip(xs, ys)))
    b = sxy / sxx
    log_a = my - b * mx
    a = math.exp(log_a)
    ss_tot = sum(((y - my) ** 2 for y in ys))
    y_pred = [log_a + b * x for x in xs]
    ss_res = sum(((y - yp) ** 2 for y, yp in zip(ys, y_pred)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    return (a, b, r2)

def write_summary_report(report_path: Path, rows: list[dict], args, tsv_path: Path) -> None:
    nmi_by_var_type: dict[tuple[str, str], list[float]] = defaultdict(list)
    runtime_by_var: dict[str, list[float]] = defaultdict(list)
    dt_used_by_var: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        var = row['variant']
        dtype = row['dataset_type']
        nmi_by_var_type[var, dtype].append(float(row['nmi_mean']))
        runtime_by_var[var].append(float(row['runtime_sec']))
        if row.get('diffusion_time_used') not in ('', None):
            try:
                dt_used_by_var[var].append(float(row['diffusion_time_used']))
            except (ValueError, TypeError):
                pass
    nmi_by_var_ds: dict[tuple[str, str, str, int], float] = {}
    for row in rows:
        key = (row['variant'], row['dataset'], row['linkage'], int(row['seed']))
        nmi_by_var_ds[key] = float(row['nmi_mean'])

    def delta_pairs(var_a: str, var_b: str) -> list[float]:
        deltas = []
        for (var, ds, lnk, sd), nmi in nmi_by_var_ds.items():
            if var == var_b:
                a_nmi = nmi_by_var_ds.get((var_a, ds, lnk, sd))
                if a_nmi is not None:
                    deltas.append(nmi - a_nmi)
        return deltas
    all_variants = sorted({r['variant'] for r in rows})
    all_types = sorted({r['dataset_type'] for r in rows})
    lines = ['# SDGCA Modified Ablation Study — Summary Report', '', f"Generated: {datetime.now().isoformat(timespec='seconds')}", f'TSV: `{tsv_path.name}`', f'Rows: {len(rows)}', '', '## Experimental parameters', '', f'- m: `{args.m}`', f'- runs per call: `{args.runs}`', f'- seeds: `{args.seeds}`', f'- linkage: `{args.methods}`', f'- qd_alpha: `{args.qd_alpha}`', f'- profile: `{args.profile}`', '', '## Variants', '', '| Variant | Label |', '|---|---|']
    all_variant_cfgs = VARIANTS + (TUNED_VARIANTS if args.include_tuned else [])
    for vc in all_variant_cfgs:
        lines.append(f"| `{vc['variant']}` | {vc['label']} |")
    lines += ['', '## Mean NMI by variant × dataset type', '']
    header = '| Variant | ' + ' | '.join(all_types) + ' | Mean all |'
    sep = '|' + '---|' * (len(all_types) + 2)
    lines.append(header)
    lines.append(sep)
    for var in all_variants:
        parts = [f'`{var}`']
        all_vals: list[float] = []
        for dtype in all_types:
            vals = nmi_by_var_type.get((var, dtype), [])
            if vals:
                v = _mean(vals)
                parts.append(f'{v:.4f}')
                all_vals.append(v)
            else:
                parts.append('—')
        parts.append(f'{_mean(all_vals):.4f}' if all_vals else '—')
        lines.append('| ' + ' | '.join(parts) + ' |')
    lines += ['', '## Delta NMI relative to baseline A (B−A, C−A)', '', 'Positive = modified wins, negative = baseline wins.', 'Threshold 0.002 used as non-trivial difference.', '', '| Comparison | Mean Δ NMI | N pairs |', '|---|---:|---:|']
    comparisons = [('B_modified_fixed_random', 'A_baseline_random', 'B − A (modifications, fixed t, random)'), ('C_modified_adaptive_random', 'A_baseline_random', 'C − A (modifications, adaptive t, random)'), ('C_modified_adaptive_random', 'B_modified_fixed_random', 'C − B (adaptive − fixed t)'), ('D_baseline_qd', 'A_baseline_random', 'D − A (QD effect on baseline)'), ('E_modified_fixed_qd', 'B_modified_fixed_random', 'E − B (QD effect on modified fixed)'), ('F_modified_adaptive_qd', 'C_modified_adaptive_random', 'F − C (QD effect on modified adaptive)')]
    for var_b, var_a, label in comparisons:
        d = delta_pairs(var_a, var_b)
        if d:
            lines.append(f'| {label} | {_fmt(_mean(d))} | {len(d)} |')
        else:
            lines.append(f'| {label} | — | 0 |')
    lines += ['', '## Mean runtime by variant (seconds)', '', '| Variant | Mean runtime (s) | Min | Max |', '|---|---:|---:|---:|']
    for var in all_variants:
        rt = runtime_by_var.get(var, [])
        if rt:
            lines.append(f'| `{var}` | {_mean(rt):.1f} | {min(rt):.1f} | {max(rt):.1f} |')
        else:
            lines.append(f'| `{var}` | — | — | — |')
    lines += ['', '## Adaptive diffusion time statistics', '', '| Variant | Mean t | Min t | Max t | N samples |', '|---|---:|---:|---:|---:|']
    for var in all_variants:
        dt = dt_used_by_var.get(var, [])
        if dt:
            lines.append(f'| `{var}` | {_mean(dt):.3f} | {min(dt):.3f} | {max(dt):.3f} | {len(dt)} |')
    lines += ['', '## Runtime scaling (t ≈ a · n^b)', '']
    if args.profile == 'scaling' or len({int(r['n_objects']) for r in rows}) >= 3:
        by_var_n: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            by_var_n[row['variant']][int(row['n_objects'])].append(float(row['runtime_sec']))
        for var in sorted(all_variants):
            pts: list[tuple[float, float]] = []
            for n_obj, rts in sorted(by_var_n[var].items()):
                pts.append((float(n_obj), float(sum(rts) / len(rts))))
            fit = _loglog_power_fit(pts)
            if fit:
                a, b, r2 = fit
                lines.append(f'- **`{var}`**: `time ≈ {a:.4e} · n^{b:.3f}` (R²={r2:.3f}), точек={len(pts)}')
            elif pts:
                lines.append(f'- **`{var}`**: недостаточно точек для устойчивой оценки степени (n={len(pts)})')
    else:
        lines.append('_Нет минимум трёх различных n в данных — секция пропущена. Запустите `--profile scaling`._')
    lines += ['', '## Computational complexity (theoretical)', '', '| Algorithm | Time complexity | Notes |', '|---|---|---|', '| SDGCA baseline | O(m·n² + it·n³) | ADMM: n³ per SDGCA iteration |', '| SDGCA modified (fixed t) | O(m·n² + K³ + it·n³) | +K³ for expm(L_K), K=total clusters |', '| SDGCA modified (adaptive t) | O(m·n² + K³ + K³ + it·n³) | extra K³ for eigvalsh of K×K Laplacian |', '', 'Where K = m × avg(k_i) ≪ n. For m=8, avg_k=4 → K≈32.', 'Runtime overhead of adaptive t ≈ O(K³) ≈ negligible vs O(n³).', '', '## Correct interpretation guidelines', '', '**Allowed conclusions:**', '- Compare B vs A on same dataset/linkage/seed: isolates fuzzy weighting + heat-kernel diffusion effect', '- Compare C vs B on same conditions: isolates adaptive t from fixed t', '- Compare D/E/F vs A/B/C: isolates QD-selection effect', '- Report dataset-type-stratified analysis: where does modification help', '', '**Not allowed without additional evidence:**', "- Claiming 'modification helps' if B > A on only 1-2 datasets", '- Mixing tuned variant G with honest comparison A/B/C', '- Claiming statistical significance without Wilcoxon test or sufficient N', '', '## Notes', '', '- Variants G and H (tuned) use DEFAULT_PARAMS per dataset; this is NOT an honest comparison', '  with baseline. They are included only to show the upper bound under per-dataset tuning.', '- Runtime comparison is wall-clock on same machine; heat-kernel via expm() is the main', '  cost difference in modified vs baseline.']
    report_path.write_text('\n'.join(lines), encoding='utf-8')

def choose_datasets(profile: str) -> list[tuple[str, str]]:
    return {'smoke': SMOKE_DATASETS, 'mini': MINI_DATASETS, 'scaling': SCALING_DATASETS, 'full': FULL_DATASETS}.get(profile, MINI_DATASETS)

def main() -> None:
    parser = argparse.ArgumentParser(description='Ablation study для SDGCA modified.')
    parser.add_argument('--profile', choices=['smoke', 'mini', 'scaling', 'full'], default='mini')
    parser.add_argument('--m', type=int, default=8)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seeds', nargs='+', type=int, default=[19])
    parser.add_argument('--methods', nargs='+', default=['average'], choices=['average', 'complete', 'single', 'ward'])
    parser.add_argument('--qd-alpha', dest='qd_alpha', type=float, default=0.5)
    parser.add_argument('--include-tuned', dest='include_tuned', action='store_true', help='Включить варианты G/H с per-dataset DEFAULT_PARAMS (нечестное сравнение).')
    parser.add_argument('--variants', nargs='+', default=None, help='Запустить только указанные варианты (например A_baseline_random C_modified_adaptive_random).')
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    created_at = datetime.now().isoformat(timespec='seconds')
    tsv_path = RESULTS_DIR / f'sdgca_modified_ablation_{timestamp}.tsv'
    report_path = RESULTS_DIR / f'sdgca_modified_ablation_{timestamp}.md'
    latest_tsv = RESULTS_DIR / 'sdgca_modified_ablation_latest.tsv'
    latest_report = RESULTS_DIR / 'sdgca_modified_ablation_summary.md'
    datasets = choose_datasets(args.profile)
    variant_cfgs = list(VARIANTS)
    if args.include_tuned:
        variant_cfgs += TUNED_VARIANTS
    if args.variants:
        variant_cfgs = [v for v in variant_cfgs if v['variant'] in args.variants]
    total = len(datasets) * len(variant_cfgs) * len(args.methods) * len(args.seeds)
    print('=' * 72)
    print('SDGCA Modified Ablation Study')
    print(f'profile={args.profile}  datasets={len(datasets)}  variants={len(variant_cfgs)}')
    print(f'm={args.m}  runs={args.runs}  seeds={args.seeds}  methods={args.methods}')
    print(f'include_tuned={args.include_tuned}  planned_rows={total}')
    print('=' * 72)
    for v in variant_cfgs:
        print(f"  {v['variant']:40s} {v['label']}")
    print()
    rows: list[dict] = []
    write_header = True
    for dataset_name, suffix in datasets:
        path = find_dataset(dataset_name, suffix)
        if path is None:
            print(f'[MISS] {dataset_name}{suffix}')
            continue
        dtype = get_dataset_type(dataset_name)
        print(f'\n[DATASET] {dataset_name}  type={dtype}', flush=True)
        for variant_cfg in variant_cfgs:
            for linkage in args.methods:
                for seed in args.seeds:
                    print(f"  {variant_cfg['variant']}/{linkage}/seed={seed}", end=' ... ', flush=True)
                    row = run_variant(path=path, dataset_name=dataset_name, variant_cfg=variant_cfg, linkage=linkage, m=args.m, runs=args.runs, seed=seed, qd_alpha=args.qd_alpha, created_at=created_at)
                    if row is None:
                        print('skip', flush=True)
                        continue
                    rows.append(row)
                    dt_info = f"  t={row['diffusion_time_used']}" if row.get('diffusion_time_used') not in ('', None) else ''
                    print(f"NMI={row['nmi_mean']:.4f}±{row['nmi_std']:.4f}  ARI={row['ari_mean']:.4f}  {row['runtime_sec']:.1f}s{dt_info}", flush=True)
                    append_rows(tsv_path, [row], write_header=write_header)
                    write_header = False
                    shutil.copyfile(tsv_path, latest_tsv)
    if not rows:
        print('[WARN] no rows generated')
        return
    write_rows(tsv_path, rows)
    shutil.copyfile(tsv_path, latest_tsv)
    canonical_tsv = RESULTS_DIR / 'sdgca_modified_ablation.tsv'
    shutil.copyfile(tsv_path, canonical_tsv)
    write_summary_report(report_path, rows, args, tsv_path)
    shutil.copyfile(report_path, latest_report)
    print(f"\n{'=' * 72}")
    print(f'[OK] TSV:           {tsv_path}')
    print(f'[OK] Report:        {report_path}')
    print(f'[OK] Canonical TSV: {canonical_tsv}')
    print(f'[OK] Latest TSV:    {latest_tsv}')
    print(f'[OK] Latest report: {latest_report}')
    print(f'[OK] Total rows:    {len(rows)}')
    print(f"{'=' * 72}")
if __name__ == '__main__':
    main()
