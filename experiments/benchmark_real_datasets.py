from __future__ import annotations
import csv
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import scipy.io
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'consensus_lab'))
from algorithms_base import AlgorithmRegistry
from metrics import compute_nmi, compute_ari, compute_pairwise_f_score
RUNS = 3
M = 20
METHOD = 'ward'
SEED = 42
MIN_N = 30
OUTPUT = ROOT / 'results' / 'benchmark_real_datasets.tsv'
ALGOS = ['hierarchical_baseline', 'hierarchical_weighted', 'sdgca', 'sdgca_modified']
HEADER = ['dataset', 'source', 'n', 'K', 'dim', 'base_members_nmi', 'base_members_ari', 'kmeans_nmi', 'kmeans_ari', 'ward_nmi', 'ward_ari', 'hier_base_nmi', 'hier_base_ari', 'hier_wtd_nmi', 'hier_wtd_ari', 'sdgca_nmi', 'sdgca_ari', 'sdgca_mod_nmi', 'sdgca_mod_ari', 'best_simple', 'best_consensus', 'consensus_wins', 'consensus_gain_nmi', 'elapsed_sec']

def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            done.add(row['dataset'])
    return done

def collect_datasets() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    real_dir = ROOT / 'datasets' / 'real'
    if real_dir.exists():
        for p in sorted(real_dir.glob('*.npz')):
            try:
                d = np.load(p, allow_pickle=True)
                if d['gt'].size >= MIN_N:
                    entries.append((p, 'real_npz'))
            except Exception:
                pass
    mat_dir = ROOT / 'datasets'
    for p in sorted(mat_dir.glob('*.mat')):
        try:
            m = scipy.io.loadmat(str(p))
            gt = m.get('gt', m.get('label', m.get('y', None)))
            if gt is not None and int(gt.size) >= MIN_N:
                entries.append((p, 'mat'))
        except Exception:
            pass
    return entries

def load_dataset(path: Path, source: str) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    if source == 'real_npz':
        d = np.load(path, allow_pickle=True)
        X = d['X'].astype(np.float64)
        gt = d['gt'].ravel().astype(np.int64)
        members = d['members'] if 'members' in d else None
        return (X, gt, members)
    else:
        m = scipy.io.loadmat(str(path))
        gt_raw = m.get('gt', m.get('label', m.get('y')))
        if gt_raw is None:
            raise ValueError(f'No gt/label/y in {path.name}')
        gt = gt_raw.ravel().astype(np.int64)
        gt = gt - gt.min() + 1
        members_raw = m.get('members', None)
        members = np.asarray(members_raw, dtype=np.int64) if members_raw is not None else None
        X_raw = None
        for key in ('X', 'x', 'fea', 'data', 'features'):
            if key in m:
                X_raw = m[key]
                break
        if X_raw is not None:
            X = np.asarray(X_raw, dtype=np.float64)
            if X.shape[0] != gt.size and X.shape[1] == gt.size:
                X = X.T
        else:
            X = None
        return (X, gt, members)

def base_member_metrics(members: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    nmis, aris = ([], [])
    for j in range(members.shape[1]):
        col = members[:, j].ravel().astype(np.int64)
        if len(np.unique(col)) < 2:
            continue
        nmis.append(compute_nmi(col, gt))
        aris.append(compute_ari(col, gt))
    if not nmis:
        return (0.0, 0.0)
    return (float(np.mean(nmis)), float(np.mean(aris)))

def kmeans_metrics(X: np.ndarray, gt: np.ndarray, k: int, n_runs: int=5, seed: int=42) -> tuple[float, float]:
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    best_nmi, best_ari = (-1.0, -1.0)
    for i in range(n_runs):
        km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 9999)))
        labels = km.fit_predict(X) + 1
        nmi = compute_nmi(labels, gt)
        ari = compute_ari(labels, gt)
        if nmi > best_nmi:
            best_nmi, best_ari = (nmi, ari)
    return (best_nmi, best_ari)

def ward_metrics(X: np.ndarray, gt: np.ndarray, k: int) -> tuple[float, float]:
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X) + 1
    return (compute_nmi(labels, gt), compute_ari(labels, gt))

def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUTPUT)
    datasets = collect_datasets()
    remaining = [(p, src) for p, src in datasets if p.stem not in done]
    print(f'Real biomedical datasets: {len(datasets)}')
    print(f'Already done: {len(done)}')
    print(f'Remaining: {len(remaining)}')
    print(f'Output: {OUTPUT}')
    print(f'Config: runs={RUNS}, m={M}, method={METHOD}')
    print()
    write_header = not OUTPUT.exists() or OUTPUT.stat().st_size == 0
    out_f = OUTPUT.open('a', newline='', encoding='utf-8')
    writer = csv.DictWriter(out_f, fieldnames=HEADER, delimiter='\t')
    if write_header:
        writer.writeheader()
    completed = 0
    t_all = time.time()
    for idx, (p, source) in enumerate(remaining):
        t0 = time.time()
        name = p.stem
        print(f'[{idx + 1}/{len(remaining)}] {name} ({source}) ...', flush=True)
        try:
            X, gt, members = load_dataset(p, source)
            n = int(gt.size)
            k = int(np.unique(gt).size)
            dim = int(X.shape[1]) if X is not None else -1
            print(f'  n={n} K={k} d={dim}', flush=True)
            if members is not None and members.shape[1] > 0:
                bm_nmi, bm_ari = base_member_metrics(members, gt)
            else:
                bm_nmi, bm_ari = (float('nan'), float('nan'))
            print(f'  base_members NMI={bm_nmi:.4f}', flush=True)
            if X is not None:
                km_nmi, km_ari = kmeans_metrics(X, gt, k, seed=SEED)
                print(f'  kmeans       NMI={km_nmi:.4f}', flush=True)
                wd_nmi, wd_ari = ward_metrics(X, gt, k)
                print(f'  ward         NMI={wd_nmi:.4f}', flush=True)
            else:
                km_nmi, km_ari = (float('nan'), float('nan'))
                wd_nmi, wd_ari = (float('nan'), float('nan'))
                print(f'  kmeans/ward  skipped (no X)', flush=True)
            algo_results: dict[str, tuple[float, float]] = {}
            for algo_name in ALGOS:
                try:
                    algo = AlgorithmRegistry.get(algo_name)
                    res = algo.run(dataset_path=str(p), m=M, runs=RUNS, method=METHOD, seed=SEED)
                    algo_results[algo_name] = (res.nmi_mean, res.ari_mean)
                    print(f'  {algo_name:28s} NMI={res.nmi_mean:.4f}', flush=True)
                except Exception as exc:
                    print(f'  {algo_name:28s} ERROR: {exc}', flush=True)
                    algo_results[algo_name] = (float('nan'), float('nan'))
            simple_nmis = [v for v in [bm_nmi, km_nmi, wd_nmi] if not np.isnan(v)]
            best_simple = max(simple_nmis) if simple_nmis else float('nan')
            cons_nmis = {k2: v[0] for k2, v in algo_results.items() if not np.isnan(v[0])}
            if cons_nmis:
                best_cons_name = max(cons_nmis, key=lambda x: cons_nmis[x])
                best_cons = cons_nmis[best_cons_name]
            else:
                best_cons_name, best_cons = ('none', float('nan'))
            consensus_wins = not np.isnan(best_cons) and (not np.isnan(best_simple)) and (best_cons > best_simple)
            gain = best_cons - best_simple if not np.isnan(best_cons) else float('nan')
            hr_b = algo_results['hierarchical_baseline']
            hr_w = algo_results['hierarchical_weighted']
            sg = algo_results['sdgca']
            sg_m = algo_results['sdgca_modified']
            row = {'dataset': name, 'source': source, 'n': n, 'K': k, 'dim': dim, 'base_members_nmi': round(bm_nmi, 5), 'base_members_ari': round(bm_ari, 5), 'kmeans_nmi': round(km_nmi, 5), 'kmeans_ari': round(km_ari, 5), 'ward_nmi': round(wd_nmi, 5), 'ward_ari': round(wd_ari, 5), 'hier_base_nmi': round(hr_b[0], 5), 'hier_base_ari': round(hr_b[1], 5), 'hier_wtd_nmi': round(hr_w[0], 5), 'hier_wtd_ari': round(hr_w[1], 5), 'sdgca_nmi': round(sg[0], 5), 'sdgca_ari': round(sg[1], 5), 'sdgca_mod_nmi': round(sg_m[0], 5), 'sdgca_mod_ari': round(sg_m[1], 5), 'best_simple': round(best_simple, 5), 'best_consensus': round(best_cons, 5), 'consensus_wins': int(consensus_wins), 'consensus_gain_nmi': round(gain, 5), 'elapsed_sec': round(time.time() - t0, 1)}
            writer.writerow(row)
            out_f.flush()
            done.add(name)
            completed += 1
            marker = 'WIN' if consensus_wins else 'loss'
            print(f'  -> simple={best_simple:.4f}  consensus={best_cons:.4f}  [{marker}]', flush=True)
        except Exception as exc:
            print(f'  ERROR: {exc}', flush=True)
            traceback.print_exc()
        print()
    out_f.close()
    total_min = (time.time() - t_all) / 60
    print(f'Done in {total_min:.1f} min. Results: {OUTPUT}')
    rows: list[dict] = []
    with OUTPUT.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows:
        return

    def avg(col: str) -> float:
        vals = [float(r[col]) for r in rows if r.get(col) not in ('', 'nan')]
        return sum(vals) / len(vals) if vals else float('nan')
    n_rows = len(rows)
    wins = sum((1 for r in rows if r.get('consensus_wins') == '1'))
    losses = n_rows - wins
    print()
    print('=' * 70)
    print('SUMMARY: Does consensus clustering outperform simple algorithms?')
    print('=' * 70)
    print(f'Datasets evaluated: {n_rows}')
    print(f'Consensus WINS  (higher NMI): {wins:3d} / {n_rows}  ({100 * wins / n_rows:.0f}%)')
    print(f'Consensus LOSES (lower NMI):  {losses:3d} / {n_rows}  ({100 * losses / n_rows:.0f}%)')
    print()
    cols = [('Base members', 'base_members_nmi'), ('KMeans', 'kmeans_nmi'), ('Ward hierarch.', 'ward_nmi'), ('Hier. baseline', 'hier_base_nmi'), ('Hier. weighted', 'hier_wtd_nmi'), ('SDGCA', 'sdgca_nmi'), ('SDGCA modified', 'sdgca_mod_nmi')]
    print(f"{'Algorithm':<22}  {'Avg NMI':>8}")
    print('-' * 34)
    for label, col in cols:
        print(f'  {label:<20}  {avg(col):8.4f}')
    print()
    print(f"Average NMI gain (best_consensus - best_simple): {avg('consensus_gain_nmi'):+.4f}")
if __name__ == '__main__':
    main()
