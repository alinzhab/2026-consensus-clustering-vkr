from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PROFILE_COLUMNS = ['algorithm', 'variant', 'selection_strategy', 'qd_alpha', 'linkage', 'm']

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'results' / 'single_vs_consensus_benchmark.tsv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'selected_consensus_profile.json')
    parser.add_argument('--metric', choices=['NMI', 'ARI', 'F-score', 'rank'], default='rank')
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\t')
    df = df[(df['split'] == 'validation') & (df['algorithm_family'] == 'consensus') & (df['status'] == 'ok')].copy()
    if df.empty:
        raise SystemExit('No successful validation consensus rows found.')
    for col in ['NMI', 'ARI', 'F-score', 'runtime_sec']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['profile_id'] = df[PROFILE_COLUMNS].fillna('').map(str).agg(lambda row: '|'.join(row.to_list()), axis=1)
    if args.metric == 'rank':
        df['rank_score'] = df.groupby('dataset_id')['NMI'].rank(method='average', ascending=False)
        summary = df.groupby('profile_id').agg(mean_rank=('rank_score', 'mean'), mean_nmi=('NMI', 'mean'), mean_ari=('ARI', 'mean'), mean_f=('F-score', 'mean'), mean_runtime=('runtime_sec', 'mean'), datasets=('dataset_id', 'nunique')).reset_index().sort_values(['mean_rank', 'mean_nmi', 'mean_runtime'], ascending=[True, False, True])
    else:
        metric_column = {'NMI': 'mean_nmi', 'ARI': 'mean_ari', 'F-score': 'mean_f'}[args.metric]
        summary = df.groupby('profile_id').agg(mean_rank=('NMI', lambda s: np.nan), mean_nmi=('NMI', 'mean'), mean_ari=('ARI', 'mean'), mean_f=('F-score', 'mean'), mean_runtime=('runtime_sec', 'mean'), datasets=('dataset_id', 'nunique')).reset_index().sort_values([metric_column, 'mean_runtime'], ascending=[False, True])
    best = summary.iloc[0].to_dict()
    sample = df[df['profile_id'] == best['profile_id']].iloc[0]
    profile = {col: None if pd.isna(sample[col]) else sample[col] for col in PROFILE_COLUMNS}
    profile.update({'selected_on': 'validation', 'selection_metric': args.metric, 'mean_validation_nmi': float(best['mean_nmi']), 'mean_validation_ari': float(best['mean_ari']), 'mean_validation_f_score': float(best['mean_f']), 'mean_validation_runtime_sec': float(best['mean_runtime']), 'validation_dataset_count': int(best['datasets']), 'note': 'Global profile selected on dataset-level validation split. Do not tune on test.'})
    if str(profile.get('qd_alpha', '')) in {'', 'nan'}:
        profile['qd_alpha'] = None
    else:
        profile['qd_alpha'] = float(profile['qd_alpha'])
    profile['m'] = int(profile['m'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    summary.to_csv(args.output.with_suffix('.candidates.tsv'), sep='\t', index=False)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
