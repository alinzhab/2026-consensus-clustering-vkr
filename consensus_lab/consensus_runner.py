from __future__ import annotations
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
from hierarchical_consensus import load_dataset, validate_gt, validate_members, validate_method
from metrics import compute_ari, compute_nmi, compute_pairwise_f_score

def _build_bc_idx(members: np.ndarray, m: int, cnt_times: int, rng: np.random.Generator, selection_strategy: str, qd_alpha: float) -> np.ndarray:
    pool_size = members.shape[1]
    if selection_strategy == 'qd':
        from ensemble_selection import select_qd_subset
        qd_indices = select_qd_subset(members, m, qd_alpha=qd_alpha)
        return np.tile(qd_indices, (cnt_times, 1))
    return np.vstack([rng.permutation(pool_size)[:m] for _ in range(cnt_times)])

def run_consensus_loop(dataset_path: str | Path, build_consensus_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray], *, data_name: str | None=None, seed: int=19, m: int=40, cnt_times: int=20, method: str='average', selection_strategy: str='random', qd_alpha: float=0.5, clamp_m_name: str | None=None) -> dict:
    members, gt = load_dataset(dataset_path)
    pool_size = int(members.shape[1])
    m = int(m)
    if m < 1:
        raise ValueError('m must be at least 1')
    if clamp_m_name:
        from sdgca import _clamp_m

        m = _clamp_m(m, pool_size, clamp_m_name)
    elif m > pool_size:
        warnings.warn(
            f'requested m={m} exceeds pool size {pool_size} (columns in members); using m={pool_size}',
            RuntimeWarning,
            stacklevel=2,
        )
        m = pool_size
    members = validate_members(members, m)
    gt = validate_gt(gt, members.shape[0])
    method = validate_method(method)
    if cnt_times < 1:
        raise ValueError('cnt_times must be positive')
    cls_nums = int(np.unique(gt).size)
    rng = np.random.default_rng(seed)
    bc_idx = _build_bc_idx(members, m, cnt_times, rng, selection_strategy, qd_alpha)
    from hierarchical_consensus import get_cls_result
    nmi_scores = np.zeros(cnt_times, dtype=np.float64)
    ari_scores = np.zeros(cnt_times, dtype=np.float64)
    f_scores = np.zeros(cnt_times, dtype=np.float64)
    extra_per_run: list[dict] = []
    for run_idx in range(cnt_times):
        base_cls = members[:, bc_idx[run_idx, :]]
        result = build_consensus_fn(base_cls, gt, m)
        if isinstance(result, tuple):
            consensus_matrix, run_extra = result
        else:
            consensus_matrix, run_extra = (result, {})
        labels = get_cls_result(consensus_matrix, cls_nums, method=method)
        nmi_scores[run_idx] = compute_nmi(labels, gt)
        ari_scores[run_idx] = compute_ari(labels, gt)
        f_scores[run_idx] = compute_pairwise_f_score(labels, gt)
        if run_extra:
            extra_per_run.append(run_extra)
    out = {'data_name': data_name or Path(dataset_path).stem, 'nmi_mean': float(np.mean(nmi_scores)), 'nmi_std': float(np.std(nmi_scores)), 'ari_mean': float(np.mean(ari_scores)), 'ari_std': float(np.std(ari_scores)), 'f_mean': float(np.mean(f_scores)), 'f_std': float(np.std(f_scores)), 'nmi_scores': nmi_scores, 'ari_scores': ari_scores, 'f_scores': f_scores, 'selected_base_clusterings': bc_idx[0].astype(int).tolist(), 'method': method, 'm': int(m), 'runs': int(cnt_times), 'seed': int(seed), 'pool_size': pool_size, 'selection_strategy': selection_strategy, 'qd_alpha': float(qd_alpha) if selection_strategy == 'qd' else None}
    if extra_per_run:
        out['_extra_per_run'] = extra_per_run
    return out
