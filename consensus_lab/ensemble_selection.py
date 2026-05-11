from __future__ import annotations
import numpy as np
__all__ = ['DATASET_TYPES', 'compute_partition_quality', 'compute_pairwise_agreement', 'get_dataset_type', 'partition_agreement', 'select_qd_subset']
DATASET_TYPES: dict[str, str] = {'Ecoli': 'real_bio', 'GLIOMA': 'real_bio', 'Lung': 'real_bio', 'BBC': 'real_text', 'orlraws10P': 'high_dimensional', 'Aggregation': 'compact', 'densired_compact_hard': 'compact', 'analysis_densired_compact': 'compact', 'analysis_simple_separated': 'compact', 'custom_densired_dataset': 'compact', 'analysis_simple_overlap': 'overlapping', 'analysis_repliclust_oblong': 'overlapping', 'repliclust_oblong_overlap': 'overlapping', 'densired_stretched_hard': 'elongated', 'analysis_densired_stretched': 'elongated', 'analysis_imbalanced': 'imbalanced', 'analysis_highdim': 'high_dimensional', 'repliclust_highdim_hard': 'high_dimensional', 'densired_mix_hard': 'mixed_complex', 'analysis_repliclust_heterogeneous': 'mixed_complex', 'repliclust_heterogeneous_hard': 'mixed_complex', 'design_compact_easy_5k': 'compact', 'design_compact_easy_8k': 'compact', 'design_overlap_moderate': 'overlapping', 'design_overlap_oblong': 'overlapping', 'design_imbalanced_6x': 'imbalanced', 'design_imbalanced_oblong': 'imbalanced', 'design_highdim_20d': 'high_dimensional', 'design_highdim_40d': 'high_dimensional', 'design_elongated_2d': 'elongated', 'design_elongated_density': 'elongated', 'design_density_varied_low_noise': 'density_varied', 'design_density_varied_noisy': 'density_varied', 'design_mixed_complex_6d': 'mixed_complex', 'design_mixed_complex_branchy': 'mixed_complex', 'design_mini_compact': 'compact', 'design_mini_overlap': 'overlapping', 'design_mini_imbalanced': 'imbalanced', 'design_mini_highdim': 'high_dimensional', 'design_mini_elongated': 'elongated', 'design_mini_density_varied': 'density_varied', 'design_mini_mixed_complex': 'mixed_complex'}
DATASET_TYPE_LABELS: dict[str, str] = {'real_bio': 'Реальные (биология)', 'real_text': 'Реальные (текст/документы)', 'compact': 'Синт. компактные', 'overlapping': 'Синт. перекрывающиеся', 'elongated': 'Синт. вытянутые', 'imbalanced': 'Синт. с дисбалансом', 'high_dimensional': 'Высокоразмерные', 'mixed_complex': 'Синт. сложные', 'density_varied': 'Синт. неоднородная плотность', 'unknown': 'Неизвестный тип'}

def get_dataset_type(dataset_name: str) -> str:
    name = str(dataset_name)
    for suffix in ('.mat', '.npz', '.csv', '.tsv', '.txt', '.json'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
    if name in DATASET_TYPES:
        return DATASET_TYPES[name]
    for key, dtype in DATASET_TYPES.items():
        if key in name or name in key:
            return dtype
    return 'unknown'

def partition_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    n = int(a.size)
    _, inv_a = np.unique(a, return_inverse=True)
    _, inv_b = np.unique(b, return_inverse=True)
    ka = int(inv_a.max()) + 1
    kb = int(inv_b.max()) + 1
    cont = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(cont, (inv_a, inv_b), 1)
    row_sq = float(np.sum(cont.sum(axis=1) ** 2))
    col_sq = float(np.sum(cont.sum(axis=0) ** 2))
    both_sq = float(np.sum(cont ** 2))
    return float((n ** 2 - row_sq - col_sq + 2.0 * both_sq) / n ** 2)

def compute_pairwise_agreement(members: np.ndarray) -> np.ndarray:
    M = members.shape[1]
    agg = np.eye(M, dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            v = partition_agreement(members[:, i], members[:, j])
            agg[i, j] = v
            agg[j, i] = v
    return agg

def compute_partition_quality(agreement_matrix: np.ndarray) -> np.ndarray:
    M = agreement_matrix.shape[0]
    if M == 1:
        return np.ones(1, dtype=np.float64)
    off_diag_sum = agreement_matrix.sum(axis=1) - 1.0
    return off_diag_sum / float(M - 1)

def select_qd_subset(members: np.ndarray, m: int, qd_alpha: float=0.5, agreement_matrix: np.ndarray | None=None, quality_scores: np.ndarray | None=None) -> np.ndarray:
    M = int(members.shape[1])
    m = min(int(m), M)
    if m == M:
        return np.arange(M, dtype=np.int64)
    if agreement_matrix is None:
        agreement_matrix = compute_pairwise_agreement(members)
    if quality_scores is None:
        quality_scores = compute_partition_quality(agreement_matrix)
    qd_alpha = float(np.clip(qd_alpha, 0.0, 1.0))
    selected: list[int] = [int(np.argmax(quality_scores))]
    remaining = list(set(range(M)) - {selected[0]})
    while len(selected) < m and remaining:
        sel_arr = np.array(selected, dtype=np.int64)
        rem_arr = np.array(remaining, dtype=np.int64)
        diss = 1.0 - agreement_matrix[np.ix_(rem_arr, sel_arr)]
        avg_div = diss.mean(axis=1)
        qual = quality_scores[rem_arr]
        scores = qd_alpha * qual + (1.0 - qd_alpha) * avg_div
        best_local = int(np.argmax(scores))
        best_global = remaining[best_local]
        selected.append(best_global)
        remaining.pop(best_local)
    return np.array(selected[:m], dtype=np.int64)
