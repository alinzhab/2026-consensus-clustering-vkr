from __future__ import annotations
from typing import Sequence, Union
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
__all__ = ['contingency', 'compute_nmi', 'compute_ari', 'compute_pairwise_f_score']
_LabelArray = Union[np.ndarray, Sequence[int]]

def contingency(labels_a: _LabelArray, labels_b: _LabelArray) -> np.ndarray:
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    if labels_a.size != labels_b.size:
        raise ValueError('label vectors must have the same length')
    _, inv_a = np.unique(labels_a, return_inverse=True)
    _, inv_b = np.unique(labels_b, return_inverse=True)
    cont = np.zeros((inv_a.max() + 1, inv_b.max() + 1), dtype=np.int64)
    np.add.at(cont, (inv_a, inv_b), 1)
    return cont

def compute_nmi(pred_labels: _LabelArray, true_labels: _LabelArray) -> float:
    return float(normalized_mutual_info_score(true_labels, pred_labels))

def compute_ari(pred_labels: _LabelArray, true_labels: _LabelArray) -> float:
    return float(adjusted_rand_score(true_labels, pred_labels))

def compute_pairwise_f_score(pred_labels: _LabelArray, true_labels: _LabelArray) -> float:
    cont = contingency(pred_labels, true_labels).astype(np.float64)
    row_sums = np.sum(cont, axis=1)
    col_sums = np.sum(cont, axis=0)
    intersection_pairs = float(np.sum(cont * (cont - 1.0) / 2.0))
    predicted_pairs = float(np.sum(row_sums * (row_sums - 1.0) / 2.0))
    true_pairs = float(np.sum(col_sums * (col_sums - 1.0) / 2.0))
    precision = 1.0 if predicted_pairs == 0.0 else intersection_pairs / predicted_pairs
    recall = 1.0 if true_pairs == 0.0 else intersection_pairs / true_pairs
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
