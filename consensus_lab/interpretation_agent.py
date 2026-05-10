"""Local interpretation agent for dataset diagnostics and recommendations.

The agent is intentionally deterministic: it does not change clustering
results and does not call external LLM APIs.  It explains already computed
diagnostics and recommendations in user-facing text, which keeps the research
protocol reproducible.
"""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, digits: int = 3, default: str = "not available") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _level(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "unknown"
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def _pick_primary_algorithm(recommendations: dict[str, Any]) -> str:
    per_algorithm = recommendations.get("per_algorithm") or {}
    method = (per_algorithm.get("hierarchical_weighted") or {}).get("method")
    if "sdgca_modified" in per_algorithm:
        if method == "single":
            return "hierarchical_weighted"
        return "sdgca_modified"
    if "hierarchical_weighted" in per_algorithm:
        return "hierarchical_weighted"
    return "hierarchical_baseline"


def interpret_dataset(analysis: dict[str, Any], recommendations: dict[str, Any]) -> dict[str, Any]:
    """Explain dataset structure and recommended consensus-clustering settings."""
    n = analysis.get("n_objects") or analysis.get("n_samples")
    d = analysis.get("n_features")
    k = analysis.get("n_classes")
    has_gt = analysis.get("gt_shape") is not None or k is not None
    has_members = bool(analysis.get("has_members"))

    density_variation = analysis.get("density_variation")
    overlap_ratio = analysis.get("overlap_ratio")
    imbalance_ratio = analysis.get("imbalance_ratio")
    elongation_max = analysis.get("elongation_max")
    outlier_ratio = analysis.get("outlier_ratio")
    silhouette = analysis.get("silhouette_score")
    exp_2d = analysis.get("explained_variance_2d")

    findings: list[str] = []
    warnings: list[str] = []

    findings.append(
        f"Dataset size: {n or 'unknown'} objects, {d or 'unknown'} features"
        + (f", {k} classes by gt." if k is not None else ".")
    )
    if has_members:
        findings.append("The dataset already has members, so it can be used by consensus clustering.")
    else:
        findings.append("The dataset has no members; the system must build base clusterings from X first.")

    if has_gt:
        findings.append(
            "gt is available and must be used only for evaluation/diagnostics, not for building clusters."
        )
    else:
        warnings.append(
            "gt is absent: recommendations are based on internal structure diagnostics, not NMI/ARI/F-score."
        )

    dim_level = _level(analysis.get("dimensionality_ratio"), 0.10, 0.50)
    density_level = _level(density_variation, 0.30, 0.60)
    overlap_level = _level(overlap_ratio, 0.15, 0.30)
    imbalance_level = _level(imbalance_ratio, 2.0, 5.0) if imbalance_ratio is not None else "unknown"
    elongation_level = _level(elongation_max, 3.0, 8.0) if elongation_max is not None else "unknown"

    if dim_level == "high":
        warnings.append("High d/n ratio: distance-based clustering can become less reliable.")
    if density_level == "high":
        findings.append("Density is heterogeneous; average linkage or SDGCA-style methods are worth checking.")
    if overlap_level == "high":
        warnings.append("High overlap by nearest enemy/friend distances: metrics can be low for all algorithms.")
    if imbalance_level == "high":
        warnings.append("Strong class imbalance: small clusters can be lost by unweighted methods.")
    if elongation_level == "high":
        findings.append("Clusters look elongated; single/average linkage should be inspected carefully.")
    if outlier_ratio is not None and outlier_ratio > 0.10:
        warnings.append("Outlier ratio is elevated; QD-selection and larger m may be useful.")

    primary = _pick_primary_algorithm(recommendations)
    per_algorithm = recommendations.get("per_algorithm") or {}
    primary_params = per_algorithm.get(primary, {})

    summary = (
        "The agent recommends starting with "
        f"{primary}, linkage={primary_params.get('method', 'average')}, "
        f"m={recommendations.get('m')}, selection strategy from the experiment settings."
    )

    evidence = {
        "silhouette_score": _fmt(silhouette),
        "density_variation": _fmt(density_variation),
        "overlap_ratio": _fmt(overlap_ratio),
        "imbalance_ratio": _fmt(imbalance_ratio),
        "elongation_max": _fmt(elongation_max),
        "explained_variance_2d": _fmt(exp_2d),
        "outlier_ratio": _fmt(outlier_ratio),
    }

    return {
        "title": "AI interpretation agent",
        "summary": summary,
        "findings": findings,
        "warnings": warnings,
        "recommended_algorithm": primary,
        "recommended_parameters": {
            "m": recommendations.get("m"),
            "k_min": recommendations.get("k_min"),
            "k_max": recommendations.get("k_max"),
            "strategy": recommendations.get("strategy"),
            "preprocessing": recommendations.get("preprocessing"),
            **primary_params,
        },
        "evidence": evidence,
        "protocol_note": (
            "This is an interpretation layer. It explains diagnostics and recommendations; "
            "it does not participate in clustering and does not tune on the test split."
        ),
    }

