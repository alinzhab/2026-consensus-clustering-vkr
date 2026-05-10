"""Analyze test-split benefit of consensus clustering over single baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def bootstrap_ci(values: np.ndarray, seed: int = 19, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def possible_reason(row: pd.Series) -> str:
    dtype = row.get("dataset_type", "")
    if row["best_nmi"] < 0.35:
        if dtype == "overlapping":
            return "overlap makes cluster boundaries ambiguous for all methods"
        if dtype == "high_dimensional":
            return "high dimensionality weakens distance-based single and base clusterings"
        if dtype == "imbalanced":
            return "small clusters are likely absorbed by larger clusters"
        if dtype == "density_varied":
            return "different densities distort Euclidean distances and co-association"
        if dtype == "mixed_complex":
            return "multiple hard factors act together: overlap, imbalance, high dimension"
        if dtype == "elongated":
            return "non-spherical shape is sensitive to linkage and base clustering quality"
        return "low separability or weak base ensemble"
    if row["consensus_delta_vs_single"] < -0.02:
        return "best single baseline fits this geometry better than the current consensus profile"
    return "moderate case; inspect linkage/base ensemble diversity"


def profile_mask(df: pd.DataFrame, profile: dict) -> pd.Series:
    mask = df["algorithm"].astype(str) == str(profile["algorithm"])
    for col in ["variant", "selection_strategy", "linkage"]:
        if profile.get(col) is not None:
            mask &= df[col].astype(str) == str(profile[col])
    if profile.get("qd_alpha") is None:
        mask &= df["qd_alpha"].isna() | (df["qd_alpha"].astype(str) == "")
    else:
        mask &= pd.to_numeric(df["qd_alpha"], errors="coerce").round(6) == round(float(profile["qd_alpha"]), 6)
    mask &= pd.to_numeric(df["m"], errors="coerce") == int(profile["m"])
    return mask


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "single_vs_consensus_benchmark.tsv")
    parser.add_argument("--profile", type=Path, default=ROOT / "results" / "selected_consensus_profile.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    for col in ["NMI", "ARI", "F-score", "runtime_sec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    test = df[(df["split"] == "test") & (df["status"] == "ok")].copy()
    if test.empty:
        raise SystemExit("No successful test rows found.")

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    singles = test[test["algorithm_family"] == "single"].copy()
    consensus_all = test[test["algorithm_family"] == "consensus"].copy()
    selected_consensus = consensus_all[profile_mask(consensus_all, profile)].copy()
    if selected_consensus.empty:
        raise SystemExit("Selected profile has no successful test rows.")

    best_single = singles.sort_values("NMI", ascending=False).groupby("dataset_id").head(1)
    selected_best = selected_consensus.sort_values("NMI", ascending=False).groupby("dataset_id").head(1)
    paired = selected_best.merge(
        best_single,
        on="dataset_id",
        suffixes=("_consensus", "_single"),
        how="inner",
    )
    if paired.empty:
        raise SystemExit("No paired single/consensus test datasets found.")

    paired["delta_nmi"] = paired["NMI_consensus"] - paired["NMI_single"]
    paired["delta_ari"] = paired["ARI_consensus"] - paired["ARI_single"]
    paired["delta_f"] = paired["F-score_consensus"] - paired["F-score_single"]
    paired["runtime_ratio"] = paired["runtime_sec_consensus"] / paired["runtime_sec_single"].replace(0, np.nan)

    ci_low, ci_high = bootstrap_ci(paired["delta_nmi"].to_numpy())
    try:
        wilcoxon_p = float(wilcoxon(paired["delta_nmi"]).pvalue)
    except Exception:
        wilcoxon_p = np.nan

    stat_rows = [
        {
            "test": "wilcoxon_consensus_minus_best_single_nmi",
            "statistic": "",
            "p_value": wilcoxon_p,
            "n": int(len(paired)),
        }
    ]
    wide = test.pivot_table(index="dataset_id", columns="algorithm", values="NMI", aggfunc="mean")
    common_algorithms = [col for col in wide.columns if wide[col].notna().sum() == wide.shape[0]]
    if len(common_algorithms) >= 3 and wide.shape[0] >= 3:
        try:
            fried = friedmanchisquare(*[wide[col].to_numpy() for col in common_algorithms])
            stat_rows.append(
                {
                    "test": "friedman_all_common_algorithms_nmi",
                    "statistic": float(fried.statistic),
                    "p_value": float(fried.pvalue),
                    "n": int(wide.shape[0]),
                }
            )
        except Exception:
            pass

    by_type = (
        paired.groupby("dataset_type_consensus")
        .agg(
            datasets=("dataset_id", "nunique"),
            mean_single_nmi=("NMI_single", "mean"),
            mean_consensus_nmi=("NMI_consensus", "mean"),
            mean_delta_nmi=("delta_nmi", "mean"),
            win_rate=("delta_nmi", lambda s: float(np.mean(s > 0))),
            mean_runtime_ratio=("runtime_ratio", "mean"),
        )
        .reset_index()
        .rename(columns={"dataset_type_consensus": "dataset_type"})
    )

    best_all = test.sort_values("NMI", ascending=False).groupby("dataset_id").head(1)
    worst_all = test.sort_values("NMI", ascending=True).groupby("dataset_id").head(1)
    failure = paired.merge(
        best_all[["dataset_id", "algorithm", "NMI"]].rename(columns={"algorithm": "best_algorithm", "NMI": "best_nmi"}),
        on="dataset_id",
        how="left",
    ).merge(
        worst_all[["dataset_id", "algorithm"]].rename(columns={"algorithm": "worst_algorithm"}),
        on="dataset_id",
        how="left",
    )
    failure["consensus_delta_vs_single"] = failure["delta_nmi"]
    failure["dataset_type"] = failure["dataset_type_consensus"]
    failure["possible_reason"] = failure.apply(possible_reason, axis=1)
    failure_out = failure[
        [
            "dataset_id",
            "dataset_type",
            "difficulty_level_consensus",
            "best_algorithm",
            "worst_algorithm",
            "best_nmi",
            "consensus_delta_vs_single",
            "possible_reason",
        ]
    ].rename(columns={"difficulty_level_consensus": "difficulty_level"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_type.to_csv(args.out_dir / "single_vs_consensus_by_type.tsv", sep="\t", index=False)
    pd.DataFrame(stat_rows).to_csv(args.out_dir / "single_vs_consensus_stat_tests.tsv", sep="\t", index=False)
    failure_out.to_csv(args.out_dir / "failure_cases.tsv", sep="\t", index=False)

    summary = [
        "# Single vs consensus clustering summary",
        "",
        "Protocol: dataset-level test split only. Ground truth is used only for final metrics.",
        "",
        "## Selected consensus profile",
        "```json",
        json.dumps(profile, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Test result",
        f"- Paired test datasets: {len(paired)}",
        f"- Mean NMI, best single: {paired['NMI_single'].mean():.4f}",
        f"- Mean NMI, selected consensus: {paired['NMI_consensus'].mean():.4f}",
        f"- Mean delta NMI: {paired['delta_nmi'].mean():+.4f}",
        f"- Bootstrap 95% CI for delta NMI: [{ci_low:+.4f}, {ci_high:+.4f}]",
        f"- Win-rate consensus over best single: {np.mean(paired['delta_nmi'] > 0):.3f}",
        f"- Mean runtime ratio consensus/single: {paired['runtime_ratio'].mean():.2f}",
        f"- Wilcoxon p-value: {wilcoxon_p:.6g}" if np.isfinite(wilcoxon_p) else "- Wilcoxon p-value: not available",
        "",
        "## By dataset type",
        to_markdown_table(by_type),
        "",
        "## Honest interpretation",
    ]
    if paired["delta_nmi"].mean() > 0 and (not np.isfinite(wilcoxon_p) or wilcoxon_p < 0.05):
        summary.append("The selected consensus clustering profile improves over the best single baseline on this test split.")
    elif paired["delta_nmi"].mean() > 0:
        summary.append("The selected consensus clustering profile has positive average delta, but the evidence is not statistically strong on this limited test split.")
    else:
        summary.append("The selected consensus clustering profile does not improve over the best single baseline on this test split.")
    summary.append("Do not claim superiority before running the full 500-dataset protocol.")
    (args.out_dir / "single_vs_consensus_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary[:18]))


if __name__ == "__main__":
    main()
