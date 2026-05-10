"""Full comparison benchmark: sdgca vs sdgca_modified (v2 with partition agreements).

Runs both algorithms on all qualifying datasets (n <= MAX_N) and writes results
to results/sdgca_comparison_v2.tsv incrementally — safe to interrupt and resume.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "consensus_lab"))

from sdgca import run_sdgca
from sdgca_modified import run_sdgca_modified

# ── config ────────────────────────────────────────────────────────────────────
MAX_N        = 400
RUNS         = 2        # independent runs per (dataset, algo, method)
M            = 20       # ensemble size
METHODS      = ["ward", "average"]
SEED         = 42
OUTPUT_FILE  = ROOT / "results" / "sdgca_comparison_v2.tsv"
# ─────────────────────────────────────────────────────────────────────────────

HEADER = [
    "dataset", "n", "k", "dim",
    "method",
    "sdgca_nmi", "sdgca_ari", "sdgca_f", "sdgca_nmi_std", "sdgca_ari_std",
    "mod_nmi",   "mod_ari",   "mod_f",   "mod_nmi_std",   "mod_ari_std",
    "delta_nmi", "delta_ari",
    "winner",
    "sdgca_sec", "mod_sec",
]


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            done.add((row["dataset"], row["method"]))
    return done


def collect_datasets() -> list[Path]:
    synth = ROOT / "datasets" / "massive_synthetic"
    real  = ROOT / "datasets"
    paths: list[Path] = []
    for p in sorted(synth.glob("*.npz")):
        try:
            n = np.load(p, allow_pickle=True)["gt"].size
            if n <= MAX_N:
                paths.append(p)
        except Exception:
            pass
    for p in sorted(real.glob("*.mat")):
        try:
            n = np.load(p, allow_pickle=True)["gt"].size
            if n <= MAX_N:
                paths.append(p)
        except Exception:
            pass
    return paths


def dataset_meta(p: Path) -> tuple[int, int, int]:
    d = np.load(p, allow_pickle=True)
    gt = d["gt"].ravel()
    n  = gt.size
    k  = len(np.unique(gt))
    try:
        dim = int(d["X"].shape[1])
    except Exception:
        dim = -1
    return n, k, dim


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUTPUT_FILE)

    datasets = collect_datasets()
    total    = sum(len(METHODS) - sum(1 for m in METHODS if (p.stem, m) in done)
                   for p in datasets)
    remaining = [(p, m) for p in datasets for m in METHODS if (p.stem, m) not in done]

    print(f"Datasets qualifying (n<={MAX_N}): {len(datasets)}")
    print(f"Already done: {len(done)//len(METHODS) if METHODS else 0} datasets")
    print(f"Tasks remaining: {len(remaining)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Settings: runs={RUNS}, m={M}, methods={METHODS}")
    print()

    write_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
    out_f = OUTPUT_FILE.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=HEADER, delimiter="\t")
    if write_header:
        writer.writeheader()

    completed = 0
    t_start_all = time.time()

    for idx, (p, method) in enumerate(remaining):
        if (p.stem, method) in done:
            continue

        n, k, dim = dataset_meta(p)
        t0 = time.time()

        try:
            t_s = time.time()
            r_base = run_sdgca(
                str(p), cnt_times=RUNS, m=M, method=method, seed=SEED,
            )
            sdgca_sec = round(time.time() - t_s, 1)

            t_s = time.time()
            r_mod = run_sdgca_modified(
                str(p), cnt_times=RUNS, m=M, method=method, seed=SEED,
            )
            mod_sec = round(time.time() - t_s, 1)

            dnmi   = round(r_mod["nmi_mean"] - r_base["nmi_mean"], 5)
            dari   = round(r_mod["ari_mean"] - r_base["ari_mean"], 5)
            winner = "modified" if dnmi > 0.001 else ("sdgca" if dnmi < -0.001 else "tie")

            row = {
                "dataset":     p.stem,
                "n":           n,
                "k":           k,
                "dim":         dim,
                "method":      method,
                "sdgca_nmi":   round(r_base["nmi_mean"], 5),
                "sdgca_ari":   round(r_base["ari_mean"], 5),
                "sdgca_f":     round(r_base["f_mean"],   5),
                "sdgca_nmi_std": round(r_base["nmi_std"], 5),
                "sdgca_ari_std": round(r_base["ari_std"], 5),
                "mod_nmi":     round(r_mod["nmi_mean"],  5),
                "mod_ari":     round(r_mod["ari_mean"],  5),
                "mod_f":       round(r_mod["f_mean"],    5),
                "mod_nmi_std": round(r_mod["nmi_std"],   5),
                "mod_ari_std": round(r_mod["ari_std"],   5),
                "delta_nmi":   dnmi,
                "delta_ari":   dari,
                "winner":      winner,
                "sdgca_sec":   sdgca_sec,
                "mod_sec":     mod_sec,
            }
            writer.writerow(row)
            out_f.flush()
            done.add((p.stem, method))
            completed += 1

            elapsed = time.time() - t_start_all
            rate    = completed / elapsed * 60
            left    = len(remaining) - completed
            eta_min = left / max(rate, 1e-6)

            mark = " WIN" if winner == "modified" else (" loss" if winner == "sdgca" else " tie")
            print(
                f"[{completed:3d}/{len(remaining)}] {p.stem[:28]:28s} "
                f"{method:7s} n={n:3d}  "
                f"sdgca={r_base['nmi_mean']:.4f}  mod={r_mod['nmi_mean']:.4f}  "
                f"dNMI={dnmi:+.4f}{mark}  "
                f"{sdgca_sec:.0f}s+{mod_sec:.0f}s  ETA {eta_min:.0f}min",
                flush=True,
            )

        except Exception as exc:
            elapsed_ds = round(time.time() - t0, 1)
            print(f"  ERROR {p.stem} [{method}] after {elapsed_ds}s: {exc}", flush=True)
            traceback.print_exc()

    out_f.close()
    print()
    print(f"Done. Total time: {(time.time()-t_start_all)/60:.1f} min")
    print(f"Results: {OUTPUT_FILE}")

    # ── summary ───────────────────────────────────────────────────────────────
    rows: list[dict] = []
    with OUTPUT_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    if not rows:
        return

    mod_wins  = sum(1 for r in rows if r["winner"] == "modified")
    sdgca_wins = sum(1 for r in rows if r["winner"] == "sdgca")
    ties      = sum(1 for r in rows if r["winner"] == "tie")
    n_rows    = len(rows)

    def avg(field: str) -> float:
        vals = [float(r[field]) for r in rows if r[field] not in ("", "nan")]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total evaluations (dataset x method):  {n_rows}")
    print(f"modified wins:  {mod_wins:3d}  ({100*mod_wins/n_rows:.0f}%)")
    print(f"sdgca wins:     {sdgca_wins:3d}  ({100*sdgca_wins/n_rows:.0f}%)")
    print(f"ties:           {ties:3d}  ({100*ties/n_rows:.0f}%)")
    print()
    print(f"{'Metric':<20} {'sdgca':>8} {'modified':>10} {'delta':>8}")
    print("-" * 50)
    for metric, base_col, mod_col in [
        ("NMI mean",   "sdgca_nmi", "mod_nmi"),
        ("ARI mean",   "sdgca_ari", "mod_ari"),
        ("F-score mean","sdgca_f",  "mod_f"),
    ]:
        b = avg(base_col)
        m = avg(mod_col)
        print(f"{metric:<20} {b:8.4f} {m:10.4f} {m-b:+8.4f}")


if __name__ == "__main__":
    main()
