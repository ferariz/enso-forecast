#!/usr/bin/env python
"""Walk-forward cross-validation for ENSO phase prediction.

Produces fold-level F1 scores, mean ± std across folds, and a
paired t-test comparing each model against the persistence baseline.

This is the evaluation protocol required for statistical significance
claims in a scientific publication.

Usage
-----
    python scripts/run_walk_forward_cv.py
    python scripts/run_walk_forward_cv.py --target enso_t6
    python scripts/run_walk_forward_cv.py --test-window 24 --gap 6

Output
------
    outputs/metrics/cv_results.json   — fold-level results
    outputs/metrics/cv_summary.json   — mean ± std, significance tests
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_all
from src.utils.io import read_parquet
from src.feature_engineering.builder import get_feature_columns
from src.modeling.trainer import ModelTrainer
from src.modeling.baselines import PersistenceBaseline
from src.validation.splits import walk_forward_cv

LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward CV for ENSO prediction")
    p.add_argument("--dataset",     default="data/processed/enso_dataset.parquet")
    p.add_argument("--config-dir",  default="configs")
    p.add_argument("--output-dir",  default="outputs")
    p.add_argument("--target",      default=None,
                   help="Single target (enso_t1/t3/t6). Default: all three.")
    p.add_argument("--cv-end",      default="2018-12",
                   help="Last month used in CV (held-out test excluded).")
    p.add_argument("--test-window", type=int, default=12,
                   help="Months per CV test fold. Default 12.")
    p.add_argument("--min-train",   type=int, default=120,
                   help="Minimum training months per fold. Default 120.")
    p.add_argument("--gap",         type=int, default=0,
                   help="Gap months between train end and test start. "
                        "Use --gap-per-horizon to set gap=h automatically.")
    p.add_argument("--gap-per-horizon", action="store_true", default=False,
                   help="Set gap=h for each horizon h (1,3,6) to prevent "
                        "label-overlap leakage. Overrides --gap.")
    p.add_argument("--step",        type=int, default=None,
                   help="Step months between folds. Default = test_window.")
    return p.parse_args()


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "enso_phase", "enso_t1", "enso_t3", "enso_t6",
        "nino34_t1", "nino34_t3", "nino34_t6",
    }
    return [c for c in get_feature_columns(df) if c not in exclude]


def evaluate_fold(
    fold,
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    cfg: dict,
) -> dict:
    """Train and evaluate all models on one CV fold."""
    train_rows = fold.train.dropna(subset=feature_cols + [target])
    test_rows  = fold.test.dropna(subset=[target])

    if len(train_rows) < 50 or len(test_rows) < 3:
        return None

    X_train = train_rows[feature_cols]
    y_train = train_rows[target]
    X_test  = test_rows[feature_cols].ffill().fillna(0)
    y_test  = test_rows[target]

    fold_results = {
        "fold":       fold.fold_index,
        "train_start": str(fold.train_start.date()),
        "train_end":   str(fold.train_end.date()),
        "test_start":  str(fold.test_start.date()),
        "test_end":    str(fold.test_end.date()),
        "n_train":    len(X_train),
        "n_test":     len(y_test),
        "models":     {},
    }

    # ── Trained models ────────────────────────────────────────────────────────
    for model_name, model_cfg in cfg["modeling"]["models"].items():
        if not model_cfg.get("enabled", True):
            continue
        trainer = ModelTrainer(model_name=model_name, params=model_cfg["params"])
        trainer.fit(X_train, y_train)
        preds = trainer.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro",
                      zero_division=0, labels=LABEL_ORDER)
        fold_results["models"][model_name] = round(float(f1), 4)

    # ── Persistence baseline ──────────────────────────────────────────────────
    if "enso_phase" in test_rows.columns:
        pers_preds = PersistenceBaseline().predict(test_rows["enso_phase"])
        pers_f1 = f1_score(y_test, pers_preds, average="macro",
                           zero_division=0, labels=LABEL_ORDER)
        fold_results["models"]["persistence"] = round(float(pers_f1), 4)

    return fold_results


def diebold_mariano_hac(scores_a: list, scores_b: list, max_lag: int = None) -> dict:
    """Diebold-Mariano test with HAC (Newey-West) variance estimator.

    Properly accounts for autocorrelation in fold-level loss differences,
    unlike a paired t-test which assumes i.i.d. differences.

    Reference: Diebold & Mariano (1995), "Comparing Predictive Accuracy".
    HAC variance: Newey & West (1987) with Bartlett kernel.

    Parameters
    ----------
    scores_a, scores_b : list[float]
        Per-fold scores for two models (higher = better).
    max_lag : int or None
        Maximum lag for Newey-West. Default = floor(n^(1/3)).

    Returns
    -------
    dict with dm_stat, p_value, mean_diff, hac_se, max_lag
    """
    d = np.array(scores_a) - np.array(scores_b)
    n = len(d)
    d_bar = np.mean(d)

    if max_lag is None:
        max_lag = int(np.floor(n ** (1/3)))

    # Newey-West HAC variance estimator with Bartlett kernel
    gamma_0 = np.mean((d - d_bar) ** 2)
    hac_var = gamma_0

    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1)  # Bartlett kernel
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        hac_var += 2 * weight * gamma_k

    hac_var = max(hac_var, 1e-12)  # numerical floor
    hac_se = np.sqrt(hac_var / n)

    dm_stat = d_bar / hac_se
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)

    return {
        "dm_stat":  round(float(dm_stat), 4),
        "p_value":  round(float(p_value), 6),
        "mean_diff": round(float(d_bar), 4),
        "hac_se":   round(float(hac_se), 4),
        "max_lag":  max_lag,
    }


def summarise(
    fold_results: list[dict],
    target: str,
) -> dict:
    """Compute mean ± std F1 and Diebold-Mariano (HAC) test vs persistence.

    Replaces the paired t-test, which assumes i.i.d. fold differences
    and can inflate significance when walk-forward folds are correlated.
    See Dietterich (1998) — who actually *warns against* the paired
    t-test on resampled folds — and Diebold & Mariano (1995).
    """
    model_names = [m for m in fold_results[0]["models"] if m != "persistence"]
    pers_scores = [f["models"].get("persistence", np.nan) for f in fold_results]

    summary = {"target": target, "n_folds": len(fold_results), "models": {}}

    for model_name in model_names + ["persistence"]:
        scores = [f["models"].get(model_name, np.nan) for f in fold_results]
        scores = [s for s in scores if not np.isnan(s)]

        entry = {
            "mean_f1":  round(float(np.mean(scores)), 4),
            "std_f1":   round(float(np.std(scores)),  4),
            "min_f1":   round(float(np.min(scores)),  4),
            "max_f1":   round(float(np.max(scores)),  4),
            "n_folds":  len(scores),
        }

        # Diebold-Mariano test with HAC variance vs persistence
        if model_name != "persistence" and len(pers_scores) == len(scores):
            dm = diebold_mariano_hac(scores, pers_scores)
            entry["dm_stat_vs_persistence"] = dm["dm_stat"]
            entry["dm_pval_vs_persistence"] = dm["p_value"]
            entry["dm_hac_se"]              = dm["hac_se"]
            entry["dm_max_lag"]             = dm["max_lag"]
            entry["significant_p05"]        = bool(dm["p_value"] < 0.05)
            entry["significant_p01"]        = bool(dm["p_value"] < 0.01)

        summary["models"][model_name] = entry

    return summary


def main():
    args = parse_args()
    cfg  = load_all(args.config_dir)
    df   = read_parquet(args.dataset)
    if "date" in df.columns:
        df = df.set_index("date")

    out_dir = Path(args.output_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets      = [args.target] if args.target else cfg["modeling"]["targets"]
    feature_cols = get_feature_cols(df)

    print("=" * 60)
    print("  Walk-Forward Cross-Validation")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Targets:     {targets}")
    print(f"  CV end:      {args.cv_end}")
    print(f"  Test window: {args.test_window} months")
    print(f"  Min train:   {args.min_train} months")
    print(f"  Gap:         {args.gap} months")
    print(f"  Features:    {len(feature_cols)}")
    print()

    # ── Per-horizon embargo mapping ─────────────────────────────────────────
    # For horizon h, the label at time t uses the ENSO phase at t+h.
    # Without a gap, the last h months of training have labels that overlap
    # with the test window — this is label-overlap leakage (M3).
    # Fix: set gap_months = h for each horizon.
    HORIZON_GAP = {"enso_t1": 1, "enso_t3": 3, "enso_t6": 6}

    all_fold_results = {}
    all_summaries    = {}

    for target in targets:
        # Determine gap for this target
        if args.gap_per_horizon:
            gap = HORIZON_GAP.get(target, args.gap)
            print(f"\n  [embargo] Using gap_months={gap} for {target} "
                  f"(--gap-per-horizon)")
        else:
            gap = args.gap

        # Generate folds with per-target gap
        folds = walk_forward_cv(
            df,
            cv_end           = args.cv_end,
            test_window      = args.test_window,
            min_train_months = args.min_train,
            gap_months       = gap,
            step_months      = args.step,
        )

        print(f"\n{'─'*60}")
        print(f"  Target: {target}  (gap={gap} months)")
        print(f"{'─'*60}")

        fold_results = []
        for fold in folds:
            result = evaluate_fold(fold, df, target, feature_cols, cfg)
            if result:
                fold_results.append(result)

        summary = summarise(fold_results, target)
        all_fold_results[target] = fold_results
        all_summaries[target]    = summary

        # Print summary table
        print(f"\n  {'Model':25s}  {'Mean F1':>8}  {'Std':>6}  "
              f"{'Min':>6}  {'Max':>6}  {'p-val':>8}  {'Sig?':>5}")
        print("  " + "─" * 70)
        for model_name, m in summary["models"].items():
            p_str  = f"{m.get('p_val_vs_persistence', float('nan')):.3f}" \
                     if "p_val_vs_persistence" in m else "  —  "
            sig    = "✓" if m.get("significant_p05") else " "
            print(f"  {model_name:25s}  {m['mean_f1']:>8.3f}  "
                  f"{m['std_f1']:>6.3f}  {m['min_f1']:>6.3f}  "
                  f"{m['max_f1']:>6.3f}  {p_str:>8}  {sig:>5}")

    # Save results
    cv_results_path = out_dir / "cv_results.json"
    cv_summary_path = out_dir / "cv_summary.json"

    with cv_results_path.open("w") as f:
        json.dump(all_fold_results, f, indent=2, default=str)
    with cv_summary_path.open("w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\n[cv] Fold results  → {cv_results_path}")
    print(f"[cv] CV summary    → {cv_summary_path}")
    print()


if __name__ == "__main__":
    main()
