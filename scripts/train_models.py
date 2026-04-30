#!/usr/bin/env python
"""Train all models for all target horizons and evaluate against baselines.

Steps:
    1. Load processed dataset and configs
    2. Apply temporal split (train / val / test)
    3. For each target (enso_t1, enso_t3, enso_t6):
       a. Train all enabled models
       b. Evaluate on test set
       c. Compare against climatology and persistence baselines
    4. Print comparison table
    5. Save metrics to outputs/metrics/

Run from repo root:
    python scripts/train_models.py
    python scripts/train_models.py --target enso_t3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_all
from src.utils.io import read_parquet
from src.feature_engineering.builder import get_feature_columns
from src.validation.splits import time_split
from src.modeling.trainer import ModelTrainer
from src.modeling.baselines import ClimatologyBaseline, PersistenceBaseline
from src.evaluation.metrics import compare, to_dataframe, save


def parse_args():
    p = argparse.ArgumentParser(description="Train ENSO phase prediction models")
    p.add_argument(
        "--target",
        default=None,
        help="Single target to train: enso_t1 | enso_t3 | enso_t6. "
             "If omitted, trains all three.",
    )
    p.add_argument(
        "--dataset",
        default="data/processed/enso_dataset.parquet",
    )
    p.add_argument(
        "--config-dir",
        default="configs",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
    )
    return p.parse_args()


def prepare_split(df, cfg):
    """Apply the temporal split from modeling.yaml."""
    val_cfg = cfg["modeling"]["validation"]
    return time_split(
        df,
        train_end  = val_cfg["train_end"],
        test_start = val_cfg["test_start"],
        val_start  = val_cfg.get("val_start"),
        val_end    = val_cfg.get("val_end"),
    )


def get_feature_cols(df):
    """Feature columns = everything except targets and enso_phase."""
    exclude = {"enso_phase", "enso_t1", "enso_t3", "enso_t6"}
    return [c for c in get_feature_columns(df) if c not in exclude]


def train_and_evaluate(df, split, target, cfg, output_dir):
    """Train all models for one target and evaluate on the test set.

    Returns a dict of {model_name: metrics}.
    """
    feature_cols = get_feature_cols(df)

    # ── Prepare train / test matrices ────────────────────────────────────────
    train_rows = split.train.dropna(subset=feature_cols + [target])
    test_rows  = split.test.dropna(subset=[target])

    X_train = train_rows[feature_cols]
    y_train = train_rows[target]
    X_test  = test_rows[feature_cols].ffill().fillna(0)
    y_test  = test_rows[target]

    print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Train class dist: {y_train.value_counts().to_dict()}")

    # ── Train models ─────────────────────────────────────────────────────────
    models_dir = output_dir / "models" / target
    predictions = {}

    for model_name, model_cfg in cfg["modeling"]["models"].items():
        if not model_cfg.get("enabled", True):
            continue

        trainer = ModelTrainer(
            model_name=model_name,
            params=model_cfg["params"],
        )
        trainer.fit(X_train, y_train)
        predictions[model_name] = trainer.predict(X_test)
        trainer.save(models_dir / f"{model_name}.joblib")

    # ── Baselines ─────────────────────────────────────────────────────────────
    if cfg["modeling"]["baselines"].get("climatology", True):
        clim = ClimatologyBaseline().fit(y_train)
        predictions["climatology"] = clim.predict(len(y_test))

    if cfg["modeling"]["baselines"].get("persistence", True):
        if "enso_phase" in test_rows.columns:
            predictions["persistence"] = PersistenceBaseline().predict(
                test_rows["enso_phase"]
            )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = compare(y_test, predictions, target=target)
    return results


def main():
    args    = parse_args()
    cfg     = load_all(args.config_dir)
    df      = read_parquet(args.dataset)
    out_dir = Path(args.output_dir)

    # Restore DatetimeIndex if parquet stored it as a column
    if "date" in df.columns:
        df = df.set_index("date")

    targets = (
        [args.target] if args.target
        else cfg["modeling"]["targets"]
    )

    print("=" * 60)
    print("  ENSO Model Training Pipeline")
    print("=" * 60)
    print(f"\n  Dataset:  {args.dataset}")
    print(f"  Targets:  {targets}")

    split = prepare_split(df, cfg)

    # ── Train and evaluate each target ────────────────────────────────────────
    all_results = {}
    for target in targets:
        print(f"\n{'─'*60}")
        print(f"  Target: {target}")
        print(f"{'─'*60}")
        all_results[target] = train_and_evaluate(df, split, target, cfg, out_dir)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics" / "results.json"
    save(all_results, metrics_path)

    # ── Print comparison table ────────────────────────────────────────────────
    df_res = to_dataframe(all_results)

    print(f"\n{'='*60}")
    print("  Results summary (test set)")
    print(f"{'='*60}\n")
    print(df_res.to_string(index=False))

    # ── Highlight if any model beats all baselines ────────────────────────────
    print()
    baseline_names = {"climatology", "persistence"}
    for target in targets:
        target_rows = df_res[df_res["target"] == target]
        baseline_f1 = target_rows[
            target_rows["model"].isin(baseline_names)
        ]["f1_macro"].max()
        best_model = target_rows[
            ~target_rows["model"].isin(baseline_names)
        ].nlargest(1, "f1_macro")

        if not best_model.empty:
            best_name = best_model["model"].iloc[0]
            best_f1   = best_model["f1_macro"].iloc[0]
            beats = "✓ beats baselines" if best_f1 > baseline_f1 else "✗ does not beat baselines"
            print(f"  {target}: best model = {best_name} "
                  f"(F1={best_f1:.3f}) | baseline max={baseline_f1:.3f} | {beats}")

    print()


if __name__ == "__main__":
    main()
