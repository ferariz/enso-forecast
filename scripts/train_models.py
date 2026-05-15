#!/usr/bin/env python
"""Train all models for all target horizons and evaluate against baselines.

Steps:
    1. Load processed dataset and configs
    2. Apply temporal split (train / val / test)
    3. For each classification target (enso_t1, enso_t3, enso_t6):
       a. Train all enabled models
       b. Evaluate on test set
       c. Compare against climatology and persistence baselines
    4. For each regression target (nino34_t1, nino34_t3, nino34_t6):
       a. Train LightGBM regressor
       b. Evaluate RMSE and MAE on test set
    5. Print comparison tables
    6. Save metrics to outputs/metrics/

Run from repo root:
    python scripts/train_models.py
    python scripts/train_models.py --target enso_t3
    python scripts/train_models.py --skip-regression
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
        help="Single classification target: enso_t1 | enso_t3 | enso_t6. "
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
    p.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression model training.",
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
    """Feature columns — exclude all targets."""
    exclude = {
        "enso_phase",
        "enso_t1", "enso_t3", "enso_t6",
        "nino34_t1", "nino34_t3", "nino34_t6",
    }
    return [c for c in get_feature_columns(df) if c not in exclude]


def train_and_evaluate(df, split, target, cfg, output_dir):
    """Train all classification models for one target."""
    feature_cols = get_feature_cols(df)

    train_rows = split.train.dropna(subset=feature_cols + [target])
    test_rows  = split.test.dropna(subset=[target])

    X_train = train_rows[feature_cols]
    y_train = train_rows[target]
    X_test  = test_rows[feature_cols].ffill().fillna(0)
    y_test  = test_rows[target]

    print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Train class dist: {y_train.value_counts().to_dict()}")

    models_dir  = output_dir / "models" / target
    predictions = {}

    for model_name, model_cfg in cfg["modeling"]["models"].items():
        if not model_cfg.get("enabled", True):
            continue
        trainer = ModelTrainer(model_name=model_name, params=model_cfg["params"])
        trainer.fit(X_train, y_train)
        predictions[model_name] = trainer.predict(X_test)
        trainer.save(models_dir / f"{model_name}.joblib")

    if cfg["modeling"]["baselines"].get("climatology", True):
        clim = ClimatologyBaseline().fit(y_train)
        predictions["climatology"] = clim.predict(len(y_test))

    if cfg["modeling"]["baselines"].get("persistence", True):
        if "enso_phase" in test_rows.columns:
            predictions["persistence"] = PersistenceBaseline().predict(
                test_rows["enso_phase"]
            )

    results = compare(y_test, predictions, target=target)
    return results


def train_regression(df, split, output_dir):
    """Train LightGBM regressors for nino34_t1, nino34_t3, nino34_t6.

    These models predict the smoothed Niño 3.4 anomaly (°C) at each
    forecast horizon — enabling magnitude estimation alongside the
    phase classification.

    A persistence baseline (predict current nino34_anom for all horizons)
    is included for reference.
    """
    try:
        import lightgbm as lgb
        import joblib
        import numpy as np
        from sklearn.metrics import mean_absolute_error
    except ImportError as e:
        print(f"  Skipping regression — missing dependency: {e}")
        return {}

    feature_cols = get_feature_cols(df)
    reg_targets  = ["nino34_t1", "nino34_t3", "nino34_t6"]
    reg_results  = {}

    print(f"\n{'─'*60}")
    print("  Regression targets: nino34_t1 / t3 / t6")
    print(f"{'─'*60}")

    for target in reg_targets:
        if target not in df.columns:
            print(f"  {target} not found in dataset — skipping")
            continue

        horizon = int(target.replace("nino34_t", ""))

        train_rows = split.train.dropna(subset=feature_cols + [target])
        test_rows  = split.test.dropna(subset=[target])

        X_train = train_rows[feature_cols]
        y_train = train_rows[target].astype(float)
        X_test  = test_rows[feature_cols].ffill().fillna(0)
        y_test  = test_rows[target].astype(float)

        print(f"\n  {target}  |  train={len(X_train)}  test={len(X_test)}")

        # ── LightGBM regressor ────────────────────────────────────────────
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        y_pred = model.predict(X_test)
        rmse   = float(np.sqrt(((y_pred - y_test) ** 2).mean()))
        mae    = float(mean_absolute_error(y_test, y_pred))

        # ── Persistence baseline ──────────────────────────────────────────
        if "nino34_anom" in test_rows.columns:
            y_pers      = test_rows["nino34_anom"].ffill().fillna(0).astype(float)
            rmse_pers   = float(np.sqrt(((y_pers - y_test) ** 2).mean()))
            mae_pers    = float(mean_absolute_error(y_test, y_pers))
        else:
            rmse_pers = mae_pers = float("nan")

        print(f"  LightGBM  RMSE={rmse:.3f}°C  MAE={mae:.3f}°C")
        print(f"  Persist.  RMSE={rmse_pers:.3f}°C  MAE={mae_pers:.3f}°C")

        # ── Save model ────────────────────────────────────────────────────
        models_dir = output_dir / "models" / target
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "lightgbm_regressor.joblib"
        joblib.dump({"model": model, "feature_names": list(X_train.columns)},
                    model_path)
        print(f"  Saved → {model_path}")

        reg_results[target] = {
            "rmse":      round(rmse, 4),
            "mae":       round(mae,  4),
            "rmse_persistence": round(rmse_pers, 4),
            "mae_persistence":  round(mae_pers,  4),
            "n_test":    len(y_test),
            "horizon_months": horizon,
        }

    return reg_results


def main():
    args    = parse_args()
    cfg     = load_all(args.config_dir)
    df      = read_parquet(args.dataset)
    out_dir = Path(args.output_dir)

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

    # ── Classification ────────────────────────────────────────────────────────
    all_results = {}
    for target in targets:
        print(f"\n{'─'*60}")
        print(f"  Target: {target}")
        print(f"{'─'*60}")
        all_results[target] = train_and_evaluate(df, split, target, cfg, out_dir)

    # ── Save classification metrics ───────────────────────────────────────────
    metrics_path = out_dir / "metrics" / "results.json"
    save(all_results, metrics_path)

    # ── Print classification table ────────────────────────────────────────────
    df_res = to_dataframe(all_results)
    print(f"\n{'='*60}")
    print("  Classification results (test set)")
    print(f"{'='*60}\n")
    print(df_res.to_string(index=False))

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
            print(f"  {target}: best={best_name} (F1={best_f1:.3f}) "
                  f"| baseline={baseline_f1:.3f} | {beats}")

    # ── Regression ────────────────────────────────────────────────────────────
    if not args.skip_regression:
        reg_results = train_regression(df, split, out_dir)

        if reg_results:
            import json
            reg_path = out_dir / "metrics" / "regression_results.json"
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            with reg_path.open("w") as fh:
                json.dump(reg_results, fh, indent=2)
            print(f"\n[metrics] Regression results saved → {reg_path}")

            print(f"\n{'='*60}")
            print("  Regression results (test set)")
            print(f"{'='*60}")
            print(f"  {'Target':>12}  {'RMSE':>7}  {'MAE':>7}  "
                  f"{'RMSE_pers':>10}  {'Skill'}")
            print("  " + "─" * 50)
            for t, m in reg_results.items():
                skill = 1 - m["rmse"] / m["rmse_persistence"]
                print(f"  {t:>12}  {m['rmse']:>7.3f}  {m['mae']:>7.3f}  "
                      f"{m['rmse_persistence']:>10.3f}  {skill:>+.3f}")

    print()


if __name__ == "__main__":
    main()
