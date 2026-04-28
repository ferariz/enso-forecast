"""Command-line interface for the ENSO forecast pipeline.

Usage
-----
    enso build-dataset
    enso train-model --target enso_t1
    enso export-kaggle
    enso run-all
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.panel import Panel

app = typer.Typer(
    name="enso",
    help="ENSO Early Phase Prediction — dataset construction and benchmarking pipeline",
    add_completion=False,
)

CONFIG_DIR = Path("configs")
RAW_DIR    = Path("data/raw")
INTERIM    = Path("data/interim")
PROCESSED  = Path("data/processed")
OUTPUTS    = Path("outputs")


@app.command("build-dataset")
def build_dataset(
    config_dir: Path = typer.Option(CONFIG_DIR, help="Directory containing YAML configs"),
    raw_dir:    Path = typer.Option(RAW_DIR,    help="Directory for raw cached files"),
    out_path:   Path = typer.Option(PROCESSED / "enso_dataset.parquet", help="Output parquet path"),
    skip_fetch: bool = typer.Option(False, help="Use cached raw files only (no network calls)"),
) -> None:
    """Fetch raw data, preprocess, engineer features, label, validate, save."""
    from src.utils.config import load_all_configs
    from src.utils.io import write_parquet
    from src.ingestion.registry import build_raw_dataset
    from src.preprocessing.cleaner import clean
    from src.labeling.enso_phase import label
    from src.feature_engineering.builder import build_features, get_feature_columns
    from src.validation.leakage_check import run_all_checks

    rprint(Panel("[bold cyan]🌊 Building ENSO dataset[/bold cyan]", expand=False))

    cfg = load_all_configs(config_dir)

    # 1. Ingest
    raw_df = build_raw_dataset(config=cfg["data_sources"], raw_dir=raw_dir)

    # 2. Preprocess
    tr = cfg["data_sources"].get("time_range", {})
    clean_df = clean(raw_df, start=tr.get("start"), end=tr.get("end"))

    # 3. Label (before feature engineering to avoid using future labels as features)
    labeled_df = label(clean_df)

    # 4. Feature engineering
    featured_df = build_features(labeled_df, config=cfg["features"])

    # 5. Leakage checks
    feature_cols = get_feature_columns(featured_df)
    checks = run_all_checks(featured_df, feature_cols)
    if not all(checks.values()):
        rprint("[bold red]⚠️  Leakage checks FAILED. Aborting.[/bold red]")
        raise typer.Exit(code=1)

    # 6. Save
    write_parquet(featured_df.reset_index(), out_path)
    rprint(f"[green]✓ Dataset saved → {out_path}[/green]")
    rprint(f"  Shape: {featured_df.shape[0]} rows × {featured_df.shape[1]} columns")


@app.command("train-model")
def train_model(
    target:     str  = typer.Option("enso_t3", help="Target column: enso_t1 | enso_t3 | enso_t6"),
    dataset:    Path = typer.Option(PROCESSED / "enso_dataset.parquet", help="Input dataset"),
    config_dir: Path = typer.Option(CONFIG_DIR),
    output_dir: Path = typer.Option(OUTPUTS),
) -> None:
    """Train all enabled models for the specified target and evaluate against baselines."""
    import pandas as pd
    from src.utils.config import load_all_configs
    from src.utils.io import read_parquet
    from src.feature_engineering.builder import get_feature_columns
    from src.validation.splits import time_split
    from src.modeling.trainer import train_all_models
    from src.modeling.baselines import ClimatologyBaseline, PersistenceBaseline
    from src.evaluation.metrics import evaluate_all, results_to_dataframe, save_metrics
    from src.evaluation.plots import plot_lead_time_comparison, plot_confusion_matrix

    rprint(Panel(f"[bold cyan]🤖 Training models for target: {target}[/bold cyan]", expand=False))

    cfg = load_all_configs(config_dir)
    df  = read_parquet(dataset).set_index("date") if "date" in pd.read_parquet(dataset).columns else read_parquet(dataset)

    val_cfg = cfg["modeling"]["validation"]
    split   = time_split(
        df,
        train_end  = val_cfg["train_end"],
        test_start = val_cfg["test_start"],
        val_start  = val_cfg.get("val_start"),
        val_end    = val_cfg.get("val_end"),
    )

    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop rows with NaN in features or target
    train_clean = split.train.dropna(subset=feature_cols + [target])
    test_clean  = split.test.dropna(subset=[target])

    X_train = train_clean[feature_cols]
    y_train = train_clean[target]
    X_test  = test_clean[feature_cols].fillna(0)
    y_test  = test_clean[target]

    # Train models
    trained = train_all_models(
        X_train, y_train,
        target=target,
        config=cfg["modeling"],
        output_dir=output_dir / "models",
    )

    # Collect predictions
    preds: dict[str, object] = {}
    for name, trainer in trained.items():
        preds[name] = trainer.predict(X_test)

    # Baselines
    clim = ClimatologyBaseline().fit(y_train)
    preds["climatology"] = clim.predict(len(y_test))

    pers = PersistenceBaseline()
    if "enso_phase" in test_clean.columns:
        preds["persistence"] = pers.predict(test_clean["enso_phase"])

    # Evaluate
    results = {target: evaluate_all(y_test, preds, target)}
    df_res  = results_to_dataframe(results)

    rprint(df_res.to_string(index=False))
    save_metrics(results, output_dir / "metrics" / f"{target}_results.json")

    # Plots
    figs_dir = output_dir / "figures"
    for name, model_results in results[target].items():
        if "confusion_matrix" in model_results:
            plot_confusion_matrix(
                model_results["confusion_matrix"],
                title=f"{target} / {name}",
                save_path=figs_dir / f"cm_{target}_{name}.png",
            )

    rprint(f"[green]✓ Training complete. Metrics → {output_dir / 'metrics'}[/green]")


@app.command("export-kaggle")
def export_kaggle(
    dataset:    Path = typer.Option(PROCESSED / "enso_dataset.parquet"),
    config_dir: Path = typer.Option(CONFIG_DIR),
    out_dir:    Path = typer.Option(Path("data/kaggle_export")),
) -> None:
    """Produce the clean Kaggle-ready dataset bundle."""
    import pandas as pd
    from src.utils.config import load_all_configs
    from src.utils.io import read_parquet
    from src.export.kaggle_exporter import export

    rprint(Panel("[bold cyan]📦 Exporting Kaggle dataset[/bold cyan]", expand=False))

    cfg = load_all_configs(config_dir)
    df  = read_parquet(dataset)
    if "date" in df.columns:
        df = df.set_index("date")

    export(df, config=cfg["export"], output_dir=out_dir)
    rprint(f"[green]✓ Kaggle export complete → {out_dir}[/green]")


@app.command("run-all")
def run_all(
    config_dir: Path = typer.Option(CONFIG_DIR),
) -> None:
    """Run the full pipeline: build → train (all targets) → export."""
    rprint(Panel("[bold magenta]🚀 Running full pipeline[/bold magenta]", expand=False))
    build_dataset(config_dir=config_dir)
    for target in ["enso_t1", "enso_t3", "enso_t6"]:
        train_model(target=target, config_dir=config_dir)
    export_kaggle(config_dir=config_dir)
    rprint("[bold green]✓ Full pipeline complete.[/bold green]")


if __name__ == "__main__":
    app()
