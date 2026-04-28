"""Evaluation visualisations: confusion matrices, lead-time curves, SHAP."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

LABEL_ORDER = ["La Niña", "Neutral", "El Niño"]
PALETTE = {"La Niña": "#4393c3", "Neutral": "#f7f7f7", "El Niño": "#d6604d"}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_confusion_matrix(
    cm: list[list[int]],
    title: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix as a heatmap."""
    cm_arr = np.array(cm, dtype=float)
    cm_norm = cm_arr / cm_arr.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=cm_arr.astype(int),
        fmt="d",
        cmap="RdBu_r",
        vmin=0, vmax=1,
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title or "Confusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Confusion matrix saved → {save_path}")
    return fig


def plot_lead_time_comparison(
    results_df: pd.DataFrame,
    metric: str = "f1_macro",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot model and baseline performance across lead times.

    Parameters
    ----------
    results_df:
        Output of ``evaluation.metrics.results_to_dataframe()``.
    metric:
        Column to plot on the y-axis.
    """
    # Map target names to numeric lead times
    lead_map = {"enso_t1": 1, "enso_t3": 3, "enso_t6": 6}
    df = results_df.copy()
    df["lead_months"] = df["target"].map(lead_map)
    df = df.dropna(subset=["lead_months"])

    fig, ax = plt.subplots(figsize=(7, 4))

    linestyles = {
        "climatology":          ("--", "gray",    "o"),
        "persistence":          (":",  "dimgray", "s"),
        "logistic_regression":  ("-",  "#2196F3", "^"),
        "random_forest":        ("-",  "#4CAF50", "D"),
        "lightgbm":             ("-",  "#FF5722", "*"),
    }

    for model, grp in df.groupby("model"):
        grp = grp.sort_values("lead_months")
        ls, color, marker = linestyles.get(model, ("-", "black", "o"))
        ax.plot(
            grp["lead_months"],
            grp[metric],
            linestyle=ls,
            color=color,
            marker=marker,
            markersize=7,
            label=model,
            linewidth=1.8,
        )

    ax.set_xticks([1, 3, 6])
    ax.set_xlabel("Lead time (months)", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title("Model performance vs lead time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Lead-time comparison saved → {save_path}")
    return fig


def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    class_names: list[str],
    save_dir: Path | None = None,
) -> None:
    """Plot SHAP beeswarm summary for each class."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plots")
        return

    for i, cls in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values[i],
            X,
            show=False,
            max_display=20,
            plot_type="dot",
        )
        plt.title(f"SHAP summary — {cls}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / f"shap_{cls.replace(' ', '_').replace('ñ', 'n')}.png"
            fig.savefig(p)
            logger.info(f"SHAP plot saved → {p}")
        plt.close(fig)


def plot_class_distribution(
    df: pd.DataFrame,
    col: str = "enso_phase",
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of ENSO phase frequencies in the dataset."""
    counts = df[col].value_counts().reindex(LABEL_ORDER).fillna(0)
    colors = [PALETTE[l] for l in LABEL_ORDER]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(LABEL_ORDER, counts.values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("ENSO Phase Distribution", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig
