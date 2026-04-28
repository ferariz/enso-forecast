# 🌊 ENSO Early Phase Prediction — Dataset & Benchmarking Pipeline

> A reproducible, physically-informed backend for constructing, validating, and benchmarking datasets for early ENSO phase prediction.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Repository Philosophy](#repository-philosophy)
3. [Repository Structure](#repository-structure)
4. [Dataset Construction Pipeline](#dataset-construction-pipeline)
5. [Data Sources](#data-sources)
6. [Feature Engineering](#feature-engineering)
7. [ENSO Labeling](#enso-labeling)
8. [Temporal Validation Methodology](#temporal-validation-methodology)
9. [Models & Baselines](#models--baselines)
10. [Evaluation](#evaluation)
11. [Reproducing Results](#reproducing-results)
12. [Exporting to Kaggle](#exporting-to-kaggle)
13. [Configuration Reference](#configuration-reference)
14. [CLI Reference](#cli-reference)

---

## Problem Definition

The **El Niño–Southern Oscillation (ENSO)** is the dominant mode of interannual climate variability on Earth, with major consequences for precipitation, drought, agriculture, and extreme weather globally. Predicting ENSO phase *months in advance* is one of the most impactful applied problems in climate science.

**This repository frames ENSO prediction as a supervised classification task:**

| | |
|---|---|
| **Input** | Climate indices observed at time *t* |
| **Output** | ENSO phase at time *t + L* |
| **Horizons (L)** | 1, 3, and 6 months |
| **Classes** | El Niño · Neutral · La Niña |
| **Resolution** | Monthly |
| **Time range** | January 1980 – present |

Labels follow the **Oceanic Niño Index (ONI)** convention: a 3-month running mean of the Niño 3.4 SST anomaly exceeding ±0.5 °C.

---

## Repository Philosophy

This repo is **not** a Kaggle competition. It is a research backend with two separate concerns:

```
enso-forecast (this repo)              Kaggle
───────────────────────────────        ──────────────────────────────
• Data engineering & validation   →    • Clean tabular dataset (export)
• Research-grade feature work          • Simple baseline notebooks
• Model benchmarking pipeline          • Community exploration
• Leakage prevention & auditing
• Reproducible configuration
```

The pipeline enforces **strict temporal ordering** everywhere and provides automated leakage detection to prevent methodological errors that are common in climate ML work.

---

## Repository Structure

```
enso-forecast/
├── configs/
│   ├── data_sources.yaml     # URLs, filenames, time range
│   ├── features.yaml         # Which variables and transformations to apply
│   ├── modeling.yaml         # Train/val/test splits, model hyperparameters
│   └── export.yaml           # Kaggle export settings, column renames
│
├── data/
│   ├── raw/                  # Unmodified files as downloaded from providers
│   ├── interim/              # Intermediate representations (merged raw)
│   ├── processed/            # Final feature-engineered dataset
│   └── kaggle_export/        # Ready-to-upload Kaggle dataset bundle
│
├── src/
│   ├── ingestion/            # Per-source loaders (NOAA CPC, BOM)
│   ├── preprocessing/        # Time-range filtering, gap handling, imputation
│   ├── labeling/             # ENSO phase labeling & target generation
│   ├── feature_engineering/  # Physically-informed feature construction
│   ├── validation/           # Temporal splits, leakage detection
│   ├── modeling/             # Trainer, baselines (climatology, persistence)
│   ├── evaluation/           # Metrics, confusion matrices, SHAP, plots
│   ├── export/               # Kaggle export pipeline
│   ├── utils/                # Config loading, I/O helpers, logging
│   └── cli.py                # Typer CLI (enso build-dataset, train-model, …)
│
├── notebooks/
│   ├── research/
│   │   └── 01_dataset_eda.ipynb
│   └── validation_checks/
│       └── 01_leakage_audit.ipynb
│
├── scripts/
│   ├── build_dataset.py
│   ├── run_training.py
│   └── export_kaggle_dataset.py
│
├── tests/
│   ├── test_labeling.py
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_leakage.py
│
├── outputs/
│   ├── models/               # Serialised .joblib model files
│   ├── metrics/              # JSON metric reports per target
│   └── figures/              # Confusion matrices, lead-time curves, SHAP plots
│
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Dataset Construction Pipeline

The full pipeline has five sequential stages:

```
[1] Ingest          [2] Preprocess       [3] Label          [4] Features        [5] Validate
─────────────       ──────────────────   ─────────────────  ──────────────────  ─────────────────
Fetch / cache       Filter 1980–present  ONI-convention     Lags, rolling       Leakage checks
NOAA CPC data  →    Regularise to   →    ENSO phase at  →   stats, diffs,   →   Index monotonic
(Niño indices,      monthly MS grid      current t          calendar features   No future info
 SOI, zonal wind,   Forward-fill                            MJO sin/cos         Target alignment
 optional MJO)      short gaps           Shift targets                          verified
                                         for horizons
                                         t+1, t+3, t+6
```

Each stage is independently testable and config-driven.

---

## Data Sources

All sources are freely available from NOAA CPC and BOM (no API keys required).

| Variable | Source | Physical meaning |
|---|---|---|
| **Niño 3.4 anomaly** | NOAA CPC (ERSSTv5) | Primary ENSO diagnostic: SST anomaly averaged 5°S–5°N, 120°W–170°W |
| **Niño 1+2 anomaly** | NOAA CPC | Far eastern Pacific SST — precursor to developing El Niño |
| **Niño 3 anomaly** | NOAA CPC | Central/eastern Pacific SST |
| **Niño 4 anomaly** | NOAA CPC | Western Pacific warm pool — La Niña precursor |
| **SOI** | NOAA CPC | Normalised pressure difference Tahiti − Darwin. Negative SOI → weakened trade winds → El Niño |
| **850 hPa zonal wind** | NOAA CPC | Equatorial Walker circulation strength. Westerly anomalies → Kelvin wave propagation |
| **MJO (RMM1/RMM2)** | BOM Australia | *(optional)* Madden–Julian Oscillation — sub-seasonal precursor to ENSO |

Raw files are stored **unchanged** in `data/raw/` and never overwritten once cached. Re-running the pipeline with `--skip-fetch` uses the cached copy.

---

## Feature Engineering

All features are **strictly backward-looking**. No transformation uses any information from time *t+1* or beyond.

### For each base variable

| Feature type | Formula | Rationale |
|---|---|---|
| **Current value** | `x_t` | Instantaneous state |
| **Lag 1** | `x_{t-1}` | Recent memory |
| **Lag 3** | `x_{t-3}` | Seasonal memory |
| **Lag 6** | `x_{t-6}` | ENSO persistence timescale |
| **3-month rolling mean** | `mean(x_{t-2}, x_{t-1}, x_t)` | Smooths noise; approximates ONI for Niño 3.4 |
| **3-month rolling std** | `std(x_{t-2}, x_{t-1}, x_t)` | Variability proxy |
| **Diff(1)** | `x_t - x_{t-1}` | Month-over-month tendency / rate of change |

### Calendar features

The seasonal cycle is critical for ENSO predictability (boreal spring barrier). It is encoded as:

```
month_sin = sin(2π · month / 12)
month_cos = cos(2π · month / 12)
```

This preserves the circular continuity of the calendar (December → January is not a jump).

### MJO encoding (optional)

The MJO phase (1–8) is encoded on the unit circle to avoid spurious ordinality:

```
mjo_sin = sin(2π · (phase - 1) / 8)    # active MJO only (amplitude ≥ 1.0)
mjo_cos = cos(2π · (phase - 1) / 8)
```

### Leakage prevention

The automated `leakage_check` module verifies:
- No lag column has a non-positive index
- No target column appears in the feature set
- The time index is strictly monotonically increasing
- No feature has suspiciously perfect correlation with a future target

---

## ENSO Labeling

Labels follow the **ONI convention** (NOAA Climate Prediction Center):

```
smoothed_t = rolling_mean(nino34_anom, window=3)   # backward-looking

enso_phase_t =  "El Niño"  if smoothed_t >  0.5 °C
                "La Niña"  if smoothed_t < -0.5 °C
                "Neutral"  otherwise
```

Prediction targets at lead time *L* are generated by shifting the smoothed series:

```python
future_smoothed = smoothed.shift(-L)          # pull future value into current row
enso_tL = phase_from_value(future_smoothed)   # apply same thresholds
```

This correctly represents "what ENSO phase will be observed L months from now" without any information leakage into the feature window.

---

## Temporal Validation Methodology

> ⚠️ **All splits respect strict time ordering.** Random or stratified splitting would constitute temporal data leakage and produce wildly optimistic evaluation metrics.

### Fixed time split

```
────────────────────────────────────────────────────────────────
  TRAIN               VAL               TEST
  1980–2015           2016–2018         2019–present
────────────────────────────────────────────────────────────────
```

The **test set is held out completely** until final evaluation. Model selection and hyperparameter choices are made based on validation performance only.

### Walk-forward validation (optional)

For more robust estimates, especially important for publication:

```
Fold 1:  Train [1980–1994]  Eval [1995]
Fold 2:  Train [1980–1995]  Eval [1996]
...
Fold N:  Train [1980–2008]  Eval [2009]
```

Each fold expands the training window by 12 months and evaluates on the next 12 months. This mimics real-world operational forecasting.

---

## Models & Baselines

### Baselines (minimum bar to beat)

| Baseline | Description |
|---|---|
| **Climatology** | Always predicts the most frequent class in training data |
| **Persistence** | Predicts ENSO_t = ENSO_{t+L} (no change) |

Persistence is deceptively strong at L=1 (ENSO phases last ~6–18 months). A model that cannot beat persistence at short lead times is not useful.

### Models

| Model | Why it's here |
|---|---|
| **Logistic Regression** | Linear baseline; interpretable; fast |
| **Random Forest** | Non-linear; captures interaction effects; robust |
| **LightGBM** | State-of-the-art tabular model; handles correlated features well |

All models use `class_weight='balanced'` to account for class imbalance (Neutral is over-represented). All are trained on the same feature set with deterministic seeds.

---

## Evaluation

### Metrics

| Metric | Why |
|---|---|
| **Accuracy** | Simple, interpretable overall performance |
| **F1 macro** | Equally weights all three classes regardless of frequency — critical for imbalanced ENSO labels |
| **Per-class F1** | Reveals whether El Niño and La Niña events (rarer but more important) are well-predicted |
| **Confusion matrix** | Shows systematic misclassification patterns |

### Lead-time performance curves

Performance is evaluated at all three horizons (t+1, t+3, t+6) and plotted together to understand skill decay with lead time. This is the primary research output.

### SHAP feature importance

For LightGBM, SHAP TreeExplainer values are computed to identify:
- Which climate indices contribute most to predictions
- Whether the model has learned physically meaningful patterns
- Lead-time-specific feature importance shifts

---

## Reproducing Results

### 1. Install

```bash
git clone https://github.com/your-org/enso-forecast
cd enso-forecast
pip install -e ".[dev]"
```

### 2. Full pipeline (one command)

```bash
make run-all
# or: enso run-all
```

### 3. Step by step

```bash
# Fetch data, preprocess, label, engineer features, validate
enso build-dataset

# Train all models for all horizons
enso train-model --target enso_t1
enso train-model --target enso_t3
enso train-model --target enso_t6

# Export Kaggle dataset bundle
enso export-kaggle
```

### 4. Run tests

```bash
make test
# or: pytest tests/ -v
```

### Determinism

- All random seeds set via `random_state=42` in model configs
- No random splits anywhere in the pipeline
- Configs are version-controlled and pinned
- Raw data files are cached and never overwritten

---

## Exporting to Kaggle

The export pipeline produces a self-contained bundle in `data/kaggle_export/`:

```
data/kaggle_export/
├── enso_train.parquet       # Training set (1980–2018)
├── enso_test.parquet        # Test set (2019–present), targets included for validation
├── data_dictionary.csv      # Column descriptions, dtypes, missing value counts
└── metadata.json            # Dataset provenance, label encoding, date ranges
```

Column names are renamed for clarity (e.g. `nino34_anom` → `sst_anom_nino34`). Integer-encoded targets (`enso_t1_int`, etc.) are included alongside string labels for flexibility.

To regenerate after any pipeline change:
```bash
enso export-kaggle
```

---

## Configuration Reference

| File | Controls |
|---|---|
| `configs/data_sources.yaml` | Source URLs, raw filenames, time range |
| `configs/features.yaml` | Which variables to transform, lag depths, rolling windows |
| `configs/modeling.yaml` | Train/val/test dates, model hyperparameters, SHAP settings |
| `configs/export.yaml` | Kaggle column renames, label encoding, file format |

Edit these files to change the pipeline without touching source code.

---

## CLI Reference

```bash
enso --help

Commands:
  build-dataset   Fetch + preprocess + label + features + validate
  train-model     Train all models for one target, evaluate vs baselines
  export-kaggle   Produce Kaggle-ready dataset bundle
  run-all         Full pipeline in one shot
```

---

## License

MIT. Data from NOAA CPC and BOM is subject to their respective public data policies.
