# 🌊 ENSO Early Phase Prediction — Dataset & Benchmarking Pipeline

> **Dataset:** [ferariz/enso-early-phase-prediction](https://www.kaggle.com/datasets/ferariz/enso-early-phase-prediction)  
> **GitHub:** [ferariz/enso-forecast](https://github.com/ferariz/enso-forecast)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Backend pipeline for constructing, validating, and benchmarking a monthly ENSO phase prediction dataset. Ingests public NOAA CPC indices, engineers physically-informed features, enforces strict temporal validation, and exports a clean tabular dataset for Kaggle.

---

## What is ENSO?

The El Niño–Southern Oscillation (ENSO) is the dominant mode of interannual climate variability on Earth. It alternates between three phases:

| Phase | Niño 3.4 anomaly | Global impact |
|---|---|---|
| **El Niño** | > +0.5 °C | Drought in Australia/SE Asia, flooding in South America |
| **La Niña** | < −0.5 °C | Intensified trade winds, above-normal Atlantic hurricane seasons |
| **Neutral** | ±0.5 °C | Near-average conditions |

---

## Task

Given monthly climate indices observed at time **t**, predict the ENSO phase at:
- **t+1** — 1 month ahead
- **t+3** — 3 months ahead
- **t+6** — 6 months ahead

Targets are provided for both **classification** (El Niño / Neutral / La Niña) and **regression** (smoothed Niño 3.4 anomaly in °C).

**Primary metric:** Macro F1 (weights all three classes equally)  
**Time range:** January 1980 – present  
**Resolution:** Monthly

---

## Benchmark results

Evaluated on held-out test set (2019–2026), which includes the strong 2020–2023 triple-dip La Niña:

| Horizon | Best model | F1 macro | Persistence baseline |
|---|---|---|---|
| **t+1** | LightGBM | **0.945** | 0.858 |
| **t+3** | Logistic Regression | **0.802** | 0.610 |
| **t+6** | Logistic Regression | **0.608** | 0.419 |

LR outperforms LightGBM at t+3 and t+6 — simpler models generalise better at longer horizons on tabular climate data.

**Regression target (nino34_t3):** RMSE = 0.25°C — competitive with operational dynamical forecasts at this range.

---

## Key finding: the Spring Predictability Barrier

ENSO forecast skill is strongly modulated by the initialization month:

- **June–November inits** (post-barrier growth phase): F1 ~0.71–0.75 at t+6
- **February–March inits** (approaching the barrier): F1 ~0.15

The dataset includes two engineered features that capture this:
- `crosses_spring_tL` — does the forecast window pass through boreal spring (MAM)?
- `init_in_growth_phase` — is initialization in the high-skill JJA–SON window?

---

## Repository structure

```
scripts/
  build_dataset.py          # full pipeline: ingest → preprocess → label → features → validate
  train_models.py           # train all models, evaluate vs baselines
  export_kaggle_dataset.py  # produce Kaggle-ready dataset bundle

src/
  ingestion/                # NOAA CPC loaders: Niño indices, SOI, zonal wind
  preprocessing/            # time filter, monthly grid, gap imputation
  labeling/                 # ONI-convention phase labeling, target generation
  feature_engineering/      # lags, rolling stats, spring barrier features
  validation/               # temporal splits, automated leakage checks
  modeling/                 # LR, RF, LightGBM + climatology/persistence baselines
  evaluation/               # metrics, spring barrier stratification, SHAP
  export/                   # Kaggle export pipeline

notebooks/
  research/
    enso_starter_local.ipynb    # full analysis notebook (local paths)
    enso_starter_kaggle.ipynb   # Kaggle upload version

data/
  raw/                      # unmodified NOAA files (cached after first run)
  processed/                # enso_dataset.parquet (564 rows × 61 columns)
  kaggle_export/            # clean tabular export for Kaggle

outputs/
  models/                   # trained .joblib files (gitignored)
  metrics/results.json      # benchmark results (tracked)
```

---

## Quick start

```bash
git clone https://github.com/ferariz/enso-forecast
cd enso-forecast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Build dataset (downloads ~3 NOAA files on first run, uses cache after)
python scripts/build_dataset.py

# 2. Train models and evaluate
python scripts/train_models.py

# 3. Export Kaggle dataset bundle
python scripts/export_kaggle_dataset.py
```

---

## Data sources

All sources are freely available — no API keys required.

| Variable | Provider | Physical meaning |
|---|---|---|
| Niño 3.4 anomaly | NOAA CPC (ERSSTv5) | Primary ENSO diagnostic |
| Niño 1+2, 3, 4 anomalies | NOAA CPC | Eastern, central, western Pacific SST |
| SOI | NOAA CPC | Normalised pressure difference Tahiti − Darwin |
| 850 hPa zonal wind | NOAA CPC | Walker circulation strength |

---

## Features

54 features, all **strictly backward-looking** (no future leakage):

| Type | Example | Captures |
|---|---|---|
| Current value | `sst_anom_nino34` | Instantaneous state |
| Lags 1, 3, 6m | `sst_anom_nino34_lag3` | Memory at seasonal timescales |
| 3m rolling mean | `sst_anom_nino34_rm3` | Smoothed state ≈ ONI index |
| 3m rolling std | `sst_anom_nino34_rstd3` | Signal strengthening? |
| 1m diff | `sst_anom_nino34_diff1` | Rate of change |
| Spring barrier | `crosses_spring_t6` | Does forecast cross MAM? |
| Growth phase | `init_in_growth_phase` | Is init in JJA–SON? |

---

## Validation methodology

All splits respect strict time ordering — no random shuffling.

```
Train            Validation       Test (held out)
1980 → 2015      2016 → 2018      2019 → present
```

Automated leakage checks run on every dataset build:
- Index is monotonically increasing
- No feature encodes a negative lag (future data)
- No target column in the feature set
- `enso_t1[i]` equals `enso_phase[i+1]` (shift correctness)

---

## Live analysis

### 🌊 ENSO 2026: Is El Niño developing?
As of April 2026, the tropical Pacific is showing early El Niño signals:
ocean warming (+0.23°C) while the atmosphere remains in a La Niña-like
pattern (SOI +2.0). The spring predictability barrier is active.

See the analysis notebook:
[enso_2026_forecast.ipynb](notebooks/research/enso_2026_forecast.ipynb)

**Update planned for August 2026** — when post-barrier skill jumps from
F1 ~0.43 to ~0.75 and the model predictions become meaningfully reliable.

---

## Roadmap

- [ ] MJO features (BOM RMM index)
- [ ] Thermocline depth (D20 index)
- [ ] Walk-forward cross-validation
- [ ] t+9 month horizon
- [ ] 2026 El Niño onset analysis

---

## License

MIT. Data from NOAA CPC is public domain.
