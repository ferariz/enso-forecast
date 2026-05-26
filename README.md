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
| **t+1** | LightGBM | **0.955** | 0.858 |
| **t+3** | LightGBM | **0.798** | 0.614 |
| **t+6** | Logistic Regression | **0.617** | 0.414 |

LR outperforms LightGBM at t+6. With WWV added, LightGBM now wins at t+3 as well.
The high single-split scores partly reflect the 2019–2026 test period, which included
the unusually persistent 2020–2023 triple-dip La Niña.

**Regression (nino34_t6):** RMSE = 0.525°C, skill = +0.286 vs persistence.
Adding WWV subsurface heat content improved t+6 regression skill by ~32%.

---

## Walk-forward cross-validation

The single-split benchmark uses a fixed test set (2019–2026). Walk-forward CV
across 29 folds (1990–2018) gives a more conservative, cross-regime estimate:

| Target | Best model | CV mean F1 | CV std | Significant vs persistence? |
|---|---|---|---|---|
| t+1 | LightGBM | 0.684 | 0.231 | ✓ p<0.001 |
| t+3 | LightGBM | 0.465 | 0.158 | ✓ p=0.002 |
| t+6 | LightGBM | 0.271 | 0.086 | ✓ p=0.044 |

The high standard deviation reflects the ENSO cycle — some periods are
inherently more predictable than others. At t+6, only LightGBM is
statistically significant (p=0.044); LR and RF are not robust across
all regimes, though they perform well on the 2019–2026 test set.

Run CV locally:
```bash
python scripts/run_walk_forward_cv.py
```

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
  run_walk_forward_cv.py    # 29-fold CV with significance testing

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
  metrics/cv_results.json   # walk-forward CV fold results
  metrics/cv_summary.json   # CV mean/std/p-values
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
| WWV | NOAA PMEL | Warm water volume above 20°C isotherm — subsurface heat content |

---

## Features

61 features, all **strictly backward-looking** (no future leakage):

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

For cross-validation, walk-forward CV uses 29 non-overlapping folds
across 1990–2018 with expanding training windows (min 120 months).
See `src/validation/splits.py` and `scripts/run_walk_forward_cv.py`.

Automated leakage checks run on every dataset build:
- Index is monotonically increasing
- No feature encodes a negative lag (future data)
- No target column in the feature set
- `enso_t1[i]` equals `enso_phase[i+1]` (shift correctness)

---

## Live analysis

### 🌊 ENSO 2026: Is El Niño developing?
**Update May 2026:** El Niño is now near-certain. The SOI dropped from
+2.0 to **−11.2** — the atmosphere coupled with the ocean warming.
IRI assigns 98% probability to El Niño for May–July 2026.
Model predictions (April init, WWV-aware): **+1.46–1.72°C** by October.

See the analysis notebook:
[enso_2026_forecast.ipynb](notebooks/research/enso_2026_forecast.ipynb)

**Update planned for August 2026** — when post-barrier skill jumps from
F1 ~0.43 to ~0.75 and the model predictions become meaningfully reliable.

---

## Roadmap

- [x] WWV subsurface heat content (NOAA PMEL)
- [ ] WWV_E / WWV_W east-west split
- [ ] MJO features (BOM RMM index)
- [ ] Thermocline depth (D20 index)
- [x] Walk-forward cross-validation (29 folds, paired t-test vs persistence)
- [ ] t+9 month horizon
- [ ] 2026 El Niño onset analysis

---

## Research

This dataset and pipeline are being developed toward a scientific publication.
If you use this dataset in your research, please cite the Kaggle dataset and
the GitHub repository for now — a formal citation will be added once a paper
is submitted.

Areas of active investigation:
- Walk-forward CV implemented — 29 folds, t+3 significant p<0.005, t+6 marginal p=0.044
- Statistical significance of LR vs LightGBM performance gap at t+6
- CV reveals test set optimism: single-split F1 partly reflects 2019-2026 predictability
- Spring barrier quantification as a function of initialization month
- Prospective validation on the 2026 ENSO season

Potential venues: *Environmental Data Science*, *Int. Journal of Climatology*,
*Geoscientific Model Development*, or a climate ML workshop.

Collaborations and discussions welcome — open an issue on GitHub.

---

## License

MIT. Data from NOAA CPC is public domain.
