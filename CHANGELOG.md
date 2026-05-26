# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v2.2] — 2026-05-26

### Added
- Walk-forward cross-validation (`src/validation/splits.py`, `scripts/run_walk_forward_cv.py`)
  - 29 folds, expanding training window, paired t-test vs persistence
  - t+3 significant p<0.005, t+6 marginal p=0.044
  - Results saved to `outputs/metrics/cv_results.json` and `cv_summary.json`
- ENSO 2026 forecast notebook (`notebooks/research/enso_2026_forecast.ipynb`)
  - Spring barrier analysis, historical analogues, spaghetti plot
  - Ridge regression alongside LightGBM for tail risk quantification
  - Breaking update: SOI flipped to −11.2 (May 16, 2026), El Niño near-certain
- Column descriptions in `dataset-metadata.json` for all 68 exported columns

### Changed
- Updated benchmarks: t+1 LightGBM 0.955, t+3 LightGBM 0.798, t+6 LR 0.617
- README: added CV section, updated live analysis, roadmap progress

---

## [v2.0] — 2026-05-20

### Added
- **WWV (Warm Water Volume)** subsurface heat content feature (`src/ingestion/wwv_loader.py`)
  - Source: NOAA PMEL, monthly, 1980–present
  - Standardised anomaly (`wwv_anom_std`) added as base variable
  - 7 derived features (lags, rolling mean/std, diff)
  - t+6 regression skill improved from +0.217 to +0.286 (~32% gain)
  - t+6 classification F1 improved from 0.553 to 0.617
- Ridge regression comparison in 2026 forecast notebook
  - Quantifies extrapolation ceiling of tree models (~+1.90°C)
  - Gap between Ridge and LightGBM = tail risk estimate

### Changed
- Dataset: 54 → 61 features, 61 → 68 total columns
- Kaggle dataset: v2.0 published with WWV features

---

## [v1.1] — 2026-05-16

### Added
- Regression targets: `nino34_t1`, `nino34_t3`, `nino34_t6` (smoothed Niño 3.4 anomaly in °C)
- Regression training in `scripts/train_models.py`
  - LightGBM regressor per horizon, saved to `outputs/models/nino34_tL/`
  - RMSE skill table vs persistence baseline
- Spring predictability barrier features
  - `crosses_spring_t1/t3/t6` — boolean, does forecast window cross MAM?
  - `init_in_growth_phase` — boolean, is init in JJA–SON high-skill window?
- Stratified evaluation by spring barrier regime (`src/evaluation/metrics.py`)
  - `evaluate_by_spring_barrier()` — F1 split by spring-crossing vs not
  - `evaluate_by_init_month()` — F1 by calendar month of initialization
- Starter notebook fully updated: SHAP, lead-time curve, spring barrier heatmap, regression scatter

### Fixed
- Critical leakage: regression targets and `nino34_smoothed` were included in feature set
- Export pipeline: base anomaly columns were dropped before renaming

---

## [v1.0.0] — 2026-05-05

### Added
- Full data pipeline: ingest → preprocess → label → feature engineering → validate → export
- Data sources: NOAA CPC Niño indices (ERSSTv5), SOI, 850 hPa zonal wind
- 50 features: lags (1/3/6m), rolling mean/std, first differences, calendar encoding
- Classification targets: `enso_t1`, `enso_t3`, `enso_t6` (El Niño / Neutral / La Niña)
- Models: Logistic Regression, Random Forest, LightGBM + persistence/climatology baselines
- Automated leakage checks (4 checks, run on every build)
- Kaggle export pipeline (`scripts/export_kaggle_dataset.py`)
- Starter notebook: EDA, LR baseline, LightGBM, SHAP, lead-time curve
- Kaggle dataset published: [ferariz/enso-early-phase-prediction](https://www.kaggle.com/datasets/ferariz/enso-early-phase-prediction)

### Benchmarks (test set 2019–2026)
- t+1: LightGBM F1 = 0.945
- t+3: LR F1 = 0.769
- t+6: LR F1 = 0.553
