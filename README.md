# Toronto Bike Share Demand Prediction

## Problem
Bike share systems need to anticipate station-level demand so bikes can be rebalanced more effectively.

## Data
I used Bike Share Toronto ridership data from [City of Toronto Open Data](https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data).

## Approach
I built a pipeline that ingests trip data, aggregates it into station-hour demand counts, engineers time and lag features, trains an ML model, and evaluates performance on a future holdout year.

Pipeline: `ingest` → `preprocess` → `features` → `train` → `predict` → `evaluate`

## Models
Three models were compared, all evaluated on 2025 holdout data:

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|
| Baseline (station × hour × dow mean) | — | — | 0.8020 | 2.4792 |
| Linear Regression | 0.5588 | 0.6223 | 0.6360 | 1.4417 |
| Random Forest | 0.6128 | 0.6445 | 0.5656 | 1.3988 |
| **LightGBM** | **0.6497** | **0.6660** | **0.5478** | **1.3558** |

LightGBM was selected as the primary model for faster training on 39M rows, lower memory usage via histogram binning, and native support for high-cardinality categoricals like `station_id`. Hyperparameters are tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV on a 500k-row sample).

## Features
- **Time features:** year, hour, day of week, month, is_weekend, is_rush_hour
- **Lag features:** demand 1 hour ago, demand 24 hours ago (per station)
- **Rolling average:** 7-day rolling mean of demand (per station), shifted by 1 hour to prevent target leakage

## Engineering
- pandas preprocessing with two-level Parquet cache (preprocessed demand and engineered features), auto-invalidated when source CSVs change
- Per-year datetime format parsing to handle inconsistent CSV formats across 2021–2025
- Column pruning at read time (`usecols`) to reduce I/O across 3.85 GB of source data
- Full station × hour cross-join with zero-fill to represent hours with no trips
- Time-based train/test split: 2021–2024 train, 2025 test; `year` is dropped as a feature to prevent leakage across the split boundary
- Hyperparameter search on a representative sample to stay within compute constraints
- Streamlit dashboard (`app.py`) for interactive exploration and station-level forecasts
- Unit tests in `test/` covering feature engineering (16 tests)

## Limitations
The model uses only historical trip patterns. It does not account for weather, station capacity, special events, station openings/closures, or other external disruptions.

## How to Run

1. Download Bike Share Toronto ridership data for 2021 to 2025 from [City of Toronto Open Data](https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data).
2. Structure the folders like so:
![Image of Dataset Folder Structure](images/Screenshot%202026-05-23%20194446.png)
3. Run `pip install -r requirements.txt`
4. Run `python main.py` to train the model and print evaluation metrics.
5. Run `streamlit run app.py` to launch the interactive dashboard.

> **Note:** The first run preprocesses all source CSVs and writes Parquet caches to `data/processed/preprocessed.parquet` and `data/processed/featured_data.parquet`. Subsequent runs load these caches directly. Delete the cache files if you add or update source data.
