# Toronto Bike Share Demand Prediction

## Problem
Bike share systems need to anticipate station-level demand so bikes can be rebalanced more effectively.

## Data
I used Bike Share Toronto ridership data from [City of Toronto Open Data](https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data).

## Approach
I built a pipeline that ingests trip data, aggregates it into station-hour demand, creates time and lag features, trains an ML model, and evaluates performance on future holdout data.

## Models
- LightGBM (`LGBMRegressor`) with automated hyperparameter tuning via `RandomizedSearchCV` (20 iterations, 3-fold CV on a 500k-row sample)
- Chosen over Random Forest for significantly faster training on 39M rows, lower memory usage via histogram binning, and native support for high-cardinality categoricals like `station_id`

## Results
Reports Train R², Test R², MAE, and RMSE evaluated on 2025 holdout data.

## Engineering
- pandas preprocessing with Parquet cache (auto-invalidated when source CSVs change)
- Per-year datetime format parsing to handle inconsistent CSV formats across 2021–2025
- Column pruning at read time (`usecols`) to reduce I/O across 3.85 GB of source data
- Time-based train/test split: 2021–2024 train, 2025 test
- Hyperparameter search on a representative sample to stay within compute constraints
- Modular pipeline: `ingest` → `preprocess` → `train` → `evaluate`


## Limitations
The model uses historical trip data and may not fully capture sudden disruptions, station closures, weather shocks, or policy changes.

## How to Run

1. Download Bike Share Toronto ridership data for 2021 to 2025 from [City of Toronto Open Data](https://ckan0.cf.opendata.inter.prod-toronto.ca/en_AU/dataset/bike-share-toronto-ridership-data).
2. Structure the folders like so:
![Image of Dataset Folder Structure](images/Screenshot%202026-05-23%20194446.png)
3. Run `pip install -r requirements.txt`
4. Run `python src/main.py`

> **Note:** The first run preprocesses all source CSVs and writes a Parquet cache to `datasets/preprocessed_cache.parquet`. Subsequent runs load the cache directly. Delete the cache file if you add or update source data.

