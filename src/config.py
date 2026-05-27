TEST_YEAR = 2025

# Per-year datetime formats. 2025 files mix HH:MM and HH:MM:SS rows in the
# same file, so per-element inference is required for that year only.
DATETIME_FORMATS: dict[int, str] = {
    2021: "%m/%d/%Y %H:%M",
    2022: "%m/%d/%Y %H:%M",
    2023: "%m/%d/%Y %H:%M",
    2024: "%Y-%m-%d %H:%M:%S",
    2025: "mixed",
}

RUSH_HOURS           = [7, 8, 9, 16, 17, 18]
ROLLING_WINDOW_HOURS = 24 * 7  # 7 days

TUNE_SAMPLE_SIZE = 500_000

RANDOM_SEED = 42

RFR_PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [5, 8, 10, 15, 20],
    "min_samples_leaf": [10, 25, 50, 100],
    "max_features":     ["sqrt", 0.5, 1.0],
}

LGBM_PARAM_DIST = {
    "n_estimators":      [200, 500, 1000],
    "num_leaves":        [31, 63, 127, 255],
    "max_depth":         [-1, 8, 15],
    "min_child_samples": [20, 50, 100],
    "learning_rate":     [0.05, 0.1, 0.2],
    "subsample":         [0.7, 0.8, 1.0],
    "subsample_freq":    [1],  # must be > 0 for subsample to activate in LightGBM
    "colsample_bytree":  [0.7, 0.8, 1.0],
}

# Feature column groups used by the sklearn ColumnTransformer in train.py
NUMERIC_FEATURES     = ["lag_1h_demand", "lag_24h_demand", "rolling_7d_avg"]
CATEGORICAL_FEATURES = ["station_id"]
PASSTHROUGH_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"]
