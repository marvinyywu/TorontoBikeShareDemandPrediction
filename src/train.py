import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import evaluate
import preprocess

# Number of rows sampled from training data for hyperparameter search.
# Full 39M rows make exhaustive CV impractical; 500k preserves distribution
# while keeping each search fit to a few seconds.
_TUNE_SAMPLE_SIZE = 500_000

_PARAM_DIST = {
    "n_estimators":      [200, 500, 1000],
    "num_leaves":        [31, 63, 127, 255],
    "max_depth":         [-1, 8, 15],
    "min_child_samples": [20, 50, 100],
    "learning_rate":     [0.05, 0.1, 0.2],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0],
}


def _compute_baseline(train_df, test_df):
    means = (
        train_df.groupby(["station_id", "hour", "day_of_week"])["demand"]
        .mean()
        .rename("y_baseline")
        .reset_index()
    )
    return (
        test_df[["station_id", "hour", "day_of_week"]]
        .merge(means, on=["station_id", "hour", "day_of_week"], how="left")
        ["y_baseline"]
        .fillna(0)
        .values
    )


def _tune_hyperparameters(X_train, y_train):
    rng = np.random.default_rng(42)
    if len(X_train) > _TUNE_SAMPLE_SIZE:
        idx = rng.choice(len(X_train), size=_TUNE_SAMPLE_SIZE, replace=False)
        X_sample, y_sample = X_train[idx], y_train[idx]
    else:
        X_sample, y_sample = X_train, y_train

    search = RandomizedSearchCV(
        LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        param_distributions=_PARAM_DIST,
        n_iter=20,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # avoid nested parallelism: LGBMRegressor already uses n_jobs=-1
        random_state=42,
        verbose=1,
    )
    search.fit(X_sample, y_sample)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    return search.best_params_


def train_model(df=None):

    print("Starting model training...")

    if df is None:
        df = preprocess.preprocess_data()

    train_df = df[df["year"] < 2025].drop(columns="year")
    test_df  = df[df["year"] == 2025].drop(columns="year")

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Expected non-empty train and test splits, "
            f"got {len(train_df)} train rows and {len(test_df)} test rows."
        )

    X_train = train_df.drop(columns="demand").values
    y_train = train_df["demand"].values
    X_test  = test_df.drop(columns="demand").values
    y_test  = test_df["demand"].values

    print("\n--- Baseline (station × hour × day_of_week mean) ---")
    y_baseline = _compute_baseline(train_df, test_df)
    evaluate.evaluate_predictions(y_test, y_baseline, label="Baseline")

    print(f"\n--- LightGBM ---")
    print(f"Tuning hyperparameters on a {_TUNE_SAMPLE_SIZE:,}-row sample...")
    best_params = _tune_hyperparameters(X_train, y_train)

    model = LGBMRegressor(**best_params, n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    evaluate.evaluate_model(model, X_train, y_train, X_test, y_test)

    return model
