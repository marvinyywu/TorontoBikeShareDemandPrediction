import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import evaluate
import preprocess

# Number of rows sampled from training data for hyperparameter search.
# Full 39M rows make exhaustive CV impractical; 500k preserves distribution
# while keeping each search fit to a few seconds.
_TUNE_SAMPLE_SIZE = 500_000

_PARAM_DIST = {
    "n_estimators":     [1, 3, 5, 10],
    "max_depth":        [5, 8, 10, 15, 20],
    "min_samples_leaf": [10, 25, 50, 100],
    "max_features":     ["sqrt", 0.5, 1.0],
}


def _tune_hyperparameters(X_train, y_train):
    rng = np.random.default_rng(42)
    if len(X_train) > _TUNE_SAMPLE_SIZE:
        idx = rng.choice(len(X_train), size=_TUNE_SAMPLE_SIZE, replace=False)
        X_sample, y_sample = X_train[idx], y_train[idx]
    else:
        X_sample, y_sample = X_train, y_train

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=_PARAM_DIST,
        n_iter=20,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
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

    print(f"Tuning hyperparameters on a {_TUNE_SAMPLE_SIZE:,}-row sample...")
    best_params = _tune_hyperparameters(X_train, y_train)

    model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    evaluate.evaluate_model(model, X_train, y_train, X_test, y_test)

    return model
