from pathlib import Path
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib

from config import (
    TUNE_SAMPLE_SIZE, RANDOM_SEED,
    RFR_PARAM_DIST, LGBM_PARAM_DIST,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, PASSTHROUGH_FEATURES,
)

log = logging.getLogger(__name__)

_MODELS_DIR       = Path(__file__).parent.parent / "models"
_VALID_MODEL_TYPES = frozenset({"lightgbm", "random_forest", "linear_regression"})


def train_baseline(train_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Training baseline (station × hour × day_of_week mean)...")
    return (
        train_df.groupby(["station_id", "hour", "day_of_week"])["demand"]
        .mean()
        .rename("y_baseline")
        .reset_index()
    )


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("scale",  StandardScaler(),                                                      NUMERIC_FEATURES),
        # OrdinalEncoder handles station_id's high cardinality without the memory cost of OHE
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ("pass",   "passthrough",                                                         PASSTHROUGH_FEATURES),
    ])


def _tune_hyperparameters(
    estimator,
    param_dist: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X_train) > TUNE_SAMPLE_SIZE:
        idx      = rng.choice(len(X_train), size=TUNE_SAMPLE_SIZE, replace=False)
        X_sample = X_train.iloc[idx]
        y_sample = y_train[idx]
    else:
        X_sample, y_sample = X_train, y_train

    pipe = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", estimator),
    ])
    search = RandomizedSearchCV(
        pipe,
        param_distributions={f"model__{k}": v for k, v in param_dist.items()},
        n_iter=20,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # avoid nested parallelism: base estimator already uses n_jobs=-1
        random_state=RANDOM_SEED,
        verbose=1,
    )
    search.fit(X_sample, y_sample)
    best_params = {k[len("model__"):]: v for k, v in search.best_params_.items()}
    log.info(f"Best params: {best_params}")
    log.info(f"Best CV RMSE: {-search.best_score_:.4f}")
    return best_params


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = "lightgbm",
) -> Pipeline:
    log.info("Starting model training...")

    if model_type == "linear_regression":
        log.info("--- Linear Regression ---")
        estimator = LinearRegression()
    elif model_type == "random_forest":
        log.info("--- Random Forest ---")
        log.info(f"Tuning hyperparameters on a {TUNE_SAMPLE_SIZE:,}-row sample...")
        best_params = _tune_hyperparameters(
            RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            RFR_PARAM_DIST,
            X_train, y_train,
        )
        estimator = RandomForestRegressor(**best_params, n_jobs=-1, random_state=RANDOM_SEED)
    elif model_type == "lightgbm":
        log.info("--- LightGBM ---")
        log.info(f"Tuning hyperparameters on a {TUNE_SAMPLE_SIZE:,}-row sample...")
        best_params = _tune_hyperparameters(
            LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
            LGBM_PARAM_DIST,
            X_train, y_train,
        )
        estimator = LGBMRegressor(**best_params, n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Valid: {sorted(_VALID_MODEL_TYPES)}"
        )

    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor()),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _MODELS_DIR / f"{model_type}_model.joblib"
    joblib.dump(pipeline, out_path)
    log.info(f"Model saved to {out_path}")

    return pipeline
