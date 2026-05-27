import sys
import logging
from pathlib import Path

import joblib
import pandas as pd

# Add src/ to path so modules import each other without a package prefix
sys.path.insert(0, str(Path(__file__).parent / "src"))

import preprocess
import features
import train
import predict
import evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT_PATH  = Path(__file__).parent
MODELS_DIR = ROOT_PATH / "models"

INCLUDE_BASELINE = True
MODEL_TYPE       = "lightgbm"

if __name__ == "__main__":

    # --- Preprocessing (cache-aware: preprocess_data() checks mtime internally) ---
    demand_df = preprocess.preprocess_data()

    # --- Feature engineering (with Parquet cache) ---
    features_path = ROOT_PATH / "data" / "processed" / "featured_data.parquet"
    if not features_path.exists():
        featured_df = features.build_features(demand_df)
    else:
        log.info("Loading cached featured data.")
        featured_df = pd.read_parquet(features_path)

    # --- Split ---
    train_df, test_df = preprocess.split_data(featured_df)
    X_train = train_df.drop(columns="demand")
    y_train = train_df["demand"].values
    X_test  = test_df.drop(columns="demand")
    y_test  = test_df["demand"].values

    # --- Baseline ---
    if INCLUDE_BASELINE:
        baseline_model = train.train_baseline(train_df)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline_model, MODELS_DIR / "baseline_model.joblib")
        predictions_baseline = predict.predict_baseline(baseline_model, test_df)
        evaluate.evaluate_baseline(y_test, predictions_baseline)

    # --- Main model (load from disk if already trained) ---
    model_path = MODELS_DIR / f"{MODEL_TYPE}_model.joblib"
    if not model_path.exists():
        log.info("Trained model not found. Training a new model.")
        model = train.train_model(X_train, y_train, model_type=MODEL_TYPE)
    else:
        log.info("Found trained model. Loading from disk.")
        model = joblib.load(model_path)

    evaluate.evaluate_model(model, X_train, y_train, X_test, y_test, label=MODEL_TYPE)
