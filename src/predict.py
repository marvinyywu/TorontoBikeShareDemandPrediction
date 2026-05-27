import numpy as np
import pandas as pd


def predict_baseline(baseline_model: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    return (
        test_df[["station_id", "hour", "day_of_week"]]
        .merge(baseline_model, on=["station_id", "hour", "day_of_week"], how="left")
        ["y_baseline"]
        .fillna(0)  # unseen station/hour/dow combinations default to zero demand
        .values
    )


def predict(model, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)
