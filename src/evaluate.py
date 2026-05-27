import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

log = logging.getLogger(__name__)


def evaluate_model(model, X_train, y_train, X_test, y_test, label="") -> np.ndarray:
    y_pred   = model.predict(X_test)
    r2_train = model.score(X_train, y_train)
    r2_test  = model.score(X_test, y_test)
    tag      = f"{label} " if label else ""
    log.info(f"Train R²:       {r2_train:.4f}")
    log.info(f"Test  R²:       {r2_test:.4f}")
    log.info(f"{tag}Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    log.info(f"{tag}Test RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
    return y_pred


def evaluate_baseline(y_test, y_pred):
    log.info(f"Baseline MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    log.info(f"Baseline RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
