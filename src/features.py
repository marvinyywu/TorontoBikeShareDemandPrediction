from pathlib import Path
import logging

import pandas as pd

from config import RUSH_HOURS, ROLLING_WINDOW_HOURS

log = logging.getLogger(__name__)

_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
_FEATURES_PATH = _PROCESSED_DIR / "featured_data.parquet"


def build_features(demand: pd.DataFrame, *, write_cache: bool = True) -> pd.DataFrame:
    log.info("Starting feature engineering...")

    demand = demand.copy()

    # Lag and rolling features — shift before rolling to avoid target leakage
    g          = demand.groupby("station_id")["demand"]
    shifted_1h = g.shift(1)
    demand["lag_1h_demand"]  = shifted_1h.fillna(0)
    demand["lag_24h_demand"] = g.shift(24).fillna(0)
    demand["rolling_7d_avg"] = (
        shifted_1h
        .transform(lambda s: s.rolling(window=ROLLING_WINDOW_HOURS, min_periods=1).mean())
        .fillna(0)
    )

    # Time features
    t = demand["hour_bucket"]
    demand["year"]         = t.dt.year
    demand["hour"]         = t.dt.hour
    demand["day_of_week"]  = t.dt.dayofweek
    demand["month"]        = t.dt.month
    demand["is_weekend"]   = t.dt.dayofweek >= 5
    demand["is_rush_hour"] = t.dt.hour.isin(RUSH_HOURS)

    result = demand[[
        "station_id", "year", "hour", "day_of_week", "month",
        "is_weekend", "is_rush_hour", "lag_1h_demand", "lag_24h_demand",
        "rolling_7d_avg", "demand",
    ]]

    if write_cache:
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(_FEATURES_PATH, index=False)
        log.info(f"Feature engineering complete. Saved to {_FEATURES_PATH}")
    else:
        log.info("Feature engineering complete.")

    return result
