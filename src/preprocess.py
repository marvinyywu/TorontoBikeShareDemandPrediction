from pathlib import Path

import pandas as pd
import ingest

_DATASETS_ROOT = Path(__file__).parent.parent / "datasets"
_CACHE_PATH = _DATASETS_ROOT / "preprocessed_cache.parquet"


def _cache_is_valid() -> bool:
    if not _CACHE_PATH.exists():
        return False
    cache_mtime = _CACHE_PATH.stat().st_mtime
    newest_csv = max(
        (f.stat().st_mtime for f in _DATASETS_ROOT.rglob("*.csv")),
        default=0,
    )
    return cache_mtime >= newest_csv


def preprocess_data(df=None):

    if df is None and _cache_is_valid():
        return pd.read_parquet(_CACHE_PATH)

    raw_df = ingest.ingest_data() if df is None else df.copy()

    raw_df["hour_bucket"] = raw_df["Start_Time"].dt.floor("h")

    demand = (
        raw_df
        .groupby(["Start_Station_Id", "hour_bucket"])
        .size()
        .rename("demand")
        .reset_index()
        .rename(columns={"Start_Station_Id": "station_id"})
    )

    all_hours = pd.date_range(
        demand["hour_bucket"].min(),
        demand["hour_bucket"].max(),
        freq="h",
    )
    all_stations = demand["station_id"].unique()
    full_index = pd.MultiIndex.from_product(
        [all_stations, all_hours], names=["station_id", "hour_bucket"]
    )
    demand = (
        demand
        .set_index(["station_id", "hour_bucket"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["station_id", "hour_bucket"])
    )

    g = demand.groupby("station_id")["demand"]
    shifted_1h = g.shift(1)
    demand["lag_1h_demand"]  = shifted_1h.fillna(0)
    demand["lag_24h_demand"] = g.shift(24).fillna(0)
    demand["rolling_7d_avg"] = shifted_1h.transform(
        lambda s: s.rolling(window=24 * 7, min_periods=1).mean()
    ).fillna(0)

    t = demand["hour_bucket"]
    demand["year"]        = t.dt.year
    demand["hour"]        = t.dt.hour
    demand["day_of_week"] = t.dt.dayofweek
    demand["month"]       = t.dt.month
    demand["is_weekend"]  = t.dt.dayofweek >= 5
    demand["is_rush_hour"] = t.dt.hour.isin([7, 8, 9, 16, 17, 18])

    result = demand[[
        "station_id",
        "year",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_rush_hour",
        "lag_1h_demand",
        "lag_24h_demand",
        "rolling_7d_avg",
        "demand",
    ]]

    if df is None:
        result.to_parquet(_CACHE_PATH, index=False)

    return result
