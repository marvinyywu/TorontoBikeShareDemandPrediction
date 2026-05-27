from pathlib import Path
import logging

import pandas as pd

import ingest
from config import DATETIME_FORMATS, TEST_YEAR

log = logging.getLogger(__name__)

_DATA_ROOT     = Path(__file__).parent.parent / "data"
_PROCESSED_DIR = _DATA_ROOT / "processed"
_CACHE_PATH    = _PROCESSED_DIR / "preprocessed.parquet"

_NEEDED_COLS = {"Start_Station_Id", "Start_Time"}


def _normalize(col: str) -> str:
    return col.strip().replace("﻿", "").replace(" ", "_")


def _cache_is_valid() -> bool:
    if not _CACHE_PATH.exists():
        return False
    cache_mtime = _CACHE_PATH.stat().st_mtime
    newest_csv  = max(
        (f.stat().st_mtime for f in (_DATA_ROOT / "raw").rglob("*.csv")),
        default=0,
    )
    return cache_mtime >= newest_csv


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # year is dropped after splitting — keeping it as a feature would leak the split boundary
    train_df = df[df["year"] < TEST_YEAR].drop(columns="year")
    test_df  = df[df["year"] == TEST_YEAR].drop(columns="year")

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Expected non-empty train and test splits, "
            f"got {len(train_df)} train rows and {len(test_df)} test rows."
        )
    return train_df, test_df


def preprocess_data(raw_dfs: list[tuple[int, pd.DataFrame]] | None = None) -> pd.DataFrame:
    log.info("Starting data preprocessing...")

    if raw_dfs is None and _cache_is_valid():
        log.info("Cache is valid. Loading preprocessed data from disk.")
        return pd.read_parquet(_CACHE_PATH)

    write_cache = raw_dfs is None  # captured before raw_dfs is reassigned below
    if raw_dfs is None:
        raw_dfs = ingest.ingest_data()

    dfs = []
    for i, (year, df) in enumerate(raw_dfs):
        log.info(f"Preprocessing file {i + 1}/{len(raw_dfs)} (year={year})...")
        df = df.copy()
        df.columns = [_normalize(c) for c in df.columns]

        missing = _NEEDED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"File {i + 1} (year={year}) missing columns after normalization: {missing}"
            )

        df = df[list(_NEEDED_COLS)]
        fmt = DATETIME_FORMATS.get(year, "mixed")
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], format=fmt)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid DataFrames after preprocessing. Check raw data and column names.")

    raw_df = pd.concat(dfs, ignore_index=True)

    # Aggregate trips to station-hour demand counts
    raw_df["hour_bucket"] = raw_df["Start_Time"].dt.floor("h")
    demand = (
        raw_df.groupby(["Start_Station_Id", "hour_bucket"])
        .size()
        .rename("demand")
        .reset_index()
        .rename(columns={"Start_Station_Id": "station_id"})
    )

    # Full station × hour cross-join — fill gaps with zero demand
    all_hours    = pd.date_range(demand["hour_bucket"].min(), demand["hour_bucket"].max(), freq="h")
    all_stations = demand["station_id"].unique()
    full_index   = pd.MultiIndex.from_product(
        [all_stations, all_hours], names=["station_id", "hour_bucket"]
    )
    demand = (
        demand.set_index(["station_id", "hour_bucket"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["station_id", "hour_bucket"])
    )

    if write_cache:
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        demand.to_parquet(_CACHE_PATH, index=False)
        log.info(f"Saved preprocessed data to {_CACHE_PATH}")

    log.info("Data preprocessing complete.")
    return demand[["station_id", "hour_bucket", "demand"]]
