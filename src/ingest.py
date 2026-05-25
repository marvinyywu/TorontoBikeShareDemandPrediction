from pathlib import Path

import pandas as pd

_DATASETS_ROOT = Path(__file__).parent.parent / "datasets"

_DATETIME_FORMATS = {
    2021: "%m/%d/%Y %H:%M",
    2022: "%m/%d/%Y %H:%M",
    2023: "%m/%d/%Y %H:%M",
    2024: "%Y-%m-%d %H:%M:%S",
    # 2025 files contain both HH:MM and HH:MM:SS rows, so per-element inference is needed
    2025: "mixed",
}

_NEEDED_COLS = {"Start_Station_Id", "Start_Time"}


def _normalize(col: str) -> str:
    return col.strip().replace("﻿", "").replace(" ", "_")


def ingest_data():

    print("Starting data ingestion...")


    year_dirs = sorted(_DATASETS_ROOT.glob("bikeshare-ridership-[0-9][0-9][0-9][0-9]"))

    dfs = []
    for year_dir in year_dirs:
        year = int(year_dir.name.split("-")[-1])
        fmt = _DATETIME_FORMATS.get(year, "mixed")
        for f in sorted(year_dir.glob("*.csv")):
            df = pd.read_csv(
                f,
                encoding="cp1252",
                usecols=lambda c: _normalize(c) in _NEEDED_COLS,
            )
            df.columns = [_normalize(c) for c in df.columns]
            df["Start_Time"] = pd.to_datetime(df["Start_Time"], format=fmt)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No bikeshare CSV files found under {_DATASETS_ROOT}")

    return pd.concat(dfs, ignore_index=True)
