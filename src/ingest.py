import re
from pathlib import Path
import logging

import pandas as pd

log = logging.getLogger(__name__)

_DATASETS_ROOT = Path(__file__).parent.parent / "data" / "raw"

# Normalized column names required from each raw CSV (used by the usecols filter below)
_NEEDED_COLS = frozenset({"Start_Station_Id", "Start_Time"})


def _normalize_col(col: str) -> str:
    return col.strip().replace("﻿", "").replace(" ", "_")


def ingest_data() -> list[tuple[int, pd.DataFrame]]:
    log.info("Starting data ingestion...")

    year_dirs = sorted(_DATASETS_ROOT.glob("bikeshare-ridership-[0-9][0-9][0-9][0-9]"))

    results: list[tuple[int, pd.DataFrame]] = []
    for year_dir in year_dirs:
        m = re.search(r"(\d{4})$", year_dir.name)
        if not m:
            log.warning(f"Skipping unexpected directory: {year_dir.name}")
            continue
        year = int(m.group(1))

        for f in sorted(year_dir.glob("*.csv")):
            log.info(f"Loading {f.name} (year={year})...")
            try:
                df = pd.read_csv(
                    f,
                    encoding="cp1252",
                    usecols=lambda c: _normalize_col(c) in _NEEDED_COLS,
                )
            except Exception as e:
                log.warning(f"Skipping {f.name}: {e}")
                continue
            results.append((year, df))

    if not results:
        raise FileNotFoundError(f"No bikeshare CSV files found under {_DATASETS_ROOT}")

    total_rows = sum(len(df) for _, df in results)
    log.info(f"Ingestion complete: {len(results)} files, {total_rows:,} rows.")
    return results
