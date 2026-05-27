import pandas as pd
import pytest

import features
from config import RUSH_HOURS

EXPECTED_COLUMNS = [
    "station_id", "year", "hour", "day_of_week", "month",
    "is_weekend", "is_rush_hour", "lag_1h_demand", "lag_24h_demand",
    "rolling_7d_avg", "demand",
]


def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a demand DataFrame from (station_id, iso_datetime_str, demand) tuples."""
    df = pd.DataFrame(rows, columns=["station_id", "hour_bucket", "demand"])
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
    return df.sort_values(["station_id", "hour_bucket"]).reset_index(drop=True)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return _make_df([
        ("A", "2024-01-01 07:00", 5.0),
        ("A", "2024-01-01 08:00", 3.0),
        ("A", "2024-01-01 09:00", 2.0),
        ("B", "2024-01-01 07:00", 10.0),
        ("B", "2024-01-01 08:00", 7.0),
        ("B", "2024-01-01 09:00", 4.0),
    ])


# ── Schema ────────────────────────────────────────────────────────────────────

def test_output_columns(simple_df):
    assert list(features.build_features(simple_df).columns) == EXPECTED_COLUMNS


def test_row_count_preserved(simple_df):
    assert len(features.build_features(simple_df)) == len(simple_df)


# ── Time features ─────────────────────────────────────────────────────────────

def test_time_fields_extracted():
    # 2024-03-15 is a Friday
    df = _make_df([("S", "2024-03-15 14:00", 1.0)])
    row = features.build_features(df).iloc[0]
    assert row["year"]        == 2024
    assert row["hour"]        == 14
    assert row["day_of_week"] == 4  # Friday = 4
    assert row["month"]       == 3


def test_is_weekend_false_on_weekday():
    df = _make_df([("S", "2024-01-01 10:00", 1.0)])  # 2024-01-01 is Monday
    assert features.build_features(df).iloc[0]["is_weekend"] == False


def test_is_weekend_true_on_saturday():
    df = _make_df([("S", "2024-01-06 10:00", 1.0)])  # 2024-01-06 is Saturday
    assert features.build_features(df).iloc[0]["is_weekend"] == True


def test_is_weekend_true_on_sunday():
    df = _make_df([("S", "2024-01-07 10:00", 1.0)])  # 2024-01-07 is Sunday
    assert features.build_features(df).iloc[0]["is_weekend"] == True


def test_is_rush_hour_true_for_all_rush_hours():
    for hour in RUSH_HOURS:
        df = _make_df([("S", f"2024-01-01 {hour:02d}:00", 1.0)])
        assert features.build_features(df).iloc[0]["is_rush_hour"] == True, (
            f"Hour {hour} should be marked as rush hour"
        )


def test_is_rush_hour_false_outside_rush():
    non_rush = [h for h in range(24) if h not in RUSH_HOURS]
    for hour in non_rush:
        df = _make_df([("S", f"2024-01-01 {hour:02d}:00", 1.0)])
        assert features.build_features(df).iloc[0]["is_rush_hour"] == False, (
            f"Hour {hour} should not be marked as rush hour"
        )


# ── Lag features ──────────────────────────────────────────────────────────────

def test_lag_1h_first_row_is_zero():
    df = _make_df([
        ("S", "2024-01-01 07:00", 5.0),
        ("S", "2024-01-01 08:00", 3.0),
    ])
    assert features.build_features(df).iloc[0]["lag_1h_demand"] == 0.0


def test_lag_1h_correct_value():
    df = _make_df([
        ("S", "2024-01-01 07:00", 5.0),
        ("S", "2024-01-01 08:00", 3.0),
    ])
    result = features.build_features(df)
    assert result.iloc[1]["lag_1h_demand"] == pytest.approx(5.0)


def test_lag_24h_first_24_rows_are_zero():
    df = _make_df([("S", f"2024-01-01 {h:02d}:00", float(h + 1)) for h in range(24)])
    result = features.build_features(df)
    assert (result["lag_24h_demand"] == 0.0).all()


def test_lag_24h_correct_value():
    rows = [("S", f"2024-01-01 {h:02d}:00", 0.0) for h in range(24)]
    rows[0] = ("S", "2024-01-01 00:00", 42.0)  # hour-0 demand = 42
    rows += [("S", "2024-01-02 00:00", 0.0)]    # 24 h later should see lag_24h = 42
    result = features.build_features(_make_df(rows))
    assert result.iloc[-1]["lag_24h_demand"] == pytest.approx(42.0)


def test_station_isolation_lag_1h():
    """Grouped shift must not bleed lag across station boundaries."""
    df = _make_df([
        ("A", "2024-01-01 07:00", 100.0),
        ("B", "2024-01-01 08:00", 50.0),
    ])
    result = features.build_features(df)
    assert result[result["station_id"] == "A"].iloc[0]["lag_1h_demand"] == 0.0
    assert result[result["station_id"] == "B"].iloc[0]["lag_1h_demand"] == 0.0


def test_station_isolation_lag_24h():
    """24-hour lag must not bleed across station boundaries."""
    rows_a = [("A", f"2024-01-01 {h:02d}:00", 99.0) for h in range(24)]
    rows_b = [("B", "2024-01-02 00:00", 1.0)]
    result = features.build_features(_make_df(rows_a + rows_b))
    b_row = result[result["station_id"] == "B"].iloc[0]
    assert b_row["lag_24h_demand"] == 0.0


# ── Rolling average ───────────────────────────────────────────────────────────

def test_rolling_7d_avg_no_leakage():
    """shift(1) before rolling must exclude the current row's demand."""
    # demands:    1, 2, 3
    # shifted_1h: NaN, 1, 2
    # rolling mean (min_periods=1, NaN skipped): 0 → 1.0 → 1.5
    df = _make_df([
        ("S", "2024-01-01 00:00", 1.0),
        ("S", "2024-01-01 01:00", 2.0),
        ("S", "2024-01-01 02:00", 3.0),
    ])
    result = features.build_features(df)
    assert result.iloc[0]["rolling_7d_avg"] == pytest.approx(0.0)
    assert result.iloc[1]["rolling_7d_avg"] == pytest.approx(1.0)
    assert result.iloc[2]["rolling_7d_avg"] == pytest.approx(1.5)


def test_rolling_7d_avg_accumulates_over_time():
    """Rolling average should grow as more history is observed."""
    df = _make_df([("S", f"2024-01-01 {h:02d}:00", 10.0) for h in range(5)])
    result = features.build_features(df)
    avgs = result["rolling_7d_avg"].tolist()
    # After the first row (which is 0), each subsequent average equals 10
    assert avgs[0] == pytest.approx(0.0)
    for avg in avgs[1:]:
        assert avg == pytest.approx(10.0)
