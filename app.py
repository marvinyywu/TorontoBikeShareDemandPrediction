import sys
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT_PATH = Path(__file__).parent
sys.path.insert(0, str(ROOT_PATH / "src"))

from config import TEST_YEAR, RANDOM_SEED
import preprocess
import predict

logging.basicConfig(level=logging.INFO, format="%(message)s")

FEATURES_PATH = ROOT_PATH / "data" / "processed" / "featured_data.parquet"
MODELS_DIR    = ROOT_PATH / "models"

MODEL_OPTIONS = {
    "LightGBM":          "lightgbm",
    "Random Forest":     "random_forest",
    "Linear Regression": "linear_regression",
}

FEATURE_DESCRIPTIONS = {
    "station_id":      "Station identifier",
    "hour":            "Hour of day (0–23)",
    "day_of_week":     "Day of week (0=Mon, 6=Sun)",
    "month":           "Month of year (1–12)",
    "is_weekend":      "1 if Saturday or Sunday",
    "is_rush_hour":    "1 if morning (7–9) or evening (16–18) rush hour",
    "lag_1h_demand":   "Demand 1 hour prior",
    "lag_24h_demand":  "Demand 24 hours prior",
    "rolling_7d_avg":  "Rolling mean demand over the past 7 days",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_featured_data() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        return None
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_model(model_type: str):
    path = MODELS_DIR / f"{model_type}_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_baseline():
    path = MODELS_DIR / "baseline_model.joblib"
    return joblib.load(path) if path.exists() else None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return root_mean_squared_error(y_true, y_pred)


# ── Cached computations ────────────────────────────────────────────────────────
# All expensive calls are wrapped here. Only the model_type string is used as
# the cache key so Streamlit doesn't have to hash large numpy arrays.
# Data arguments are prefixed with _ so Streamlit skips hashing them while
# still making the dependency explicit in the function signature.

@st.cache_data(show_spinner="Running model on test set...")
def compute_test_predictions(model_type: str, _X_test: pd.DataFrame) -> np.ndarray | None:
    m = load_model(model_type)
    return None if m is None else m.predict(_X_test)


@st.cache_data(show_spinner="Computing baseline predictions...")
def compute_baseline_predictions(_test_df: pd.DataFrame) -> np.ndarray | None:
    b = load_baseline()
    return None if b is None else predict.predict_baseline(b, _test_df)


@st.cache_data(show_spinner="Computing train R²...")
def compute_train_r2(model_type: str, _X_train: pd.DataFrame, _y_train: np.ndarray) -> float | None:
    # Sampled to 100k rows — exact train R² on 39M rows isn't meaningful and
    # is extremely slow for tree ensembles.
    m = load_model(model_type)
    if m is None:
        return None
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(_X_train), size=min(100_000, len(_X_train)), replace=False)
    return m.score(_X_train.iloc[idx], _y_train[idx])


@st.cache_data(show_spinner="Computing test R²...")
def compute_test_r2(model_type: str, _X_test: pd.DataFrame, _y_test: np.ndarray) -> float | None:
    m = load_model(model_type)
    return None if m is None else m.score(_X_test, _y_test)


@st.cache_data(show_spinner="Forecasting station demand...")
def compute_station_predictions(
    model_type: str, station_id, _test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray] | None:
    m = load_model(model_type)
    if m is None:
        return None
    mask = _test_df["station_id"] == station_id
    s_df = _test_df[mask]
    if s_df.empty:
        return None
    return s_df["demand"].values, m.predict(s_df.drop(columns="demand"))


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Toronto Bike Share Demand",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 Toronto Bike Share Demand Prediction")
st.caption("Station-level hourly demand prediction using historical ridership data (2021–2025).")

# ── Data loading ──────────────────────────────────────────────────────────────

df = load_featured_data()

if df is None:
    st.error(
        "Featured data not found at `data/processed/featured_data.parquet`. "
        "Run `python main.py` first to generate the preprocessed data."
    )
    st.stop()

train_df, test_df = preprocess.split_data(df)

has_train = not train_df.empty
has_test  = not test_df.empty

if has_train:
    X_train = train_df.drop(columns="demand")
    y_train = train_df["demand"].values
if has_test:
    X_test = test_df.drop(columns="demand")
    y_test  = test_df["demand"].values


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_explore, tab_performance, tab_predict = st.tabs([
    "Overview", "Data Explorer", "Model Performance", "Predictions",
])

# ── Overview ──────────────────────────────────────────────────────────────────

with tab_overview:
    st.header("Project Overview")

    left, right = st.columns(2)

    with left:
        st.subheader("Problem")
        st.write(
            "Bike share systems need to anticipate station-level demand so bikes can "
            "be rebalanced before shortages occur. This project predicts the number of "
            "trip departures at each Toronto Bike Share station per hour."
        )

        st.subheader("Pipeline")
        st.code("ingest → preprocess → features → train → predict → evaluate", language=None)

        st.subheader("Train / Test Split")
        st.write(f"- **Train:** all years before {TEST_YEAR}")
        st.write(f"- **Test:** {TEST_YEAR} (temporal holdout)")

        st.subheader("Models")
        st.write("- **LightGBM** — gradient boosted trees, primary model")
        st.write("- **Random Forest** — ensemble of decision trees")
        st.write("- **Baseline** — historical mean demand per station × hour × day-of-week")

    with right:
        st.subheader("Dataset Statistics")
        metrics = [
            ("Total rows",    f"{len(df):,}"),
            ("Stations",      f"{df['station_id'].nunique():,}"),
            ("Train rows",    f"{len(train_df):,}" if has_train else "N/A"),
            ("Test rows",     f"{len(test_df):,}"  if has_test  else "N/A"),
            ("Features",      str(len([c for c in train_df.columns if c != "demand"]))),
        ]
        for label, value in metrics:
            st.metric(label, value)

        st.subheader("Feature Set")
        feature_cols = [c for c in (train_df if has_train else test_df).columns if c != "demand"]
        st.dataframe(
            pd.DataFrame({
                "Feature":     feature_cols,
                "Description": [FEATURE_DESCRIPTIONS.get(c, "") for c in feature_cols],
            }),
            hide_index=True,
            width='stretch',
        )


# ── Data Explorer ─────────────────────────────────────────────────────────────

with tab_explore:
    st.header("Data Explorer")

    if not has_train:
        st.warning(
            "No training data available (all data is from the test year). "
            "Add data from earlier years to see training-set distributions."
        )
        explore_df = test_df
    else:
        explore_df = train_df

    sample = explore_df.sample(min(300_000, len(explore_df)), random_state=RANDOM_SEED)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Demand by Hour of Day")
        hourly = sample.groupby("hour")["demand"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=hourly, x="hour", y="demand", marker="o", linewidth=2, color="steelblue", ax=ax)
        ax.axvspan(7,  9,  alpha=0.12, color="orange", label="Rush hour")
        ax.axvspan(16, 18, alpha=0.12, color="orange")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Demand")
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Average Demand by Day of Week")
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = sample.groupby("day_of_week")["demand"].mean().reset_index()
        daily["day_label"] = [day_labels[i] for i in daily["day_of_week"]]
        daily["type"] = ["Weekend" if i >= 5 else "Weekday" for i in daily["day_of_week"]]
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(
            data=daily, x="day_label", y="demand", hue="type",
            palette={"Weekday": "steelblue", "Weekend": "coral"},
            dodge=False, errorbar=None, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Avg Demand")
        ax.legend(title="")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Average Demand by Month")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = sample.groupby("month")["demand"].mean().reset_index()
        monthly["month_label"] = [month_labels[m - 1] for m in monthly["month"]]
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=monthly, x="month_label", y="demand", color="steelblue", errorbar=None, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Avg Demand")
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("Demand Distribution (clipped at 20)")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(
            sample["demand"].clip(upper=20),
            bins=range(0, 22),
            color="steelblue",
            ax=ax,
        )
        ax.set_xlabel("Trips / hour")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Top 10 Stations by Total Demand")
    top = (
        explore_df.groupby("station_id")["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"demand": "Total Demand"})
    )
    st.dataframe(top, hide_index=True, width='stretch')


# ── Model Performance ─────────────────────────────────────────────────────────

with tab_performance:
    st.header("Model Performance")

    if not has_test:
        st.warning("No test data available.")
        st.stop()

    model_label = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))
    model       = load_model(MODEL_OPTIONS[model_label])
    baseline    = load_baseline()

    if model is None:
        st.warning(
            f"No saved model found at `models/{MODEL_OPTIONS[model_label]}_model.joblib`. "
            "Run `main.py` with the matching `MODEL_TYPE` to train it first."
        )
    else:
        model_key = MODEL_OPTIONS[model_label]
        y_pred    = compute_test_predictions(model_key, X_test)
        train_r2  = compute_train_r2(model_key, X_train, y_train) if has_train else None
        test_r2   = compute_test_r2(model_key, X_test, y_test)

        st.subheader(f"{model_label} Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train R² (~100k sample)", f"{train_r2:.4f}" if train_r2 is not None else "N/A")
        c2.metric("Test R²",   f"{test_r2:.4f}"                          if test_r2  is not None else "N/A")
        c3.metric("Test MAE",  f"{mean_absolute_error(y_test, y_pred):.4f}")
        c4.metric("Test RMSE", f"{rmse(y_test, y_pred):.4f}")

        if baseline is not None:
            st.subheader("vs. Baseline (station × hour × day-of-week mean)")
            y_base     = compute_baseline_predictions(test_df)
            base_mae   = mean_absolute_error(y_test, y_base)
            base_rmse  = rmse(y_test, y_base)
            model_mae  = mean_absolute_error(y_test, y_pred)
            model_rmse = rmse(y_test, y_pred)

            b1, b2 = st.columns(2)
            b1.metric(
                "Baseline MAE",  f"{base_mae:.4f}",
                delta=f"{model_mae - base_mae:.4f}",
                delta_color="inverse",
            )
            b2.metric(
                "Baseline RMSE", f"{base_rmse:.4f}",
                delta=f"{model_rmse - base_rmse:.4f}",
                delta_color="inverse",
            )

        st.subheader("Predicted vs Actual — Test Set Sample")
        n = min(5_000, len(y_test))
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y_test), size=n, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Scatter: predicted vs actual
        sns.scatterplot(x=y_test[idx], y=y_pred[idx], alpha=0.25, s=8, color="steelblue", ax=axes[0])
        lim = max(y_test[idx].max(), y_pred[idx].max()) * 1.05
        axes[0].plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect fit")
        axes[0].set_xlabel("Actual Demand")
        axes[0].set_ylabel("Predicted Demand")
        axes[0].set_title("Predicted vs Actual")
        axes[0].legend()

        # Residuals histogram
        residuals = y_pred[idx] - y_test[idx]
        sns.histplot(residuals, bins=50, color="steelblue", ax=axes[1])
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Residual (predicted − actual)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residual Distribution")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── Predictions ───────────────────────────────────────────────────────────────

with tab_predict:
    st.header("Station Demand Forecast")
    st.write(f"Actual vs predicted demand for a station over the {TEST_YEAR} test period.")

    if not has_test:
        st.warning("No test data available.")
        st.stop()

    pred_model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), key="pred_model")
    pred_model       = load_model(MODEL_OPTIONS[pred_model_label])

    if pred_model is None:
        st.warning(
            f"Train the selected model first by running `main.py` with "
            f"`MODEL_TYPE = '{MODEL_OPTIONS[pred_model_label]}'`."
        )
    else:
        station = st.selectbox("Station ID", sorted(test_df["station_id"].unique()))

        result = compute_station_predictions(MODEL_OPTIONS[pred_model_label], station, test_df)

        if result is None:
            st.warning("No test data for this station.")
        else:
            y_act, y_prd = result

            c1, c2 = st.columns(2)
            c1.metric("Station MAE",  f"{mean_absolute_error(y_act, y_prd):.4f}")
            c2.metric("Station RMSE", f"{rmse(y_act, y_prd):.4f}")

            # Sliders only control the view window — prediction is already cached above
            total_hours = len(y_act)
            window = st.slider(
                "Hours to display",
                min_value=24,
                max_value=min(total_hours, 24 * 90),
                value=min(total_hours, 24 * 14),
                step=24,
            )
            start = st.slider("Start offset (hours)", 0, max(0, total_hours - window), 0, step=24)
            sl = slice(start, start + window)

            x_axis = np.arange(window)
            fig, ax = plt.subplots(figsize=(14, 4))
            sns.lineplot(x=x_axis, y=y_act[sl], alpha=0.8, linewidth=0.9, label="Actual",    color="steelblue", ax=ax)
            sns.lineplot(x=x_axis, y=y_prd[sl], alpha=0.8, linewidth=0.9, label="Predicted", color="coral",     ax=ax)
            ax.set_xlabel("Hours")
            ax.set_ylabel("Demand (trips / hour)")
            ax.set_title(f"Station {station} — {pred_model_label}")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
