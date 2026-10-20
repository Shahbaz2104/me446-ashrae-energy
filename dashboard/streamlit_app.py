"""Supplementary Streamlit dashboard for the ME-446 ASHRAE CEP.

Run from the project root:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.predict import ElectricityPredictor

DATA_DIR = config.DATA_DASHBOARD / "tableau_data"
BLUE = "#4C72B0"
GREEN = "#55A868"
ORANGE = "#DD8452"
RED = "#C44E52"
GREY = "#8C8C8C"

st.set_page_config(
    page_title="ME-446 ASHRAE Energy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E6E8EC;
        border-radius: 10px; padding: 12px 14px;}
    div[data-testid="stMetricLabel"] {font-size: 0.8rem; color: #5B6470;}
    .scope-note {color: #5B6470; font-size: 0.82rem; margin-bottom: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading electricity observations...")
def load_hourly() -> pd.DataFrame:
    columns = [
        "timestamp",
        "building_id",
        "site_id",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "meter_reading",
        "air_temperature",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
    ]
    frame = pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=columns)
    frame["primary_use"] = frame["primary_use"].astype(str)
    return frame


@st.cache_data(show_spinner="Loading model results...")
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


@st.cache_data(show_spinner="Loading predictions...")
def load_predictions() -> pd.DataFrame:
    parquet_path = DATA_DIR / "model_test_predictions.parquet"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    else:
        frame = pd.read_csv(DATA_DIR / "model_test_predictions.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["primary_use"] = frame["primary_use"].astype(str)
    return frame


@st.cache_data(show_spinner="Loading building metadata...")
def load_buildings() -> pd.DataFrame:
    return pd.read_parquet(config.BUILDINGS_PARQUET)


@st.cache_resource(show_spinner="Loading trained model...")
def load_predictor() -> ElectricityPredictor:
    return ElectricityPredictor()


def fmt_reading(value: float) -> str:
    return f"{value:,.1f}"


def filter_hourly(
    frame: pd.DataFrame,
    date_range: tuple,
    sites: list[int],
    uses: list[str],
) -> pd.DataFrame:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23)
    mask = frame["timestamp"].between(start, end)
    if sites:
        mask &= frame["site_id"].isin(sites)
    if uses:
        mask &= frame["primary_use"].isin(uses)
    return frame[mask]


def filter_predictions(
    frame: pd.DataFrame, date_range: tuple, sites: list[int], uses: list[str]
) -> pd.DataFrame:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1]) + pd.Timedelta(hours=23)
    mask = frame["timestamp"].between(start, end)
    if sites:
        mask &= frame["site_id"].isin(sites)
    if uses:
        mask &= frame["primary_use"].isin(uses)
    return frame[mask]


def metric_row(frame: pd.DataFrame) -> None:
    columns = st.columns(5)
    columns[0].metric("Total reading sum", fmt_reading(frame["meter_reading"].sum()))
    columns[1].metric("Mean per observation", fmt_reading(frame["meter_reading"].mean()))
    columns[2].metric("Buildings", f"{frame['building_id'].nunique():,}")
    columns[3].metric("Sites", f"{frame['site_id'].nunique():,}")
    columns[4].metric("Observations", f"{len(frame):,}")


def overview_page(hourly: pd.DataFrame) -> None:
    st.title("Energy Overview")
    st.markdown(
        '<p class="scope-note">Site-stratified subset: 120 buildings, 16 sites, 2016. '
        "Readings are dataset energy units (documented as kWh). The sum of readings is "
        "an energy-like total, not instantaneous power, efficiency, or savings.</p>",
        unsafe_allow_html=True,
    )
    if hourly.empty:
        st.warning("No observations match the current filters.")
        return
    metric_row(hourly)

    monthly = (
        hourly.assign(month_start=hourly["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("month_start", observed=True)["meter_reading"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    daily = (
        hourly.assign(day=hourly["timestamp"].dt.floor("D"))
        .groupby("day", observed=True)["meter_reading"]
        .mean()
        .reset_index()
    )
    daily["rolling_7d"] = daily["meter_reading"].rolling(7, min_periods=1).mean()

    left, right = st.columns(2)
    with left:
        figure = px.line(
            monthly,
            x="month_start",
            y="mean",
            markers=True,
            title="Monthly mean reading per observation",
            labels={"month_start": "Month", "mean": "Mean reading (dataset units)"},
        )
        figure.update_traces(line_color=BLUE)
        figure.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")
    with right:
        figure = px.line(
            daily,
            x="day",
            y=["meter_reading", "rolling_7d"],
            title="Daily mean and 7-day rolling mean",
            labels={"value": "Mean reading (dataset units)", "day": "Date", "variable": ""},
        )
        figure.update_traces(line_color=GREY, opacity=0.5, selector=dict(name="meter_reading"))
        figure.update_traces(line_color=RED, selector=dict(name="rolling_7d"))
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        hourly_profile = (
            hourly.groupby(["hour", "is_weekend"], observed=True)["meter_reading"]
            .mean()
            .reset_index()
        )
        hourly_profile["Day type"] = np.where(
            hourly_profile["is_weekend"] == 1, "Weekend", "Weekday"
        )
        figure = px.line(
            hourly_profile,
            x="hour",
            y="meter_reading",
            color="Day type",
            markers=True,
            title="Mean reading by hour and day type",
            labels={"hour": "Hour of day", "meter_reading": "Mean reading (dataset units)"},
        )
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")
    with right:
        site = (
            hourly.groupby("site_id", observed=True)["meter_reading"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        figure = px.bar(
            site,
            x="site_id",
            y="meter_reading",
            title="Mean reading by site",
            labels={"site_id": "Site ID", "meter_reading": "Mean reading (dataset units)"},
        )
        figure.update_traces(marker_color=GREEN)
        figure.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")


def weather_page(hourly: pd.DataFrame) -> None:
    st.title("Weather and Building Analysis")
    st.markdown(
        '<p class="scope-note">Descriptive associations, not causal effects. Sites with '
        "different climates and building mixes can show opposite temperature responses.</p>",
        unsafe_allow_html=True,
    )
    if hourly.empty:
        st.warning("No observations match the current filters.")
        return

    sample = hourly.dropna(subset=["air_temperature"]).sample(
        min(30_000, len(hourly)), random_state=config.RANDOM_SEED
    )
    left, right = st.columns(2)
    with left:
        figure = px.density_heatmap(
            sample,
            x="air_temperature",
            y="meter_reading",
            nbinsx=50,
            nbinsy=50,
            title="Temperature versus reading density",
            labels={
                "air_temperature": "Outdoor air temperature (°C)",
                "meter_reading": "Reading (dataset units)",
            },
        )
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")
    with right:
        binned = hourly.dropna(subset=["air_temperature"]).copy()
        binned["temp_bin"] = (binned["air_temperature"] / 2).round() * 2
        grouped = (
            binned.groupby("temp_bin", observed=True)["meter_reading"]
            .mean()
            .reset_index()
        )
        figure = px.line(
            grouped,
            x="temp_bin",
            y="meter_reading",
            markers=True,
            title="Mean reading in 2°C temperature bins",
            labels={"temp_bin": "Air temperature (°C)", "meter_reading": "Mean reading (dataset units)"},
        )
        figure.update_traces(line_color=ORANGE)
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        heat = (
            hourly.groupby(["hour", "month"], observed=True)["meter_reading"]
            .mean()
            .reset_index()
        )
        pivot = heat.pivot(index="hour", columns="month", values="meter_reading")
        figure = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Mean reading by hour and month",
            labels={"x": "Month", "y": "Hour of day", "color": "Mean reading"},
        )
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")
    with right:
        use = (
            hourly.groupby("primary_use", observed=True)["meter_reading"]
            .agg(["median", "mean", "count"])
            .sort_values("median")
            .reset_index()
        )
        figure = px.bar(
            use,
            x="median",
            y="primary_use",
            orientation="h",
            title="Median reading by primary use",
            labels={"median": "Median reading (dataset units)", "primary_use": ""},
            hover_data={"mean": ":.1f", "count": ":,"},
        )
        figure.update_traces(marker_color=BLUE)
        figure.update_layout(margin=dict(t=50, b=10), height=430)
        st.plotly_chart(figure, width="stretch")

    buildings = load_table("building_summary.csv")
    if not buildings.empty:
        figure = px.scatter(
            buildings,
            x="square_feet",
            y="mean_reading",
            color="primary_use",
            log_x=True,
            log_y=True,
            hover_data=["building_id", "site_id"],
            title="Building size versus mean electricity demand",
            labels={
                "square_feet": "Floor area (square feet, log scale)",
                "mean_reading": "Mean reading (log scale)",
                "primary_use": "Primary use",
            },
        )
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")


def performance_page(predictions: pd.DataFrame) -> None:
    st.title("Model Performance")
    st.markdown(
        '<p class="scope-note">Test period: December 2016, never used for training or '
        "model selection. Metrics are in dataset units. Positive residuals indicate "
        "underprediction.</p>",
        unsafe_allow_html=True,
    )
    comparison = load_table("model_comparison.csv")
    st.subheader("Model comparison")
    st.dataframe(
        comparison.style.format(
            {
                "mae": "{:.2f}",
                "rmse": "{:.2f}",
                "r2": "{:.3f}",
                "rmsle": "{:.3f}",
                "fit_seconds": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    test = comparison[comparison["split"] == "test"]
    if not test.empty:
        figure = px.bar(
            test,
            x="model",
            y=["mae", "rmse"],
            barmode="group",
            title="Test error by model",
            labels={"value": "Dataset units", "variable": "Metric", "model": ""},
        )
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")

    if predictions.empty:
        st.warning("No predictions match the current filters.")
        return
    model_name = st.selectbox(
        "Model for detailed diagnostics",
        sorted(predictions["model"].unique()),
        index=sorted(predictions["model"].unique()).index("HistGradientBoosting")
        if "HistGradientBoosting" in predictions["model"].unique()
        else 0,
    )
    selected = predictions[predictions["model"] == model_name]
    if selected.empty:
        st.warning("No predictions match the current filters for this model.")
        return

    sample = selected.sample(min(25_000, len(selected)), random_state=config.RANDOM_SEED)
    left, right = st.columns(2)
    with left:
        figure = px.scatter(
            sample,
            x="actual",
            y="predicted",
            opacity=0.25,
            title=f"Actual vs predicted — {model_name}",
            labels={"actual": "Actual (dataset units)", "predicted": "Predicted (dataset units)"},
        )
        limit = float(max(sample["actual"].quantile(0.995), sample["predicted"].quantile(0.995)))
        figure.add_trace(
            go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="Perfect prediction",
                       line=dict(color=RED, width=1.5))
        )
        figure.update_xaxes(range=[0, limit])
        figure.update_yaxes(range=[0, limit])
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")
    with right:
        figure = px.histogram(
            sample,
            x="residual",
            nbins=70,
            title=f"Residual distribution — {model_name}",
            labels={"residual": "Residual = actual − predicted (dataset units)"},
        )
        figure.update_traces(marker_color=ORANGE)
        figure.add_vline(x=0, line_color=RED, line_width=1)
        figure.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        use_error = load_table("error_by_primary_use.csv").sort_values("mae", ascending=False)
        figure = px.bar(
            use_error,
            x="mae",
            y="primary_use",
            orientation="h",
            title="Test MAE by primary use",
            hover_data=["n_rows"],
            labels={"mae": "MAE (dataset units)", "primary_use": ""},
        )
        figure.update_traces(marker_color=GREEN)
        figure.update_layout(margin=dict(t=50, b=10), height=430)
        st.plotly_chart(figure, width="stretch")
    with right:
        band = load_table("error_by_reading_band.csv")
        figure = px.bar(
            band,
            x="reading_band",
            y="mae",
            title="Test MAE by actual reading band",
            hover_data=["n_rows"],
            labels={"reading_band": "Actual reading band (dataset units)", "mae": "MAE"},
        )
        figure.update_traces(marker_color=BLUE)
        figure.update_layout(margin=dict(t=50, b=10), height=430)
        st.plotly_chart(figure, width="stretch")

    importance = load_table("feature_importance.csv").head(15).sort_values("importance_mean")
    figure = px.bar(
        importance,
        x="importance_mean",
        y="feature",
        orientation="h",
        title="Permutation importance — increase in MAE when shuffled",
        labels={"importance_mean": "ΔMAE (dataset units)", "feature": ""},
    )
    figure.update_traces(marker_color=ORANGE)
    figure.update_layout(margin=dict(t=50, b=10), height=430)
    st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Unseen case demonstrations")
        st.dataframe(load_table("unseen_case_predictions.csv"), width="stretch", hide_index=True)
    with right:
        st.subheader("Generalization to unseen buildings")
        generalization = load_table("generalization_building_holdout.csv")
        st.dataframe(generalization, width="stretch", hide_index=True)
        st.caption(
            "20% of buildings were excluded from training for this experiment. "
            "The model transfers poorly to buildings with no training history."
        )


def predict_page() -> None:
    st.title("Predict Electricity Reading")
    st.markdown(
        '<p class="scope-note">The prediction uses the saved HistGradientBoosting '
        "pipeline and the same feature engineering as training. It is a data-driven "
        "estimate, not a validated operational control signal.</p>",
        unsafe_allow_html=True,
    )
    predictor = load_predictor()
    buildings = load_buildings()
    covered = set(
        pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=["building_id"])[
            "building_id"
        ].unique()
    )
    choices = buildings[buildings["building_id"].isin(covered)].sort_values("building_id")
    if choices.empty:
        st.error("No buildings with electricity observations were found.")
        return

    left, middle, right = st.columns(3)
    with left:
        building_id = st.selectbox(
            "Building",
            choices["building_id"].tolist(),
            format_func=lambda value: f"Building {value}",
        )
    with middle:
        selected_date = st.date_input(
            "Date",
            value=pd.Timestamp("2016-12-15").date(),
            min_value=pd.Timestamp("2016-01-01").date(),
            max_value=pd.Timestamp("2016-12-31").date(),
        )
    with right:
        hour = st.slider("Hour of day", 0, 23, 12)

    timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=int(hour))
    metadata = buildings[buildings["building_id"] == building_id].iloc[0]
    record = pd.DataFrame(
        {
            "building_id": [int(building_id)],
            "timestamp": [timestamp],
            "site_id": [int(metadata["site_id"])],
            "primary_use": [str(metadata["primary_use"])],
            "square_feet": [float(metadata["square_feet"])],
            "year_built": [metadata["year_built"]],
            "floor_count": [metadata["floor_count"]],
        }
    )
    prediction = predictor.predict(record)[0]

    context, result = st.columns([1.4, 1])
    with context:
        st.subheader("Input context")
        st.write(
            f"**Building {int(building_id)}** · Site {int(metadata['site_id'])} · "
            f"{metadata['primary_use']}"
        )
        st.write(
            f"Floor area: {float(metadata['square_feet']):,.0f} sq ft · "
            f"Year built: {metadata['year_built'] if pd.notna(metadata['year_built']) else 'unknown'} · "
            f"Floors: {metadata['floor_count'] if pd.notna(metadata['floor_count']) else 'unknown'}"
        )
        st.write(f"Timestamp: **{timestamp:%Y-%m-%d %H:%M}**")
        weather = pd.read_parquet(config.DATA_INTERIM / "weather_train_filled.parquet")
        weather_row = weather[
            (weather["site_id"] == int(metadata["site_id"]))
            & (weather["timestamp"] == timestamp)
        ]
        if not weather_row.empty:
            st.write(
                f"Outdoor air temperature: {weather_row['air_temperature'].iloc[0]:.1f} °C · "
                f"Wind speed: {weather_row['wind_speed'].iloc[0]:.1f}"
            )
        else:
            st.caption("No weather observation at this exact site-hour; the model receives missing weather.")
    with result:
        st.metric("Predicted electricity reading", fmt_reading(float(prediction)))
        st.caption("Dataset energy units (documented as kWh).")

    st.divider()
    st.subheader("Previously unseen held-out cases")
    st.dataframe(
        load_table("unseen_case_predictions.csv"), width="stretch", hide_index=True
    )


def main() -> None:
    hourly_all = load_hourly()
    predictions_all = load_predictions()

    with st.sidebar:
        st.title("ME-446 Energy CEP")
        st.caption("ASHRAE Great Energy Predictor III · Fall 2026")
        page = st.radio(
            "View",
            ["Energy Overview", "Weather & Buildings", "Model Performance", "Predict"],
        )
        st.divider()
        st.subheader("Filters")
        minimum = hourly_all["timestamp"].min().date()
        maximum = hourly_all["timestamp"].max().date()
        date_range = st.date_input(
            "Date range",
            value=(minimum, maximum),
            min_value=minimum,
            max_value=maximum,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            selected_dates = date_range
        else:
            selected_dates = (minimum, maximum)
        sites = st.multiselect(
            "Sites", sorted(hourly_all["site_id"].unique().tolist()), default=[]
        )
        uses = st.multiselect(
            "Primary use",
            sorted(hourly_all["primary_use"].unique().tolist()),
            default=[],
        )
        st.divider()
        st.caption(
            "Scope: site-stratified subset (120 buildings, 16 sites, 2016). "
            "Full-data training is provided as a Colab notebook."
        )

    hourly = filter_hourly(hourly_all, selected_dates, sites, uses)
    predictions = filter_predictions(predictions_all, selected_dates, sites, uses)

    if page == "Energy Overview":
        overview_page(hourly)
    elif page == "Weather & Buildings":
        weather_page(hourly)
    elif page == "Model Performance":
        performance_page(predictions)
    else:
        predict_page()


if __name__ == "__main__":
    main()
