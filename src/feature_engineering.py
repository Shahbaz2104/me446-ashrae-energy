"""Feature generation shared by training and inference.

The training dataset is built in SQL for efficiency; this module reproduces the
same calendar, cyclical, building-age, and weather-missingness features for
individual prediction records. A consistency check in the training script
verifies that both paths produce identical feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

METADATA_COLUMNS = ["site_id", "primary_use", "square_feet", "year_built", "floor_count"]
WEATHER_COLUMNS = config.WEATHER_NUMERIC + [
    "cloud_coverage_missing",
    "precip_depth_1_hr_missing",
    "sea_level_pressure_missing",
]
MISSING_FLAG_COLUMNS = [
    "cloud_coverage_missing",
    "precip_depth_1_hr_missing",
    "sea_level_pressure_missing",
]


def sanitize_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same physical-range rules used when preparing weather data."""
    frame = frame.copy()
    if "precip_depth_1_hr" in frame:
        frame.loc[frame["precip_depth_1_hr"] < 0, "precip_depth_1_hr"] = np.nan
    if "wind_speed" in frame:
        frame.loc[frame["wind_speed"] < 0, "wind_speed"] = np.nan
    if "sea_level_pressure" in frame:
        pressure = frame["sea_level_pressure"]
        frame.loc[(pressure < 800) | (pressure > 1100), "sea_level_pressure"] = np.nan
    return frame


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    timestamp = pd.to_datetime(frame["timestamp"])
    hour = timestamp.dt.hour.astype("int8")
    dayofweek = ((timestamp.dt.dayofweek + 1) % 7).astype("int8")
    month = timestamp.dt.month.astype("int8")
    frame["hour"] = hour
    frame["dayofweek"] = dayofweek
    frame["month"] = month
    frame["is_weekend"] = dayofweek.isin([0, 6]).astype("int8")
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    if "year_built" in frame:
        age = timestamp.dt.year - frame["year_built"]
        frame["building_age"] = age.where(age >= 0)
    return frame


def _merge_metadata(frame: pd.DataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in METADATA_COLUMNS if column not in frame.columns]
    if not missing_columns:
        return frame
    available = ["building_id"] + [column for column in missing_columns if column in buildings.columns]
    metadata = buildings[available].drop_duplicates("building_id")
    merged = frame.merge(metadata, on="building_id", how="left", validate="many_to_one")
    if merged["site_id"].isna().any():
        unknown = merged.loc[merged["site_id"].isna(), "building_id"].unique().tolist()
        raise ValueError(f"Unknown building_id values: {unknown}")
    return merged


def _merge_weather(frame: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["site_id", "timestamp"]
    weather_subset = weather[[*merge_keys, *WEATHER_COLUMNS]].copy()
    merged = frame.merge(
        weather_subset,
        on=merge_keys,
        how="left",
        suffixes=("", "_weather"),
        validate="many_to_one",
    )
    for column in config.WEATHER_NUMERIC:
        weather_column = f"{column}_weather"
        if weather_column in merged:
            merged[column] = merged[column].combine_first(merged[weather_column])
            merged = merged.drop(columns=[weather_column])
    for column in MISSING_FLAG_COLUMNS:
        weather_column = f"{column}_weather"
        if weather_column in merged:
            merged[column] = merged[column].combine_first(merged[weather_column])
            merged = merged.drop(columns=[weather_column])
    return merged


def build_feature_frame(
    records: pd.DataFrame,
    buildings: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert raw prediction records into the exact training feature layout."""
    frame = records.copy()
    if "timestamp" not in frame or "building_id" not in frame:
        raise ValueError("records must contain building_id and timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if buildings is not None:
        frame = _merge_metadata(frame, buildings)
    if "site_id" not in frame:
        raise ValueError("site_id is required; supply building metadata")

    frame = sanitize_weather(frame)
    if weather is not None:
        frame = _merge_weather(frame, weather)

    frame = add_calendar_features(frame)
    for column in config.WEATHER_NUMERIC:
        if column not in frame:
            frame[column] = np.nan
    for column in MISSING_FLAG_COLUMNS:
        if column not in frame:
            frame[column] = frame[column.replace("_missing", "")].isna().astype("int8")
        else:
            frame[column] = frame[column].fillna(1).astype("int8")

    missing = [column for column in config.FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Could not construct required features: {missing}")
    return frame[config.FEATURES]


def assert_feature_consistency(
    engineered: pd.DataFrame, reference: pd.DataFrame, atol: float = 1e-6
) -> None:
    """Raise when SQL-built and Python-built features disagree."""
    for column in config.FEATURES:
        left = engineered[column]
        right = reference[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(dtype="float64"),
                right.to_numpy(dtype="float64"),
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"Feature mismatch in column {column}")
        elif not left.astype(str).equals(right.astype(str)):
            raise AssertionError(f"Feature mismatch in column {column}")
