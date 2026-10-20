import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import (
    add_calendar_features,
    build_feature_frame,
    sanitize_weather,
)


def test_sanitize_weather_removes_impossible_values():
    frame = pd.DataFrame(
        {
            "precip_depth_1_hr": [-1.0, 0.0, 2.0],
            "wind_speed": [-3.0, 4.0, 5.0],
            "sea_level_pressure": [1200.0, 1010.0, 700.0],
        }
    )
    cleaned = sanitize_weather(frame)
    assert np.isnan(cleaned.loc[0, "precip_depth_1_hr"])
    assert cleaned.loc[1, "precip_depth_1_hr"] == 0.0
    assert np.isnan(cleaned.loc[0, "wind_speed"])
    assert cleaned.loc[1, "wind_speed"] == 4.0
    assert np.isnan(cleaned.loc[0, "sea_level_pressure"])
    assert np.isnan(cleaned.loc[2, "sea_level_pressure"])
    assert cleaned.loc[1, "sea_level_pressure"] == 1010.0


def test_calendar_features_use_sunday_zero_convention():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2016-12-04 13:00:00", "2016-12-05 13:00:00"]
            )
        }
    )
    result = add_calendar_features(frame)
    assert result.loc[0, "dayofweek"] == 0
    assert result.loc[0, "is_weekend"] == 1
    assert result.loc[1, "dayofweek"] == 1
    assert result.loc[1, "is_weekend"] == 0
    assert result.loc[0, "hour"] == 13
    assert result.loc[0, "month"] == 12
    assert result.loc[0, "hour_sin"] == pytest.approx(np.sin(2 * np.pi * 13 / 24))
    assert result.loc[0, "hour_cos"] == pytest.approx(np.cos(2 * np.pi * 13 / 24))
    assert result.loc[0, "month_sin"] == pytest.approx(np.sin(2 * np.pi * 11 / 12))


def test_build_feature_frame_matches_training_layout():
    buildings = pd.DataFrame(
        {
            "building_id": [1, 2],
            "site_id": [0, 1],
            "primary_use": ["Office", "Education"],
            "square_feet": [1000.0, 2000.0],
            "year_built": [2000.0, np.nan],
            "floor_count": [2.0, np.nan],
        }
    )
    weather = pd.DataFrame(
        {
            "site_id": [0, 0],
            "timestamp": pd.to_datetime(
                ["2016-01-04 00:00:00", "2016-01-04 01:00:00"]
            ),
            "air_temperature": [10.0, 11.0],
            "dew_temperature": [5.0, 6.0],
            "wind_speed": [2.0, 2.0],
            "cloud_coverage": [np.nan, 4.0],
            "precip_depth_1_hr": [0.0, np.nan],
            "sea_level_pressure": [1010.0, 1011.0],
            "cloud_coverage_missing": [1, 0],
            "precip_depth_1_hr_missing": [0, 1],
            "sea_level_pressure_missing": [0, 0],
        }
    )
    records = pd.DataFrame(
        {
            "building_id": [1, 2],
            "timestamp": pd.to_datetime(
                ["2016-01-04 00:00:00", "2016-01-04 01:00:00"]
            ),
        }
    )
    features = build_feature_frame(records, buildings=buildings, weather=weather)

    assert list(features.columns) == config.FEATURES
    assert len(features) == 2
    assert features.loc[0, "primary_use"] == "Office"
    assert features.loc[0, "site_id"] == 0
    assert features.loc[0, "square_feet"] == 1000.0
    assert features.loc[0, "building_age"] == 16
    assert features.loc[0, "hour"] == 0
    assert features.loc[0, "is_weekend"] == 0
    assert features.loc[0, "air_temperature"] == 10.0
    assert features.loc[0, "cloud_coverage_missing"] == 1
    assert features.loc[1, "primary_use"] == "Education"
    assert features.loc[1, "site_id"] == 1
    assert np.isnan(features.loc[1, "building_age"])
    assert np.isnan(features.loc[1, "air_temperature"])
    assert features.loc[1, "cloud_coverage_missing"] == 1
    assert features.loc[1, "precip_depth_1_hr_missing"] == 1
    assert features.loc[1, "sea_level_pressure_missing"] == 1


def test_build_feature_frame_rejects_unknown_building():
    buildings = pd.DataFrame(
        {
            "building_id": [1],
            "site_id": [0],
            "primary_use": ["Office"],
            "square_feet": [1000.0],
            "year_built": [2000.0],
            "floor_count": [2.0],
        }
    )
    records = pd.DataFrame({"building_id": [99], "timestamp": ["2016-01-04 00:00:00"]})
    with pytest.raises(ValueError, match="Unknown building_id"):
        build_feature_frame(records, buildings=buildings)


def test_build_feature_frame_requires_keys():
    with pytest.raises(ValueError, match="building_id and timestamp"):
        build_feature_frame(pd.DataFrame({"building_id": [1]}))
