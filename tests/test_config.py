import pytest

from src import config


def test_target_is_not_a_feature():
    assert config.TARGET == "meter_reading"
    assert config.TARGET not in config.FEATURES


def test_feature_lists_are_unique_and_consistent():
    assert len(config.FEATURES) == len(set(config.FEATURES))
    assert set(config.FEATURES) == set(config.CATEGORICAL_FEATURES) | set(config.NUMERIC_FEATURES)
    assert len(config.FEATURES) == 24


def test_split_boundaries_are_ordered():
    assert config.TRAIN_END < config.VALID_END
    assert config.ELECTRICITY_METER == 0
    assert config.LOCAL_MAX_BUILDINGS == 120


def test_weather_columns_are_numeric_features():
    for column in config.WEATHER_NUMERIC:
        assert column in config.NUMERIC_FEATURES


@pytest.mark.parametrize("name", ["primary_use", "site_id", "building_id"])
def test_categorical_features_are_in_features(name):
    assert name in config.CATEGORICAL_FEATURES
    assert name in config.FEATURES
