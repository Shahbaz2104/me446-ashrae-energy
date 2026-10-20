from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.predict import ElectricityPredictor


@pytest.fixture(scope="module")
def predictor():
    model_path = config.MODELS_DIR / config.MODEL_FILENAME
    metadata_path = config.MODELS_DIR / config.METADATA_FILENAME
    weather_path = config.DATA_INTERIM / "weather_train_filled.parquet"
    if not all(path.exists() for path in (model_path, metadata_path, weather_path)):
        pytest.skip("Run the preprocessing and training pipeline before inference tests.")
    return ElectricityPredictor(model_path=model_path, metadata_path=metadata_path)


def test_saved_model_metadata_is_consistent(predictor):
    assert predictor.metadata["model_name"] == "HistGradientBoosting"
    assert predictor.metadata["target"] == config.TARGET
    assert predictor.metadata["feature_consistency_check"] == "passed"
    assert set(predictor.metadata["features"]) == set(config.FEATURES)


def test_predictor_returns_finite_nonnegative_values(predictor):
    records = pd.DataFrame(
        {
            "building_id": [1088],
            "timestamp": ["2016-12-19 15:00:00"],
        }
    )
    prediction = predictor.predict(records)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
    assert prediction[0] >= 0
    assert prediction[0] < 10000


def test_predictor_requires_building_and_timestamp(predictor):
    with pytest.raises(ValueError, match="building_id and timestamp"):
        predictor.predict(pd.DataFrame({"building_id": [1088]}))
