import numpy as np
import pandas as pd

from src.evaluation import reading_band_metrics, regression_metrics


def test_regression_metrics_perfect_prediction():
    metrics = regression_metrics(np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0, 20.0]))
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["rmsle"] == 0.0
    assert metrics["n_rows"] == 3


def test_regression_metrics_clips_negative_predictions():
    metrics = regression_metrics(np.array([0.0, 10.0]), np.array([-5.0, 10.0]))
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["predicted_mean"] == 5.0


def test_regression_metrics_are_finite():
    metrics = regression_metrics(np.array([5.0, 15.0, 100.0]), np.array([10.0, 10.0, 80.0]))
    for key in ("mae", "rmse", "r2", "rmsle"):
        assert np.isfinite(metrics[key])
    assert metrics["mae"] > 0


def test_reading_band_metrics_returns_expected_columns():
    frame = pd.DataFrame(
        {
            "actual": [5.0, 20.0, 75.0, 200.0, 600.0] * 20,
            "predicted": [4.0, 25.0, 70.0, 220.0, 550.0] * 20,
        }
    )
    result = reading_band_metrics(frame)
    assert {"reading_band", "mae", "rmse", "r2", "rmsle", "n_rows"}.issubset(result.columns)
    assert len(result) >= 3
