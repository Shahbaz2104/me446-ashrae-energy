"""Reusable inference interface and two previously unseen demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import assert_feature_consistency, build_feature_frame

RAW_RECORD_COLUMNS = [
    "building_id",
    "timestamp",
    "site_id",
    "primary_use",
    "square_feet",
    "year_built",
    "floor_count",
    "air_temperature",
    "dew_temperature",
    "wind_speed",
    "cloud_coverage",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "cloud_coverage_missing",
    "precip_depth_1_hr_missing",
    "sea_level_pressure_missing",
]


class ElectricityPredictor:
    """Load a fitted pipeline and predict electricity readings from raw records."""

    def __init__(
        self,
        model_path: Path = config.MODELS_DIR / config.MODEL_FILENAME,
        metadata_path: Path = config.MODELS_DIR / config.METADATA_FILENAME,
    ) -> None:
        self.model = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.buildings = pd.read_parquet(config.BUILDINGS_PARQUET)
        self.weather = pd.read_parquet(config.DATA_INTERIM / "weather_train_filled.parquet")

    def predict(self, records: pd.DataFrame) -> np.ndarray:
        features = build_feature_frame(
            records, buildings=self.buildings, weather=self.weather
        )
        prediction = self.model.predict(features)
        return np.clip(np.asarray(prediction, dtype="float64"), 0, None)


def select_demonstration_cases() -> pd.DataFrame:
    predictions = pd.read_csv(config.TABLES_DIR / "test_predictions_best_model.csv")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    median_error = predictions["absolute_error"].median()
    reliable = predictions[predictions["absolute_error"] <= median_error]

    high = reliable[reliable["actual"] >= reliable["actual"].quantile(0.95)].sort_values(
        "actual", ascending=False
    )
    case_one = high.iloc[[0]] if len(high) else reliable.sort_values("actual").iloc[[-1]]
    building_one = case_one["building_id"].iloc[0]

    typical = reliable[
        (reliable["building_id"] != building_one)
        & (reliable["actual"] >= reliable["actual"].quantile(0.25))
        & (reliable["actual"] <= reliable["actual"].quantile(0.75))
    ].sort_values("absolute_error")
    case_two = typical.iloc[[0]] if len(typical) else reliable.iloc[[-1]]
    return pd.concat([case_one, case_two], ignore_index=True)


def load_raw_rows(cases: pd.DataFrame) -> pd.DataFrame:
    columns = list(dict.fromkeys([*RAW_RECORD_COLUMNS, config.TARGET, *config.FEATURES]))
    dataset = pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=columns)
    keys = cases[["building_id", "timestamp"]].copy()
    keys["timestamp"] = pd.to_datetime(keys["timestamp"])
    merged = keys.merge(
        dataset,
        on=["building_id", "timestamp"],
        how="left",
        suffixes=("", "_dataset"),
    )
    if merged[config.TARGET].isna().any():
        raise ValueError("A demonstration case could not be located in the processed dataset.")
    return merged


def demonstrate_unseen_cases() -> pd.DataFrame:
    predictor = ElectricityPredictor()
    cases = select_demonstration_cases()
    raw = load_raw_rows(cases)
    records = raw[RAW_RECORD_COLUMNS].copy()

    engineered = build_feature_frame(records, buildings=predictor.buildings)
    assert_feature_consistency(engineered, raw[config.FEATURES])

    predictions = predictor.predict(records)
    output = pd.DataFrame(
        {
            "case_id": ["case_1_high_demand", "case_2_typical_demand"][: len(raw)],
            "building_id": raw["building_id"].to_numpy(),
            "timestamp": raw["timestamp"].to_numpy(),
            "site_id": raw["site_id"].to_numpy(),
            "primary_use": raw["primary_use"].astype(str).to_numpy(),
            "square_feet": raw["square_feet"].to_numpy(),
            "air_temperature_c": raw["air_temperature"].to_numpy(),
            "actual_reading": raw[config.TARGET].to_numpy(),
            "predicted_reading": predictions,
            "absolute_error": np.abs(raw[config.TARGET].to_numpy() - predictions),
        }
    )
    output["percentage_error"] = (
        output["absolute_error"] / output["actual_reading"].replace(0, np.nan) * 100
    )
    output.to_csv(config.TABLES_DIR / "unseen_case_predictions.csv", index=False)
    (config.TABLES_DIR / "unseen_case_predictions.md").write_text(
        output.to_markdown(index=False, floatfmt=".2f"), encoding="utf-8"
    )

    raw["case_id"] = output["case_id"].to_numpy()
    raw.to_csv(config.DATA_PROCESSED / "unseen_cases.csv", index=False)
    return output


def main() -> None:
    config.ensure_directories()
    output = demonstrate_unseen_cases()
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
