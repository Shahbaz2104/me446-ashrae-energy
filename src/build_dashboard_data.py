"""Create dashboard-ready CSV extracts for Tableau and Streamlit."""

from __future__ import annotations

import shutil

import pandas as pd

from src import config

TABLEAU_DIR = config.DATA_DASHBOARD / "tableau_data"


def build_electricity_extract() -> pd.DataFrame:
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
        "dew_temperature",
        "wind_speed",
        "cloud_coverage",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "split",
    ]
    frame = pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=columns)
    frame = frame.sort_values(["timestamp", "building_id"]).reset_index(drop=True)
    frame.to_csv(TABLEAU_DIR / "electricity_hourly.csv", index=False)
    return frame


def build_aggregates(frame: pd.DataFrame) -> None:
    building = (
        frame.groupby(["building_id", "site_id", "primary_use"], observed=True)
        .agg(
            mean_reading=("meter_reading", "mean"),
            median_reading=("meter_reading", "median"),
            total_reading=("meter_reading", "sum"),
            observations=("meter_reading", "size"),
            square_feet=("square_feet", "first"),
            year_built=("year_built", "first"),
            floor_count=("floor_count", "first"),
        )
        .reset_index()
    )
    building.to_csv(TABLEAU_DIR / "building_summary.csv", index=False)

    monthly_site = (
        frame.assign(month_start=frame["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month_start", "site_id"], observed=True)
        .agg(
            total_reading=("meter_reading", "sum"),
            mean_reading=("meter_reading", "mean"),
            observations=("meter_reading", "size"),
            mean_air_temperature=("air_temperature", "mean"),
            buildings=("building_id", "nunique"),
        )
        .reset_index()
    )
    monthly_site.to_csv(TABLEAU_DIR / "monthly_site_summary.csv", index=False)

    monthly_use = (
        frame.assign(month_start=frame["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month_start", "primary_use"], observed=True)
        .agg(
            total_reading=("meter_reading", "sum"),
            mean_reading=("meter_reading", "mean"),
            observations=("meter_reading", "size"),
            buildings=("building_id", "nunique"),
        )
        .reset_index()
    )
    monthly_use.to_csv(TABLEAU_DIR / "monthly_use_summary.csv", index=False)


def copy_tables() -> None:
    copies = {
        config.TABLES_DIR / "model_comparison.csv": TABLEAU_DIR / "model_comparison.csv",
        config.TABLES_DIR / "error_by_primary_use.csv": TABLEAU_DIR / "error_by_primary_use.csv",
        config.TABLES_DIR / "error_by_site.csv": TABLEAU_DIR / "error_by_site.csv",
        config.TABLES_DIR / "error_by_reading_band.csv": TABLEAU_DIR / "error_by_reading_band.csv",
        config.TABLES_DIR / "unseen_case_predictions.csv": TABLEAU_DIR / "unseen_case_predictions.csv",
        config.TABLES_DIR / "generalization_building_holdout.csv": TABLEAU_DIR
        / "generalization_building_holdout.csv",
        config.TABLES_DIR / "feature_importance_histgradientboosting.csv": TABLEAU_DIR
        / "feature_importance.csv",
        config.DATA_DASHBOARD / "model_test_predictions.csv": TABLEAU_DIR
        / "model_test_predictions.csv",
    }
    for source, destination in copies.items():
        if source.exists():
            shutil.copy2(source, destination)
        else:
            print(f"Warning: optional dashboard source missing: {source}")


def main() -> None:
    config.ensure_directories()
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_electricity_extract()
    build_aggregates(frame)
    copy_tables()
    print(f"Dashboard extracts written to {TABLEAU_DIR}")
    print(f"Electricity rows: {len(frame):,}")


if __name__ == "__main__":
    main()
