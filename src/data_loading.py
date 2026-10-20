"""Data acquisition helpers and typed loaders for the ASHRAE dataset."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src import config

REQUIRED_RAW_FILES = (
    config.TRAIN_CSV,
    config.TEST_CSV,
    config.BUILDINGS_CSV,
    config.WEATHER_TRAIN_CSV,
    config.WEATHER_TEST_CSV,
)


def verify_raw_files() -> None:
    """Raise a clear error when the expected competition files are absent."""
    missing = [str(path) for path in REQUIRED_RAW_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw dataset files:\n  "
            + "\n  ".join(missing)
            + "\nDownload the Kaggle dataset into data/raw before continuing."
        )


def _copy_query(con: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({query}) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def convert_raw_to_parquet(force: bool = False) -> None:
    """Convert the large CSVs to typed Parquet so later stages avoid CSV scans."""
    verify_raw_files()
    config.ensure_directories()
    targets = {
        config.TRAIN_PARQUET: config.TRAIN_CSV,
        config.TEST_PARQUET: config.TEST_CSV,
        config.BUILDINGS_PARQUET: config.BUILDINGS_CSV,
        config.WEATHER_TRAIN_PARQUET: config.WEATHER_TRAIN_CSV,
        config.WEATHER_TEST_PARQUET: config.WEATHER_TEST_CSV,
    }
    if not force and all(path.exists() for path in targets):
        return

    con = duckdb.connect()
    _copy_query(
        con,
        f"""
        SELECT CAST(building_id AS SMALLINT) AS building_id,
               CAST(meter AS TINYINT) AS meter,
               timestamp,
               CAST(meter_reading AS DOUBLE) AS meter_reading
        FROM read_csv_auto('{config.TRAIN_CSV}', header=true)
        """,
        config.TRAIN_PARQUET,
    )
    _copy_query(
        con,
        f"""
        SELECT CAST(row_id AS INTEGER) AS row_id,
               CAST(building_id AS SMALLINT) AS building_id,
               CAST(meter AS TINYINT) AS meter,
               timestamp
        FROM read_csv_auto('{config.TEST_CSV}', header=true)
        """,
        config.TEST_PARQUET,
    )
    _copy_query(
        con,
        f"""
        SELECT CAST(building_id AS SMALLINT) AS building_id,
               CAST(site_id AS TINYINT) AS site_id,
               primary_use,
               CAST(square_feet AS DOUBLE) AS square_feet,
               CAST(year_built AS SMALLINT) AS year_built,
               CAST(floor_count AS SMALLINT) AS floor_count
        FROM read_csv_auto('{config.BUILDINGS_CSV}', header=true)
        """,
        config.BUILDINGS_PARQUET,
    )
    for source, destination in (
        (config.WEATHER_TRAIN_CSV, config.WEATHER_TRAIN_PARQUET),
        (config.WEATHER_TEST_CSV, config.WEATHER_TEST_PARQUET),
    ):
        _copy_query(
            con,
            f"""
            SELECT CAST(site_id AS TINYINT) AS site_id,
                   timestamp,
                   CAST(air_temperature AS DOUBLE) AS air_temperature,
                   CAST(cloud_coverage AS DOUBLE) AS cloud_coverage,
                   CAST(dew_temperature AS DOUBLE) AS dew_temperature,
                   CAST(precip_depth_1_hr AS DOUBLE) AS precip_depth_1_hr,
                   CAST(sea_level_pressure AS DOUBLE) AS sea_level_pressure,
                   CAST(wind_direction AS DOUBLE) AS wind_direction,
                   CAST(wind_speed AS DOUBLE) AS wind_speed
            FROM read_csv_auto('{source}', header=true)
            """,
            destination,
        )
    con.close()


def load_train(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.TRAIN_PARQUET, columns=columns)


def load_test(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.TEST_PARQUET, columns=columns)


def load_buildings() -> pd.DataFrame:
    return pd.read_parquet(config.BUILDINGS_PARQUET)


def load_weather(split: str = "train") -> pd.DataFrame:
    path = config.WEATHER_TRAIN_PARQUET if split == "train" else config.WEATHER_TEST_PARQUET
    return pd.read_parquet(path)


def load_model_dataset(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=columns)


def load_cleaned(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.CLEANED_PARQUET, columns=columns)


if __name__ == "__main__":
    convert_raw_to_parquet()
    print("Typed Parquet files are ready in data/interim/.")
