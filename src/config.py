"""Project-wide paths, constants, and modelling configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_DASHBOARD = PROJECT_ROOT / "data" / "dashboard"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
BUILDINGS_CSV = DATA_RAW / "building_metadata.csv"
WEATHER_TRAIN_CSV = DATA_RAW / "weather_train.csv"
WEATHER_TEST_CSV = DATA_RAW / "weather_test.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW / "sample_submission.csv"

TRAIN_PARQUET = DATA_INTERIM / "train.parquet"
TEST_PARQUET = DATA_INTERIM / "test.parquet"
BUILDINGS_PARQUET = DATA_INTERIM / "building_metadata.parquet"
WEATHER_TRAIN_PARQUET = DATA_INTERIM / "weather_train.parquet"
WEATHER_TEST_PARQUET = DATA_INTERIM / "weather_test.parquet"

CLEANED_PARQUET = DATA_PROCESSED / "clean_electricity.parquet"
MODEL_DATASET_PARQUET = DATA_PROCESSED / "model_dataset.parquet"
SPLIT_SUMMARY_JSON = DATA_PROCESSED / "split_summary.json"

RANDOM_SEED = 42
ELECTRICITY_METER = 0
METER_NAMES = {
    0: "Electricity",
    1: "Chilled water",
    2: "Steam",
    3: "Hot water",
}

TARGET = "meter_reading"

WEATHER_NUMERIC = [
    "air_temperature",
    "dew_temperature",
    "wind_speed",
    "cloud_coverage",
    "precip_depth_1_hr",
    "sea_level_pressure",
]

CATEGORICAL_FEATURES = ["primary_use", "site_id", "building_id"]

NUMERIC_FEATURES = [
    "square_feet",
    "year_built",
    "floor_count",
    "building_age",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    *WEATHER_NUMERIC,
    "cloud_coverage_missing",
    "precip_depth_1_hr_missing",
    "sea_level_pressure_missing",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TRAIN_END = "2016-09-30 23:59:59"
VALID_END = "2016-11-30 23:59:59"

LOCAL_MAX_BUILDINGS = 120
LOCAL_MAX_ROWS = 1_000_000
LOCAL_MODEL_SAMPLE_ROWS = 250_000

MODEL_FILENAME = "final_model.joblib"
METADATA_FILENAME = "model_metadata.json"


def ensure_directories() -> None:
    """Create all project output directories if they do not exist."""
    for directory in (
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_DASHBOARD,
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        DASHBOARD_DIR / "tableau_data",
    ):
        directory.mkdir(parents=True, exist_ok=True)
