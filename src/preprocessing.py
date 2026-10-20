"""Dataset selection, cleaning, feature generation, and leakage-safe splits."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src import config

HOURS_PER_YEAR = 8784


def candidate_buildings() -> pd.DataFrame:
    """Return buildings with at least 80% electricity coverage in 2016."""
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT b.building_id, b.site_id, b.primary_use, b.square_feet,
               b.year_built, b.floor_count, c.n_rows
        FROM read_parquet('{config.BUILDINGS_PARQUET}') b
        JOIN (
            SELECT building_id, count(*) AS n_rows
            FROM read_parquet('{config.TRAIN_PARQUET}')
            WHERE meter = {config.ELECTRICITY_METER}
            GROUP BY building_id
        ) c USING (building_id)
        WHERE c.n_rows >= {HOURS_PER_YEAR} * 0.80
        ORDER BY b.site_id, b.building_id
        """
    ).fetchdf()
    con.close()
    return frame


def _allocate_proportionally(sizes: pd.Series, total: int, minimum: int = 1) -> dict:
    """Allocate an integer budget proportionally with a minimum per group."""
    sizes = sizes.sort_index()
    if total < len(sizes):
        chosen = sizes.sort_values(ascending=False).head(total).index
        return {key: (1 if key in chosen else 0) for key in sizes.index}
    allocation = {key: min(minimum, int(value)) for key, value in sizes.items()}
    remaining = total - sum(allocation.values())
    if remaining < 0:
        raise ValueError("Budget is smaller than the required minimum allocation.")
    exact = sizes / sizes.sum() * remaining
    floors = np.floor(exact).astype(int)
    for key in sizes.index:
        allocation[key] += int(floors[key])
    remainder = remaining - int(floors.sum())
    if remainder > 0:
        order = (exact - floors).sort_values(ascending=False).index
        for key in order[:remainder]:
            allocation[key] += 1
    return allocation


def select_subset_buildings(
    n_buildings: int = config.LOCAL_MAX_BUILDINGS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Select a site-stratified, use-diverse subset of electricity buildings.

    The allocation is proportional to site size with at least two buildings per
    site. Within each site, buildings are sampled across primary-use groups so
    that the subset is not dominated by a single building category.
    """
    candidates = candidate_buildings()
    rng = np.random.default_rng(seed)
    site_sizes = candidates.groupby("site_id").size()
    site_allocation = _allocate_proportionally(site_sizes, n_buildings, minimum=2)

    selected_frames = []
    for site_id, site_frame in candidates.groupby("site_id"):
        budget = site_allocation[site_id]
        use_sizes = site_frame.groupby("primary_use").size()
        use_allocation = _allocate_proportionally(use_sizes, budget, minimum=1)
        for use, use_budget in use_allocation.items():
            group = site_frame[site_frame["primary_use"] == use]
            take = min(use_budget, len(group))
            indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
            selected_frames.append(group.loc[indices])
    subset = pd.concat(selected_frames).sort_values("building_id").reset_index(drop=True)
    subset["selection_stratum"] = (
        "site_" + subset["site_id"].astype(str) + "_use_" + subset["primary_use"]
    )
    return subset


def save_subset(subset: pd.DataFrame) -> Path:
    config.ensure_directories()
    path = config.DATA_PROCESSED / "subset_buildings.csv"
    subset.to_csv(path, index=False)
    metadata = {
        "selection_method": (
            "Site-stratified sample of buildings with >=80% electricity coverage; "
            "within each site, primary-use groups allocated proportionally with a "
            "minimum of one building per present group."
        ),
        "random_seed": config.RANDOM_SEED,
        "requested_buildings": config.LOCAL_MAX_BUILDINGS,
        "selected_buildings": int(len(subset)),
        "sites": int(subset["site_id"].nunique()),
        "primary_use_categories": int(subset["primary_use"].nunique()),
        "buildings_by_site": subset.groupby("site_id").size().to_dict(),
        "buildings_by_primary_use": subset.groupby("primary_use").size().to_dict(),
    }
    (config.DATA_PROCESSED / "subset_selection.json").write_text(
        json.dumps({k: (v if not isinstance(v, dict) else {str(i): j for i, j in v.items()})
                    for k, v in metadata.items()}, indent=2),
        encoding="utf-8",
    )
    return path


def prepare_weather(split: str = "train") -> Path:
    """Forward-fill short weather gaps and flag unresolved missing values.

    Only past observations are used (forward fill with a three-hour limit), so
    the transformation does not introduce future information. Negative
    precipitation values are treated as the documented missing sentinel and not
    as physical rainfall.
    """
    weather = pd.read_parquet(
        config.WEATHER_TRAIN_PARQUET if split == "train" else config.WEATHER_TEST_PARQUET
    )
    weather = weather.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    weather["precip_depth_1_hr"] = weather["precip_depth_1_hr"].where(
        weather["precip_depth_1_hr"] >= 0
    )
    weather.loc[weather["wind_speed"] < 0, "wind_speed"] = np.nan
    weather.loc[
        (weather["sea_level_pressure"] < 800) | (weather["sea_level_pressure"] > 1100),
        "sea_level_pressure",
    ] = np.nan

    missing_flags = pd.DataFrame(
        {
            "cloud_coverage_missing": weather["cloud_coverage"].isna().astype("int8"),
            "precip_depth_1_hr_missing": weather["precip_depth_1_hr"].isna().astype("int8"),
            "sea_level_pressure_missing": weather["sea_level_pressure"].isna().astype("int8"),
        }
    )
    weather = pd.concat([weather, missing_flags], axis=1)

    for column in config.WEATHER_NUMERIC:
        weather[column] = weather.groupby("site_id")[column].ffill(limit=3)

    destination = (
        config.DATA_INTERIM / "weather_train_filled.parquet"
        if split == "train"
        else config.DATA_INTERIM / "weather_test_filled.parquet"
    )
    weather.to_parquet(destination, index=False)
    return destination


def build_model_dataset(
    building_ids: list[int] | None = None,
    output_path: Path = config.MODEL_DATASET_PARQUET,
) -> dict:
    """Join readings, metadata, and weather, then derive model features.

    When ``building_ids`` is None the full electricity dataset is processed.
    Otherwise only the documented local subset is kept.
    """
    config.ensure_directories()
    weather_path = prepare_weather("train")
    building_filter = ""
    if building_ids is not None:
        ids = ",".join(str(int(value)) for value in building_ids)
        building_filter = f"AND t.building_id IN ({ids})"

    con = duckdb.connect()
    query = f"""
        SELECT t.building_id,
               t.timestamp,
               t.meter_reading,
               b.site_id,
               b.primary_use,
               b.square_feet,
               b.year_built,
               b.floor_count,
               CAST(extract(hour FROM t.timestamp) AS TINYINT) AS hour,
               CAST(extract(dow FROM t.timestamp) AS TINYINT) AS dayofweek,
               CAST(extract(month FROM t.timestamp) AS TINYINT) AS month,
               CASE WHEN extract(dow FROM t.timestamp) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
               sin(2 * pi() * extract(hour FROM t.timestamp) / 24.0) AS hour_sin,
               cos(2 * pi() * extract(hour FROM t.timestamp) / 24.0) AS hour_cos,
               sin(2 * pi() * (extract(month FROM t.timestamp) - 1) / 12.0) AS month_sin,
               cos(2 * pi() * (extract(month FROM t.timestamp) - 1) / 12.0) AS month_cos,
               CASE WHEN b.year_built IS NULL THEN NULL
                    WHEN extract(year FROM t.timestamp) - b.year_built < 0 THEN NULL
                    ELSE extract(year FROM t.timestamp) - b.year_built END AS building_age,
               w.air_temperature,
               w.dew_temperature,
               w.wind_speed,
               w.cloud_coverage,
               w.precip_depth_1_hr,
               w.sea_level_pressure,
               COALESCE(w.cloud_coverage_missing, 1) AS cloud_coverage_missing,
               COALESCE(w.precip_depth_1_hr_missing, 1) AS precip_depth_1_hr_missing,
               COALESCE(w.sea_level_pressure_missing, 1) AS sea_level_pressure_missing,
               CASE WHEN t.timestamp <= TIMESTAMP '{config.TRAIN_END}' THEN 'train'
                    WHEN t.timestamp <= TIMESTAMP '{config.VALID_END}' THEN 'validation'
                    ELSE 'test' END AS split
        FROM read_parquet('{config.TRAIN_PARQUET}') t
        JOIN read_parquet('{config.BUILDINGS_PARQUET}') b USING (building_id)
        LEFT JOIN read_parquet('{weather_path}') w USING (site_id, timestamp)
        WHERE t.meter = {config.ELECTRICITY_METER}
        {building_filter}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    summary = con.execute(
        f"""
        SELECT split, count(*) AS n_rows,
               count(DISTINCT building_id) AS n_buildings,
               min(timestamp) AS first_timestamp,
               max(timestamp) AS last_timestamp,
               avg(meter_reading) AS mean_reading,
               count(*) FILTER (WHERE meter_reading = 0) AS zero_readings
        FROM read_parquet('{output_path}')
        GROUP BY split ORDER BY split
        """
    ).fetchdf()
    con.close()

    result = {
        "scope": "full" if building_ids is None else "local_subset",
        "output": str(output_path),
        "total_rows": int(summary["n_rows"].sum()),
        "splits": summary.to_dict(orient="records"),
    }
    (config.DATA_PROCESSED / "dataset_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def main() -> None:
    subset = select_subset_buildings()
    save_subset(subset)
    result = build_model_dataset(subset["building_id"].tolist())
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
