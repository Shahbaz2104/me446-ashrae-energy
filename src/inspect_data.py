"""Structural audit of the raw ASHRAE files.

The audit deliberately uses DuckDB so that the 20.2 million-row training file
is scanned out-of-core. Results are written to reports/data_inventory.md and a
machine-readable JSON summary.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd

from src import config

TRAIN = f"read_parquet('{config.TRAIN_PARQUET}')"
TEST = f"read_parquet('{config.TEST_PARQUET}')"
BUILDINGS = f"read_parquet('{config.BUILDINGS_PARQUET}')"
WEATHER_TRAIN = f"read_parquet('{config.WEATHER_TRAIN_PARQUET}')"
WEATHER_TEST = f"read_parquet('{config.WEATHER_TEST_PARQUET}')"


def query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows returned._"
    return frame.to_markdown(index=False, floatfmt=".3f")


def file_sizes() -> pd.DataFrame:
    rows = []
    for path in (
        config.TRAIN_CSV,
        config.TEST_CSV,
        config.BUILDINGS_CSV,
        config.WEATHER_TRAIN_CSV,
        config.WEATHER_TEST_CSV,
        config.SAMPLE_SUBMISSION_CSV,
    ):
        rows.append(
            {
                "file": path.name,
                "size_mb": path.stat().st_size / 1_048_576 if path.exists() else None,
                "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").strftime(
                    "%Y-%m-%d %H:%M"
                )
                if path.exists()
                else None,
            }
        )
    return pd.DataFrame(rows)


def train_audit(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    structural = query(
        con,
        f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT building_id) AS n_buildings,
               count(DISTINCT meter) AS n_meters,
               min(timestamp) AS min_timestamp,
               max(timestamp) AS max_timestamp,
               count(DISTINCT timestamp) AS n_distinct_timestamps,
               count(*) FILTER (WHERE meter_reading IS NULL) AS null_target,
               count(*) FILTER (WHERE meter_reading < 0) AS negative_target,
               count(*) FILTER (WHERE meter_reading = 0) AS zero_target
        FROM {TRAIN}
        """,
    )

    duplicates = query(
        con,
        f"""
        SELECT (SELECT count(*) FROM {TRAIN})
             - (SELECT count(*) FROM (SELECT DISTINCT * FROM {TRAIN})) AS duplicate_full_rows,
               (SELECT count(*) FROM {TRAIN})
             - (SELECT count(*) FROM (
                   SELECT DISTINCT building_id, meter, timestamp FROM {TRAIN}
               )) AS duplicate_business_keys
        """,
    )

    meter_table = query(
        con,
        f"""
        SELECT meter,
               count(*) AS n_rows,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_rows,
               count(DISTINCT building_id) AS n_buildings,
               min(meter_reading) AS min_reading,
               quantile_cont(meter_reading, 0.50) AS median_reading,
               max(meter_reading) AS max_reading,
               avg(meter_reading) AS mean_reading,
               count(*) FILTER (WHERE meter_reading = 0) AS n_zeros,
               round(100.0 * count(*) FILTER (WHERE meter_reading = 0) / count(*), 2) AS pct_zeros
        FROM {TRAIN}
        GROUP BY meter
        ORDER BY meter
        """,
    )

    quantiles = query(
        con,
        f"""
        SELECT quantile_cont(meter_reading, [0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999])
                   AS meter_reading_quantiles
        FROM {TRAIN}
        """,
    )

    electricity = query(
        con,
        f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT building_id) AS n_buildings,
               count(*) FILTER (WHERE meter_reading = 0) AS n_zeros,
               round(100.0 * count(*) FILTER (WHERE meter_reading = 0) / count(*), 2) AS pct_zeros,
               min(meter_reading) AS min_reading,
               quantile_cont(meter_reading, 0.50) AS median_reading,
               avg(meter_reading) AS mean_reading,
               quantile_cont(meter_reading, 0.99) AS p99_reading,
               max(meter_reading) AS max_reading
        FROM {TRAIN}
        WHERE meter = {config.ELECTRICITY_METER}
        """,
    )

    building_coverage = query(
        con,
        f"""
        WITH coverage AS (
            SELECT building_id,
                   count(*) AS n_rows,
                   sum(meter_reading) AS total_reading
            FROM {TRAIN}
            WHERE meter = {config.ELECTRICITY_METER}
            GROUP BY building_id
        )
        SELECT count(*) AS n_electricity_buildings,
               min(n_rows) AS min_rows_per_building,
               quantile_cont(n_rows, 0.50) AS median_rows_per_building,
               max(n_rows) AS max_rows_per_building,
               count(*) FILTER (WHERE total_reading = 0) AS all_zero_buildings,
               count(*) FILTER (WHERE n_rows < 8784 * 0.5) AS under_half_year_buildings
        FROM coverage
        """,
    )

    buildings_without_electricity = query(
        con,
        f"""
        SELECT count(*) AS buildings_without_meter_0
        FROM (SELECT DISTINCT building_id FROM {TRAIN}) t
        WHERE building_id NOT IN (
            SELECT DISTINCT building_id FROM {TRAIN}
            WHERE meter = {config.ELECTRICITY_METER}
        )
        """,
    )

    top_buildings = query(
        con,
        f"""
        SELECT building_id, count(*) AS n_rows, avg(meter_reading) AS mean_reading,
               max(meter_reading) AS max_reading
        FROM {TRAIN}
        WHERE meter = {config.ELECTRICITY_METER}
        GROUP BY building_id
        ORDER BY mean_reading DESC
        LIMIT 10
        """,
    )

    return {
        "structural": structural,
        "duplicates": duplicates,
        "meter_table": meter_table,
        "quantiles": quantiles,
        "electricity": electricity,
        "building_coverage": building_coverage,
        "buildings_without_electricity": buildings_without_electricity,
        "top_buildings": top_buildings,
    }


def test_audit(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    structural = query(
        con,
        f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT building_id) AS n_buildings,
               count(DISTINCT meter) AS n_meters,
               min(timestamp) AS min_timestamp,
               max(timestamp) AS max_timestamp,
               min(row_id) AS min_row_id,
               max(row_id) AS max_row_id
        FROM {TEST}
        """,
    )
    duplicates = query(
        con,
        f"""
        SELECT (SELECT count(*) FROM {TEST})
             - (SELECT count(*) FROM (
                   SELECT DISTINCT building_id, meter, timestamp FROM {TEST}
               )) AS duplicate_business_keys
        """,
    )
    meters = query(
        con,
        f"""
        SELECT meter, count(*) AS n_rows,
               count(DISTINCT building_id) AS n_buildings
        FROM {TEST}
        GROUP BY meter ORDER BY meter
        """,
    )
    overlap = query(
        con,
        f"""
        SELECT (SELECT count(DISTINCT building_id) FROM {TEST}
                WHERE building_id IN (SELECT DISTINCT building_id FROM {TRAIN})) AS test_buildings_in_train,
               (SELECT count(DISTINCT building_id) FROM {TEST}) AS test_buildings
        """,
    )
    return {
        "structural": structural,
        "duplicates": duplicates,
        "meters": meters,
        "overlap": overlap,
    }


def buildings_audit(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    structural = query(
        con,
        f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT building_id) AS n_buildings,
               count(DISTINCT site_id) AS n_sites,
               count(*) FILTER (WHERE primary_use IS NULL) AS null_primary_use,
               count(*) FILTER (WHERE square_feet IS NULL) AS null_square_feet,
               count(*) FILTER (WHERE year_built IS NULL) AS null_year_built,
               count(*) FILTER (WHERE floor_count IS NULL) AS null_floor_count,
               min(year_built) AS min_year_built,
               max(year_built) AS max_year_built,
               quantile_cont(square_feet, 0.50) AS median_square_feet,
               max(square_feet) AS max_square_feet
        FROM {BUILDINGS}
        """,
    )
    uses = query(
        con,
        f"""
        SELECT primary_use, count(*) AS n_buildings
        FROM {BUILDINGS}
        GROUP BY primary_use
        ORDER BY n_buildings DESC
        """,
    )
    sites = query(
        con,
        f"""
        SELECT site_id, count(*) AS n_buildings,
               round(avg(square_feet), 0) AS mean_square_feet,
               count(*) FILTER (WHERE year_built IS NULL) AS missing_year_built,
               count(*) FILTER (WHERE floor_count IS NULL) AS missing_floor_count
        FROM {BUILDINGS}
        GROUP BY site_id ORDER BY site_id
        """,
    )
    missing = query(
        con,
        f"""
        SELECT 'primary_use' AS column_name, round(100.0*count(*) FILTER (WHERE primary_use IS NULL)/count(*),2) AS pct_missing FROM {BUILDINGS}
        UNION ALL
        SELECT 'square_feet', round(100.0*count(*) FILTER (WHERE square_feet IS NULL)/count(*),2) FROM {BUILDINGS}
        UNION ALL
        SELECT 'year_built', round(100.0*count(*) FILTER (WHERE year_built IS NULL)/count(*),2) FROM {BUILDINGS}
        UNION ALL
        SELECT 'floor_count', round(100.0*count(*) FILTER (WHERE floor_count IS NULL)/count(*),2) FROM {BUILDINGS}
        """,
    )
    return {"structural": structural, "uses": uses, "sites": sites, "missing": missing}


def weather_audit(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, pd.DataFrame]:
    structural = query(
        con,
        f"""
        SELECT count(*) AS n_rows,
               count(DISTINCT site_id) AS n_sites,
               min(timestamp) AS min_timestamp,
               max(timestamp) AS max_timestamp,
               count(*) FILTER (WHERE air_temperature IS NULL) AS null_air_temperature,
               count(*) FILTER (WHERE dew_temperature IS NULL) AS null_dew_temperature,
               count(*) FILTER (WHERE wind_speed IS NULL) AS null_wind_speed,
               count(*) FILTER (WHERE cloud_coverage IS NULL) AS null_cloud_coverage,
               count(*) FILTER (WHERE precip_depth_1_hr IS NULL) AS null_precip_depth,
               count(*) FILTER (WHERE sea_level_pressure IS NULL) AS null_sea_level_pressure
        FROM {table}
        """,
    )
    sites = query(
        con,
        f"""
        SELECT site_id, count(*) AS n_rows,
               min(timestamp) AS first_timestamp,
               max(timestamp) AS last_timestamp,
               round(100.0*count(*) FILTER (WHERE air_temperature IS NULL)/count(*),2) AS pct_missing_air_temp,
               round(avg(air_temperature), 1) AS mean_air_temp,
               min(air_temperature) AS min_air_temp,
               max(air_temperature) AS max_air_temp
        FROM {table}
        GROUP BY site_id ORDER BY site_id
        """,
    )
    missing = query(
        con,
        f"""
        SELECT 'air_temperature' AS column_name,
               round(100.0*count(*) FILTER (WHERE air_temperature IS NULL)/count(*), 2) AS pct_missing
        FROM {table}
        UNION ALL SELECT 'dew_temperature', round(100.0*count(*) FILTER (WHERE dew_temperature IS NULL)/count(*), 2) FROM {table}
        UNION ALL SELECT 'wind_speed', round(100.0*count(*) FILTER (WHERE wind_speed IS NULL)/count(*), 2) FROM {table}
        UNION ALL SELECT 'cloud_coverage', round(100.0*count(*) FILTER (WHERE cloud_coverage IS NULL)/count(*), 2) FROM {table}
        UNION ALL SELECT 'precip_depth_1_hr', round(100.0*count(*) FILTER (WHERE precip_depth_1_hr IS NULL)/count(*), 2) FROM {table}
        UNION ALL SELECT 'sea_level_pressure', round(100.0*count(*) FILTER (WHERE sea_level_pressure IS NULL)/count(*), 2) FROM {table}
        ORDER BY column_name
        """,
    )
    duplicates = query(
        con,
        f"""
        SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT site_id, timestamp FROM {table}))
               AS duplicate_site_timestamps
        FROM {table}
        """,
    )
    return {"structural": structural, "sites": sites, "missing": missing, "duplicates": duplicates}


def merged_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return query(
        con,
        f"""
        WITH elec AS (
            SELECT t.building_id, t.timestamp, t.meter_reading, b.site_id
            FROM {TRAIN} t
            LEFT JOIN {BUILDINGS} b USING (building_id)
            WHERE t.meter = {config.ELECTRICITY_METER}
        )
        SELECT count(*) AS n_electricity_rows,
               count(*) FILTER (WHERE site_id IS NULL) AS rows_missing_site,
               count(*) FILTER (WHERE w.air_temperature IS NULL) AS rows_missing_air_temperature,
               count(*) FILTER (WHERE w.dew_temperature IS NULL) AS rows_missing_dew_temperature,
               count(*) FILTER (WHERE w.wind_speed IS NULL) AS rows_missing_wind_speed,
               count(*) FILTER (WHERE w.cloud_coverage IS NULL) AS rows_missing_cloud_coverage,
               count(*) FILTER (WHERE w.precip_depth_1_hr IS NULL) AS rows_missing_precip,
               count(*) FILTER (WHERE w.sea_level_pressure IS NULL) AS rows_missing_pressure,
               count(*) FILTER (WHERE w.site_id IS NULL) AS rows_without_weather_key
        FROM elec
        LEFT JOIN {WEATHER_TRAIN} w USING (site_id, timestamp)
        """,
    )


def sample_submission_audit(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    path = config.SAMPLE_SUBMISSION_CSV
    if not path.exists():
        return pd.DataFrame()
    return query(
        con,
        f"""
        SELECT count(*) AS n_rows, count(DISTINCT row_id) AS n_distinct_row_ids,
               min(row_id) AS min_row_id, max(row_id) AS max_row_id,
               count(*) FILTER (WHERE meter_reading IS NULL) AS null_predictions
        FROM read_csv_auto('{path}', header=true)
        """,
    )


def main() -> None:
    config.ensure_directories()
    con = duckdb.connect()
    train = train_audit(con)
    test = test_audit(con)
    buildings = buildings_audit(con)
    weather_train = weather_audit(con, WEATHER_TRAIN)
    weather_test = weather_audit(con, WEATHER_TEST)
    merged = merged_coverage(con)
    submission = sample_submission_audit(con)
    sizes = file_sizes()
    con.close()

    summary = {
        "train_rows": int(train["structural"].loc[0, "n_rows"]),
        "train_buildings": int(train["structural"].loc[0, "n_buildings"]),
        "train_min_timestamp": str(train["structural"].loc[0, "min_timestamp"]),
        "train_max_timestamp": str(train["structural"].loc[0, "max_timestamp"]),
        "test_rows": int(test["structural"].loc[0, "n_rows"]),
        "test_min_timestamp": str(test["structural"].loc[0, "min_timestamp"]),
        "test_max_timestamp": str(test["structural"].loc[0, "max_timestamp"]),
        "electricity_rows": int(train["electricity"].loc[0, "n_rows"]),
        "electricity_buildings": int(train["electricity"].loc[0, "n_buildings"]),
        "electricity_zero_pct": float(train["electricity"].loc[0, "pct_zeros"]),
        "duplicate_business_keys": int(train["duplicates"].loc[0, "duplicate_business_keys"]),
    }
    (config.REPORTS_DIR / "data_inventory.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    sections = [
        ("1. Files and sizes", sizes),
        ("2. Training set: structure", train["structural"]),
        ("3. Training set: duplicates", train["duplicates"]),
        ("4. Training set: readings by meter", train["meter_table"]),
        ("5. Training set: target quantiles", train["quantiles"]),
        ("6. Electricity (meter 0) target", train["electricity"]),
        ("7. Electricity coverage per building", train["building_coverage"]),
        ("8. Buildings without electricity readings", train["buildings_without_electricity"]),
        ("9. Highest mean electricity buildings", train["top_buildings"]),
        ("10. Test set: structure", test["structural"]),
        ("11. Test set: duplicates and overlap", test["duplicates"]),
        ("12. Test set: readings by meter", test["meters"]),
        ("13. Test/train building overlap", test["overlap"]),
        ("14. Building metadata: structure", buildings["structural"]),
        ("15. Building metadata: missingness", buildings["missing"]),
        ("16. Building metadata: primary use", buildings["uses"]),
        ("17. Building metadata: sites", buildings["sites"]),
        ("18. Weather (train): structure", weather_train["structural"]),
        ("19. Weather (train): missingness", weather_train["missing"]),
        ("20. Weather (train): per site", weather_train["sites"]),
        ("21. Weather (train): duplicates", weather_train["duplicates"]),
        ("22. Weather (test): structure", weather_test["structural"]),
        ("23. Weather (test): missingness", weather_test["missing"]),
        ("24. Weather (test): per site", weather_test["sites"]),
        ("25. Weather (test): duplicates", weather_test["duplicates"]),
        ("26. Electricity rows merged with weather", merged),
        ("27. Sample submission", submission),
    ]

    lines = [
        "# ASHRAE Great Energy Predictor III — Data Inventory",
        "",
        "Generated by `src/inspect_data.py` from the raw files in `data/raw/`. "
        "All row counts were computed with DuckDB directly from the CSV/Parquet files.",
        "",
        "## Key findings",
        "",
        f"- Training observations: **{summary['train_rows']:,}** across "
        f"**{summary['train_buildings']}** buildings.",
        f"- Training period: **{summary['train_min_timestamp']}** to "
        f"**{summary['train_max_timestamp']}** (one calendar year, leap year 2016).",
        f"- Test features period: **{summary['test_min_timestamp']}** to "
        f"**{summary['test_max_timestamp']}**; test labels are not provided.",
        f"- Electricity (meter 0): **{summary['electricity_rows']:,}** rows across "
        f"**{summary['electricity_buildings']}** buildings, "
        f"**{summary['electricity_zero_pct']:.2f}%** zero readings.",
        f"- Duplicate (building, meter, timestamp) keys in train: "
        f"**{summary['duplicate_business_keys']:,}**.",
        "- Meter coding observed: 0 = electricity, 1 = chilled water, 2 = steam, "
        "3 = hot water, consistent with the competition documentation.",
        "",
        "## Interpretation notes",
        "",
        "- `meter_reading` units are not explicitly stated in the downloadable CSV. "
        "The competition documentation defines the target as energy consumed in kWh; "
        "the re-uploaded file does not add a unit column. Units are therefore reported "
        "as **dataset energy units (documented as kWh)** and are not independently "
        "re-verified from the file.",
        "- Repeated timestamps across different buildings are expected and are not "
        "duplicates. The duplicate check uses the business key.",
        "- Zero readings are retained in the audit; their treatment is decided in the "
        "cleaning stage and documented in the report.",
        "",
    ]
    for title, frame in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(markdown_table(frame))
        lines.append("")

    (config.REPORTS_DIR / "data_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {config.REPORTS_DIR / 'data_inventory.md'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
