# Dashboard data dictionary

All files are in `data/dashboard/tableau_data/` and are also read directly by the
Streamlit app. The data cover the documented local analysis subset: 120
buildings, 16 sites, 1,030,967 electricity observations from 2016.

## `electricity_hourly.csv` (1,030,967 rows)

Building-hour electricity observations with weather and calendar attributes.

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Observation hour (local site time as supplied) |
| `building_id` | integer | Building identifier |
| `site_id` | integer | Site identifier |
| `primary_use` | text | Building primary-use category |
| `square_feet` | float | Building floor area (square feet) |
| `year_built` | integer | Construction year; null for 51.8% of rows |
| `floor_count` | integer | Floors; null for 70.9% of rows |
| `meter_reading` | float | Electricity meter reading (dataset energy units; documented as kWh) |
| `air_temperature` | float | Outdoor air temperature (°C) |
| `dew_temperature` | float | Dew-point temperature (°C) |
| `wind_speed` | float | Wind speed (dataset units, documented as m/s) |
| `cloud_coverage` | integer | Cloud cover (oktas, 0–9); missing values retained |
| `precip_depth_1_hr` | float | Hourly precipitation depth (mm); negative sentinel removed |
| `sea_level_pressure` | float | Sea-level pressure (hPa) |
| `hour` | integer | Hour of day (0–23) |
| `dayofweek` | integer | Day of week (0 = Sunday, matches source convention) |
| `month` | integer | Month (1–12) |
| `is_weekend` | integer | 1 if Saturday/Sunday, else 0 |
| `split` | text | `train` (Jan–Sep), `validation` (Oct–Nov), `test` (Dec) |

## `building_summary.csv` (120 rows)

One row per building: mean, median, and total reading; observation count; metadata.

## `monthly_site_summary.csv` (192 rows)

Month × site aggregates: total and mean reading, observation count, mean air
temperature, and distinct building count.

## `monthly_use_summary.csv` (192 rows)

Month × primary-use aggregates: total and mean reading, observation count, and
distinct building count.

## `model_comparison.csv` (8 rows)

Validation and test metrics for the mean baseline, Linear Regression, Random
Forest, and HistGradientBoosting. Columns: `model`, `split`, `n_rows`, `mae`,
`rmse`, `r2`, `rmsle`, `fit_seconds`, `predict_seconds`, `fit_rows`,
`actual_mean`, `predicted_mean`.

## `model_test_predictions.csv` (349,232 rows)

Long-format December 2016 predictions for every model. Columns: `building_id`,
`timestamp`, `site_id`, `primary_use`, `model`, `actual`, `predicted`,
`residual`, `absolute_error`.

## `error_by_primary_use.csv`, `error_by_site.csv`, `error_by_reading_band.csv`

Segment-level MAE, RMSE, R², RMSLE, and row counts for the selected model on the
test period.

## `feature_importance.csv`

Permutation importance (increase in MAE when a feature is shuffled) for the
selected model, measured on a 10,000-row test sample.

## `unseen_case_predictions.csv` (2 rows)

Two held-out December 2016 cases with actual reading, prediction, absolute error,
and percentage error.

## `generalization_building_holdout.csv` (3 rows)

Secondary experiment in which 20% of buildings were excluded from training:
errors on unseen buildings, seen buildings, and seen-building validation data.

## Unit caution

The downloadable CSV does not contain a unit column. The competition
documentation defines meter readings as energy in kWh. Dashboard labels therefore
say **dataset units (documented as kWh)** and distinguish the **sum of readings**
from instantaneous power or an efficiency metric.
