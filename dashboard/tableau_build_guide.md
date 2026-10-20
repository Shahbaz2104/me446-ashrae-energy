# Tableau Cloud/Public Build Guide

This guide describes how to build the three dashboards from the prepared CSV
files. No `.twb`/`.twbx` workbook is included, and no workbook has been published
from this environment. Follow the steps in Tableau Public Desktop (or Tableau
Desktop connected to Tableau Cloud) and publish under your own account.

## 1. Prepare Tableau

1. Install Tableau Public Desktop (free) or open Tableau Cloud in a browser.
2. Create a project/folder named `ME446 ASHRAE Energy CEP`.
3. Keep the CSV files in `data/dashboard/tableau_data/`. If using Tableau Cloud,
   publish the data sources or upload the CSVs to a cloud file location that
   Tableau can reach.
4. Set the locale/number format to English (United Kingdom or United States) so
   that thousands separators and decimals display consistently.

## 2. Connect the data

For each CSV: **Connect → Text file → select file**.

Create these connections:

| Connection name | File | First row is header |
|---|---|---|
| Electricity Hourly | `electricity_hourly.csv` | Yes |
| Building Summary | `building_summary.csv` | Yes |
| Monthly Site | `monthly_site_summary.csv` | Yes |
| Monthly Use | `monthly_use_summary.csv` | Yes |
| Model Comparison | `model_comparison.csv` | Yes |
| Model Predictions | `model_test_predictions.csv` | Yes |
| Error by Use | `error_by_primary_use.csv` | Yes |
| Error by Site | `error_by_site.csv` | Yes |
| Error by Band | `error_by_reading_band.csv` | Yes |
| Feature Importance | `feature_importance.csv` | Yes |
| Unseen Cases | `unseen_case_predictions.csv` | Yes |
| Generalization | `generalization_building_holdout.csv` | Yes |

In Tableau, verify:

- `timestamp` is recognized as Date & Time.
- `meter_reading`, `actual`, `predicted`, `residual`, `absolute_error`, `mae`,
  `rmse`, `r2`, `rmsle` are numeric measures.
- `building_id`, `site_id`, `primary_use`, `model`, `split` are dimensions.
- `year_built` and `floor_count` may contain nulls; keep them as dimensions.

## 3. Create shared calculated fields (Electricity Hourly)

| Field name | Calculation |
|---|---|
| `Reading (dataset units)` | `[meter_reading]` |
| `Mean Reading` | `AVG([meter_reading])` |
| `Total Reading` | `SUM([meter_reading])` |
| `Observations` | `COUNT([meter_reading])` |
| `Buildings` | `COUNTD([building_id])` |
| `Sites` | `COUNTD([site_id])` |
| `Month Start` | `DATETRUNC('month', [timestamp])` |
| `Day` | `DATETRUNC('day', [timestamp])` |
| `Hour` | `DATEPART('hour', [timestamp])` |
| `Day of Week` | `DATEPART('weekday', [timestamp])` |
| `Weekend` | `IF [is_weekend] = 1 THEN 'Weekend' ELSE 'Weekday' END` |
| `Reading Band` | `IF [meter_reading] < 50 THEN '0–50' ELSEIF [meter_reading] < 250 THEN '50–250' ELSEIF [meter_reading] < 1000 THEN '250–1000' ELSE '1000+' END` |

Create a parameter `Selected Model` with values: `Mean baseline`,
`Linear Regression`, `Random Forest`, `HistGradientBoosting`. For the model
dashboard, add a calculated field
`IF [model] = [Selected Model] THEN 'Selected' ELSE 'Other' END` and use it as a
filter or colour.

## 4. Dashboard 1 — Energy Overview

### KPI sheets

Create four separate worksheets with `Measure Names`/`Measure Values` or a single
text table:

1. `KPI – Total Reading`: `SUM([meter_reading])` formatted with thousands
   separators; label **Total reading sum (dataset units)**.
2. `KPI – Mean Reading`: `AVG([meter_reading])`; label **Mean per observation**.
3. `KPI – Buildings`: `COUNTD([building_id])`.
4. `KPI – Sites`: `COUNTD([site_id])`.
5. `KPI – Period`: `MIN([timestamp])` and `MAX([timestamp])` in a text table.

### Trend sheets

1. **Monthly Mean Reading**: Columns `Month Start` (continuous), Rows
   `AVG([meter_reading])`; mark type Line; add a reference line at the average.
2. **Daily Trend**: Columns `Day` (continuous), Rows `AVG([meter_reading])`;
   add a 7-day moving average table calculation:
   `WINDOW_AVG(AVG([meter_reading]), -6, 0)`.
3. **Hourly Profile**: Columns `Hour`, Rows `AVG([meter_reading])`, Colour
   `Weekend`; mark type Line.
4. **Site Ranking**: Rows `site_id`, Columns `AVG([meter_reading])`; sort
   descending; add `COUNT([meter_reading])` to Tooltip.

### Assemble

- Drag the KPI sheets into the top band.
- Place the monthly trend and daily trend side by side.
- Place the hourly profile and site ranking side by side below.
- Add a title, scope note, and filter panel (date range, site, primary use,
  building). Apply filters to all worksheets using this data source.
- Add the unit/interpretation annotation from the specification.

## 5. Dashboard 2 — Weather and Building Analysis

1. **Temperature Response**: use Electricity Hourly. Columns `air_temperature`
   (continuous, binned to 2°C), Rows `AVG([meter_reading])`, Colour `site_id`;
   mark type Line or Circle. Add a second view with `air_temperature` on
   Columns and `AVG([meter_reading])` as a scatter, detail `building_id`.
2. **Hour × Month Heatmap**: Rows `Hour`, Columns `month`, Colour
   `AVG([meter_reading])`, mark type Square; use a sequential blue palette and
   show the colour legend.
3. **Primary Use**: Rows `primary_use`, Columns `MEDIAN([meter_reading])`, mark
   type Bar; add `COUNT([meter_reading])` to Tooltip.
4. **Building Size**: connect Building Summary. Columns `square_feet` (log
   axis), Rows `mean_reading` (log axis), Detail `building_id`, Colour
   `primary_use`, mark type Circle.
5. **Site Climate Context**: use Monthly Site. Columns `site_id`, Rows
   `AVG([mean_air_temperature])` and `AVG([mean_reading])` on a dual axis;
   synchronize only if the scales are clearly labelled. Prefer two aligned
   panels to avoid misleading dual-axis comparisons.

Add filters for date, site, primary use, building, and weekday/weekend. Add a
highlight action from site marks to the temperature chart.

## 6. Dashboard 3 — Model Performance

1. **Comparison Table**: Model Comparison. Rows `model`; Columns `split`,
   `mae`, `rmse`, `r2`, `rmsle`, `fit_seconds`; format metrics to 2–3 decimals.
   Highlight `HistGradientBoosting`.
2. **Metric Bars**: filter `split = test`; Columns `mae`, `rmse`, `r2`; Rows
   `model`; use three small bar charts or a measure-values bar.
3. **Actual vs Predicted**: Model Predictions, filter
   `model = [Selected Model]` and `split` is not present in this file (it is the
   test period only). Columns `actual`, Rows `predicted`, Detail `building_id`;
   add a 45° reference line using a calculated field `[actual]`.
4. **Residual Histogram**: Columns `residual` (binned), Rows `COUNT([residual])`.
   Add a zero reference line; annotate positive = underprediction.
5. **Error by Use / Site / Band**: use the three error CSVs. Rows the grouping
   field, Columns `mae`; add `n_rows` to Tooltip.
6. **Feature Importance**: Rows `feature`, Columns `importance_mean`; sort
   descending; add error bars using `importance_std` if desired.
7. **Generalization Panel**: Rows `building_group`, Columns `mae`, `rmse`, `r2`;
   add a text annotation: "The model transfers poorly to buildings absent from
   training."
8. **Unseen Cases**: text table from `unseen_case_predictions.csv` with
   `case_id`, `building_id`, `timestamp`, `actual_reading`,
   `predicted_reading`, `absolute_error`, `percentage_error`.

Use a parameter action or filter so the model selector controls the
actual-vs-predicted and residual views.

## 7. Layout and formatting

- Dashboard size: 1200 × 800 (desktop) and a second mobile layout if time allows.
- Use white containers with 16 px padding and 12 px gaps.
- Place KPIs in a single row; charts in a 2 × 2 grid; keep one message per chart.
- Titles: 16–20 pt bold; subtitles: 10–11 pt grey; KPI values: 24–28 pt.
- Set all reading axes to **dataset units (documented as kWh)**.
- Add a footer: data scope, prepared date, and the sentence "Association is not
  causation; this dashboard is not an efficiency or savings audit."

## 8. Publish and validate

1. Save the workbook as `ME446_ASHRAE_Energy_Dashboard.twbx`.
2. Publish to Tableau Public/Cloud and copy the public URL.
3. Test in a private browser window:
   - filters update every sheet;
   - the model selector updates the performance views;
   - no sheet shows stale tooltips or wrong units;
   - the dashboard loads with the default "All" filters;
   - the published URL is accessible without login.
4. Record the URL and screenshots in the report only after the publish step
   actually succeeds.

## 9. What is intentionally not claimed

- No workbook was created in this repository.
- No dashboard was published by the project code.
- The CSV extracts are a documented subset of the full 2016 electricity data.
