# Interactive Dashboard Specification

**Status:** implementation specification and build guide. No Tableau workbook has
been authored or published in this environment. Tableau Cloud/Public authoring is
performed by the student using the files in `data/dashboard/tableau_data/`.

## Data sources

| Source | Grain | Rows | Use |
|---|---|---|---|
| `electricity_hourly.csv` | building-hour | 1,030,967 | Energy overview, weather/building analysis |
| `building_summary.csv` | building | 120 | Building comparisons, size analysis |
| `monthly_site_summary.csv` | month × site | 192 | Site trends, KPI cards |
| `monthly_use_summary.csv` | month × use | 192 | Category trends |
| `model_comparison.csv` | model × split | 8 | Model performance table |
| `model_test_predictions.csv` | model × building-hour | 349,232 | Actual/predicted and residuals |
| `error_by_primary_use.csv` | primary use | 16 | Error analysis by category |
| `error_by_site.csv` | site | 16 | Error analysis by site |
| `error_by_reading_band.csv` | reading band | 8 | Error analysis by demand level |
| `feature_importance.csv` | feature | 24 | Model interpretation |
| `unseen_case_predictions.csv` | case | 2 | Unseen-case demonstration |
| `generalization_building_holdout.csv` | building group | 3 | Generalization caution |

**Scope label:** site-stratified subset, 120 buildings, 16 sites, 2016. Do not
present these figures as full-ASRAE estimates.

## Global design

- Background `#F7F8FA`; panels white; primary blue `#4C72B0`; accent green
  `#55A868`; warning orange `#DD8452`; error red `#C44E52`.
- Font: Tableau Public's Arial/Tableau default; 10–12 pt body, 22–26 pt KPI
  values, 12–14 pt titles.
- Number formats: readings with thousands separators and 1 decimal; metrics
  2 decimals; R² 3 decimals; percentages 1 decimal.
- Every dashboard has a title block, a one-line scope note, and a filter panel.
- Always label the measure as **dataset units (documented as kWh)**.
- Never call the sum of readings "power", "efficiency", or "savings".

## Dashboard 1 — Energy Overview

### KPIs (top row)

| KPI | Calculation | Notes |
|---|---|---|
| Total reading sum | `SUM([meter_reading])` | Energy-like total over selected observations, not power |
| Mean reading per observation | `AVG([meter_reading])` | Main intensity KPI |
| Buildings | `COUNTD([building_id])` | Changes with filters |
| Sites | `COUNTD([site_id])` | Changes with filters |
| Observations | `COUNT([meter_reading])` | Makes coverage visible |
| Selected period | `MIN([timestamp])` – `MAX([timestamp])` | Text |

### Charts

1. **Monthly mean reading** — line chart, `MONTH([timestamp])` on columns,
   `AVG([meter_reading])` on rows. Add a reference line at the overall mean.
2. **Daily trend** — line chart, `DATETRUNC('day',[timestamp])` with
   `AVG([meter_reading])`; add a 7-day moving average (table calculation).
3. **Reading by hour** — line chart, hour on columns, mean reading on rows,
   separate lines for weekday/weekend using `is_weekend`.
4. **Site ranking** — bar chart, mean reading by site, sorted descending.

### Filters

Date range, site, primary use, building. Apply to all sheets using the data
source. Show "All" values by default.

### Engineering annotation

"The sum of hourly readings is an energy-like total over the selected
observations. It is not instantaneous power, and it is not an efficiency or
savings estimate. Differences between sites also reflect building mix, size, and
climate."

## Dashboard 2 — Weather and Building Analysis

### Charts

1. **Temperature response** — scatter or binned line: air temperature on
   columns, mean reading on rows; colour by site. Add a note that sites show
   opposite-signed relationships (site-level Pearson r ranges from about −0.13 to
   +0.41 in this subset).
2. **Hour × month heatmap** — hour on rows, month on columns, mean reading as
   colour. Use a sequential blue palette and show the colour legend.
3. **Primary-use comparison** — horizontal bar of median reading by primary use,
   with observation count in the tooltip.
4. **Building size** — scatter of `square_feet` (log axis) versus building mean
   reading (log axis), colour by primary use, using `building_summary.csv`.
5. **Site climate context** — dual-axis bar/line: mean reading and mean air
   temperature by site; explain that this is descriptive, not causal.

### Filters

Date range, site, primary use, building, weekend/weekday. Temperature and size
charts use the filtered `electricity_hourly` source.

### Interaction

Selecting a site in any chart highlights that site in the others (highlight
action). Selecting a primary-use bar filters the temperature chart.

## Dashboard 3 — Model Performance

### KPIs

Selected model (`HistGradientBoosting`), test MAE, test RMSE, test R², test
RMSLE, test observation count. Validation metrics are shown in the comparison
table, not as the headline KPI.

### Charts

1. **Model comparison table** — `model`, `split`, `mae`, `rmse`, `r2`, `rmsle`,
   `fit_seconds`. Highlight the selected model.
2. **Metric bars** — MAE, RMSE, and R² by model for the test split.
3. **Actual vs predicted** — scatter of `actual` versus `predicted` from
   `model_test_predictions.csv` for the selected model; add the 45° reference
   line. Use a log scale option to inspect low readings.
4. **Residual distribution** — histogram of `residual`; add a zero reference
   line. Mention that positive residuals are underprediction.
5. **Error by primary use / site / reading band** — bar charts of MAE, with
   observation counts in tooltips.
6. **Feature importance** — horizontal bar of `importance_mean` from
   `feature_importance.csv`; caption that permutation importance is not causal.
7. **Generalization panel** — table from
   `generalization_building_holdout.csv`: seen buildings vs unseen buildings.
   State plainly that the model degrades on unseen buildings.
8. **Unseen cases** — table from `unseen_case_predictions.csv`.

### Filters and parameters

Model selector parameter limited to the four rows in `model_comparison.csv`;
test/validation split selector; primary use and site filters for error charts.
The actual-vs-predicted chart must use the same model selected in the parameter.

## Interaction and accessibility checklist

- All filters show current values and an obvious "reset" action.
- Colour is never the only encoding: label categories and annotate key points.
- Tooltips state units and row counts.
- Text contrast meets WCAG AA against the light background.
- Test every filter combination for empty states; show "No observations match
  the current filters" rather than a blank chart.
- Do not use dual axes for unrelated quantities without labeling both axes.
