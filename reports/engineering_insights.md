# Engineering Insights

**Scope:** Site-stratified local subset of 120 buildings, 16 sites, 1,030,967 hourly electricity readings from 2016. These are subset findings, not full-dataset findings. The target units are dataset energy units (documented as kWh).

Each insight below is tied to a computed result, an engineering interpretation, an actionable implication, and a limitation.

---

## Insight 1 — Electricity demand is strongly right-skewed and building-dominated

**Evidence**

- Median hourly reading: **59.05** dataset units; mean: **188.78**; 99th percentile: **2454**; maximum: **41,552**.
- The mean is more than three times the median, so a small number of high-load observations dominate the total.
- Building floor area has a log–log correlation of **0.750** with building mean reading across 120 buildings.
- Permutation importance ranks `square_feet` first (**178.43** increase in MAE when permuted), ahead of `primary_use` (83.33) and `building_id` (53.68).

**Engineering interpretation**

Building scale and use set the baseline demand far more strongly than hour-to-hour weather variation. A single global average is not a meaningful benchmark for an individual building; large laboratories, hospitals, and education buildings can operate at readings orders of magnitude above small warehouses or parking structures.

**Actionable implication**

Normalize energy comparisons by floor area and primary use before flagging a building as inefficient. Use building-specific baselines or size-adjusted benchmarks, and preserve legitimate high-load peaks rather than trimming them as outliers.

**Limitation**

Floor area is missing for no buildings in the full metadata but varies in quality; the subset has a median of 45,264.5 ft². Correlation is not causation, and floor area does not capture occupancy, equipment intensity, or operating hours.

Sources: `reports/tables/eda_summary.json`; `reports/figures/reading_distribution.png`; `reports/figures/building_size_relationship.png`; `reports/tables/feature_importance_histgradientboosting.csv`.

---

## Insight 2 — Occupancy schedules create a clear weekday and midday demand cycle

**Evidence**

- Mean weekday hourly reading: **195.44**; weekend: **172.24**; weekday/weekend ratio: **1.135**.
- Peak weekday hour: **13:00**; trough: **03:00**.
- August has the highest monthly mean (**215.14**); April the lowest (**166.60**).
- The daily mean across all buildings ranges from **146.87** (15 May) to **249.89** (29 August).

**Engineering interpretation**

The midday peak and weekend reduction are consistent with occupancy-driven ventilation, lighting, and cooling schedules. The summer maximum is consistent with cooling load, while the winter months remain relatively high because heating and baseline loads persist.

**Actionable implication**

Target HVAC scheduling and setback measures at the 03:00–06:00 low-occupancy window and verify that weekend operation is genuinely required. Use hour-of-day and day-of-week features in any operational forecasting model; a daily average hides the peak that drives demand charges.

**Limitation**

The dataset does not contain occupancy, holiday, or operating-schedule labels. The December test period is winter-only, and the weekday/weekend pattern is an aggregate across 16 climates and 16 building uses.

Sources: `reports/tables/eda_summary.json`; `reports/figures/temporal_patterns.png`.

---

## Insight 3 — Temperature response is site-specific; the pooled correlation hides heating and cooling regimes

**Evidence**

- Pooled air-temperature/reading Pearson correlation: **-0.074**, which looks weak and negative.
- Site-level correlations range from **-0.134** (site 10) to **+0.406** (site 11).
- Site 11 mean reading rises from **265.74** below 10 °C to **377.17** above 25 °C.
- Sites 6, 7, and 10 show negative correlations; site 0 is positive (**+0.234**) and site 13 is weakly positive (**+0.056**) with much higher loads at warmer temperatures.
- Global means are **279.04** below 5 °C, **187.59** at 15–25 °C, and **143.01** above 30 °C.

**Engineering interpretation**

Different sites are dominated by different load components. In heating-dominated climates, colder weather increases demand; in cooling-dominated or internally load-dominated buildings, warmer weather increases demand. Pooling sites cancels these effects and produces a misleadingly weak global coefficient.

**Actionable implication**

Do not use a single linear temperature coefficient for all buildings. Add site-specific temperature terms, heating/cooling degree hours, or interaction features between site and temperature. Interpret weather sensitivity only within a defined climate and building-use segment.

**Limitation**

Weather is measured at site level, not at each building. Site 11's positive correlation may also reflect its building mix and high-load facilities rather than climate alone. The analysis is associational.

Sources: `reports/tables/eda_summary.json`; `reports/figures/temperature_relationship.png`; `reports/figures/temperature_response_by_site.png`.

---

## Insight 4 — Primary use and site produce larger differences than most weather variables

**Evidence**

- Median reading by primary use ranges from **3.75** (Religious worship) to **157.20** (Education) — a factor of about 42.
- Education also has the highest mean (**485.24**), followed by Entertainment/public assembly (**204.26**).
- Site mean readings range from **10.24** (site 5) to **1156.11** (site 7).
- `primary_use` is the second-most important permutation feature; `site_id` is fourth.

**Engineering interpretation**

Building function and campus/site context determine equipment density, schedules, and plug loads. A hospital, a school, and a parking structure are not interchangeable observations, even at the same outdoor temperature.

**Actionable implication**

Segment dashboards and models by primary use and site; report benchmarks within peer groups. For portfolio-level planning, allocate monitoring and retrofit effort to high-load segments first, but verify with sub-metering before claiming savings.

**Limitation**

The subset contains only 2–24 buildings in several primary-use categories. Site 7 has only three buildings and a very high mean, so its ranking is sensitive to those buildings. The full dataset has different category proportions.

Sources: `reports/tables/eda_summary.json`; `reports/figures/primary_use_consumption.png`; `reports/figures/site_comparison.png`.

---

## Insight 5 — Strong seen-building accuracy does not transfer to unseen buildings

**Evidence**

- Selected HistGradientBoosting test performance for seen buildings: MAE **33.20**, RMSE **69.78**, R² **0.970**, RMSLE **0.591**.
- In a 20% building-holdout experiment, unseen-building test performance: MAE **150.51**, RMSE **321.33**, R² **-0.555**, RMSLE **1.525**.
- Seen-building test performance in the same experiment: MAE **33.86**, R² **0.971**.
- `building_id` is the third-most important feature, which helps explain the gap.

**Engineering interpretation**

The main model has learned building-specific levels and schedules. It interpolates well for buildings it has seen, but it does not generalize to new buildings from metadata and weather alone. This is a critical distinction for a deployment claim: the model is a monitoring/interpolation tool, not a universal pre-design predictor.

**Actionable implication**

For a new building, collect a short history and recalibrate, or add richer building descriptors (HVAC type, occupancy, schedules, equipment intensity). Report seen-building and unseen-building metrics separately. Do not use the 0.97 R² as evidence that the model will work on an unseen portfolio.

**Limitation**

The holdout removes entire buildings but still shares the same 2016 period, sites, and weather. Performance on a future year, a new climate, or a changed building would likely be worse. The December test period also has only 126 zero readings, versus 44,796 in training, so the seen-building test metrics may be optimistic for periods with missing/zero data.

Sources: `reports/tables/model_comparison.csv`; `reports/tables/generalization_building_holdout.csv`; `reports/figures/unseen_buildings_actual_vs_predicted.png`; `models/model_metadata.json`.
