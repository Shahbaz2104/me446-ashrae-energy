# AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications

**NED University of Engineering and Technology**  
**Department of Mechanical Engineering**  
**ME-446 — AI & Internet of Things, Fall Semester 2026**  
**Complex Engineering Problem (CLO-2, PLO-2)**

| Item | Detail |
|---|---|
| Project title | AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications |
| Group number | _[To be supplied]_ |
| Students | _[Name — Roll No.]_ |
| Instructor | _[Instructor Name]_ |
| Submission date | 4 November 2026 |

> **Status and honesty note.** The dataset approval proposal is **awaiting teacher approval**; uniqueness has not been confirmed. The numerical results in this report are from a documented **120-building local subset**, not the full 1,413 electricity buildings. A full-data Colab workflow is prepared but has not been executed, so no full-data metrics are claimed. No Tableau workbook has been authored, published, or tested; the project supplies Tableau-ready data and a detailed build guide. These limitations are repeated where relevant.

---

## Abstract

This project develops and evaluates data-driven models for hourly building electricity demand using the ASHRAE Great Energy Predictor III dataset. The engineering problem is to predict the electricity meter reading of a building from weather, calendar, building metadata, and building identity, and to determine whether the resulting model is accurate enough for monitoring and demand-planning support.

The analysis uses a site-stratified subset of 120 buildings, 16 sites, and 1,030,967 hourly observations from 2016. After a structural audit, weather cleaning, feature engineering, and a leakage-safe chronological split (January–September for training, October–November for validation, December for testing), three regressors are compared: Linear Regression, Random Forest, and HistGradientBoosting, with a mean baseline for reference.

HistGradientBoosting is selected by the pre-declared validation-RMSE rule. On the December test period it achieves MAE 33.20, RMSE 69.78, R² 0.970, and RMSLE 0.591 in dataset units. Random Forest has a slightly lower test MAE (32.05) and better RMSLE (0.448) but a higher test RMSE (78.90). The strong seen-building accuracy does not transfer to new buildings: in a 20% building-holdout experiment, the model's unseen-building R² is **-0.555** and MAE is 150.51. The main operational conclusion is therefore conditional: the model is useful for buildings with historical data, but it is not a validated predictor for unseen buildings or future operating conditions without recalibration.

The project also delivers a data-quality audit, exploratory analysis, five evidence-based engineering insights, error analysis, two unseen-row demonstrations, Tableau-ready extracts, a Tableau build guide, a smoke-tested Streamlit application, tests, a reproducible notebook, and this report.

---

# Chapter 1: Introduction

## 1.1 Background

Buildings account for a large share of electricity use in urban energy systems. A significant portion is associated with heating, ventilation, and air-conditioning (HVAC), lighting, plug loads, and building services. Mechanical engineers are increasingly asked to use operational data to understand demand, schedule equipment, detect abnormal consumption, and evaluate efficiency measures.

Hourly meter data are noisy. Demand depends on weather, building size and use, occupancy schedules, equipment, and unobserved operational decisions. A model that predicts normal demand can provide a reference for monitoring: a large positive residual may indicate an unusual event, a control fault, or a data-quality problem. The same model can support short-term demand planning and help prioritize buildings for investigation.

## 1.2 Engineering motivation

The motivating problem is not simply to maximize a competition score. It is to determine whether a regression model learned from readily available data can support a mechanical-engineering decision. Three questions matter:

1. Is the model accurate for the buildings it has seen?
2. Does it transfer to buildings it has not seen?
3. Are its errors and feature effects physically plausible enough to support an operational recommendation?

A model with a high aggregate R² but poor unseen-building behavior can mislead a facilities team. This report therefore treats generalization and error structure as first-class results, not as an afterthought.

## 1.3 Problem statement

Given a building identifier, a timestamp, building metadata, and site-level weather observations, predict the hourly **electricity** meter reading. The target is the `meter_reading` field for `meter == 0`. The model is trained on 2016 data and evaluated on a later 2016 period using a chronological split. The 2017–2018 test features provided by Kaggle have no labels and are therefore not used for supervised evaluation.

## 1.4 Objectives

1. Audit the dataset for structure, duplicates, missingness, meter composition, and coverage.
2. Build a reproducible, leakage-safe preprocessing and feature-generation pipeline.
3. Implement and compare at least three regression models: Linear Regression, Random Forest, and HistGradientBoosting.
4. Select a final model using validation data only and report final test metrics separately.
5. Quantify error patterns and generalization to unseen buildings.
6. Demonstrate two previously unseen input cases through a reusable inference interface.
7. Produce three to five evidence-supported engineering insights.
8. Prepare Tableau-ready data and a build guide, plus a supplementary Streamlit dashboard.
9. Document limitations, reproducibility, and the consequences of incorrect predictions.

## 1.5 Scope

**In scope**

- Electricity meter readings (`meter == 0`) only.
- 2016 training data and a chronological holdout from the same year.
- Site-level weather as provided by the competition.
- Building metadata available in `building_metadata.csv`.
- One local subset model and one unseen-building experiment.
- Dashboard data and a Tableau construction guide.

**Out of scope**

- Chilled-water, steam, and hot-water meters.
- 2017–2018 supervised evaluation (no labels).
- Sub-metered end-use disaggregation.
- Occupancy, equipment schedules, tariff data, and economic optimization.
- A claim of measured energy savings or an energy-efficiency rating.
- A published Tableau workbook, which requires the user's Tableau account and validation.

## 1.6 Expected engineering contribution

The contribution is a traceable workflow that connects data quality to model behavior and operational use. It shows which features drive predictions, where errors concentrate, and why a high test R² does not establish deployment readiness. The dashboard and report translate the model into building-level monitoring concepts while preserving the distinction between a predicted meter reading and a verified efficiency improvement.

---

# Chapter 2: Dataset and Problem Formulation

## 2.1 Dataset origin and access

| Item | Detail |
|---|---|
| Dataset | ASHRAE Great Energy Predictor III |
| Kaggle identifier | `sumit261124/ashrae-great-energy-predictor-iii-dataset` |
| URL | https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset |
| Original competition | ASHRAE Great Energy Predictor III (2019) |
| Access date | 20 October 2026 |
| Access method | KaggleHub download |
| Local raw files | `data/raw/` (retained unchanged) |

The Kaggle re-upload does not state a separate open-data license. The files are governed by the Kaggle competition rules and the dataset's Kaggle terms. They are used here for academic coursework and are not redistributed beyond the course submission. The original competition and the re-upload are cited in the references.

## 2.2 Dataset structure

| File | Rows | Meaning |
|---|---:|---|
| `train.csv` | 20,216,100 | 2016 hourly meter readings |
| `test.csv` | 41,697,600 | 2017–2018 feature rows; no labels |
| `building_metadata.csv` | 1,449 | Site, primary use, floor area, year built, floor count |
| `weather_train.csv` | 139,773 | 2016 hourly weather by site |
| `weather_test.csv` | 277,243 | 2017–2018 hourly weather by site |
| `sample_submission.csv` | 41,697,600 | Submission template only |

Audit results:

- 1,449 buildings and 16 sites.
- Four meter types: 0 electricity, 1 chilled water, 2 steam, 3 hot water.
- Electricity rows: 12,060,910 across 1,413 buildings, with 4.40% zero readings.
- Zero duplicate `(building_id, meter, timestamp)` keys in training.
- 36 buildings have no electricity readings.
- 13 electricity buildings have fewer than half a year of readings.
- All 1,449 test buildings also appear in training, but the test features cover future years with no labels.

## 2.3 Target definition and unit caution

The target is `meter_reading` for **meter 0 electricity only**. The downloadable CSV does not include a unit column. The competition documentation describes the target as energy consumed in kWh. This report therefore uses **dataset energy units (documented as kWh)** and does not independently verify the unit. A summed reading over a period is an energy-like quantity under that documentation, but it is not instantaneous power, a building efficiency rating, or measured savings. All dashboard totals carry this caveat.

## 2.4 Input features

**Building metadata:** `building_id`, `site_id`, `primary_use`, `square_feet`, `year_built`, `floor_count`.

**Calendar:** `hour`, `dayofweek`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`.

**Derived:** `building_age` = observation year minus `year_built`, set missing when negative.

**Weather:** `air_temperature`, `dew_temperature`, `wind_speed`, `cloud_coverage`, `precip_depth_1_hr`, `sea_level_pressure`.

**Weather-quality flags:** `cloud_coverage_missing`, `precip_depth_1_hr_missing`, `sea_level_pressure_missing`.

The 24 model features are listed in `src/config.py` and are identical between SQL training-data construction and Python inference. A training-time consistency assertion verifies this equality.

## 2.5 Regression formulation

For building *i* and hour *t*, the model estimates:

\[
\hat{y}_{i,t} = f(\text{building metadata}_i,\ \text{calendar}_t,\ \text{weather}_{s(i),t})
\]

where *s(i)* is the building's site. The loss is squared error for Random Forest and HistGradientBoosting regression; Linear Regression minimizes squared error. Evaluation uses absolute and relative error measures described in Chapter 5.

The model is a **supervised, cross-sectional time-series regression** with building identity. It is not a recursive forecast: it uses observed weather at the target hour, not forecast weather. This is appropriate for retrospective monitoring and counterfactual analysis, but it is not directly a day-ahead forecasting system.

## 2.6 Research questions

1. Which of the three models provides the best validation and test performance under the same split?
2. Which features drive predictions, and are their effects physically plausible?
3. How much performance is lost when predicting buildings that were not in training?
4. Where do errors concentrate, and what does that imply for monitoring?
5. What are the consequences of over- and under-prediction for building operations?

## 2.7 Assumptions

- Meter 0 readings represent the building's electricity demand over the hourly interval.
- Site-level weather is a reasonable proxy for building-level outdoor conditions.
- Building metadata is static over 2016.
- Zero readings may represent genuine zero demand, meter downtime, or missing data; they are retained and their ambiguity is acknowledged.
- The chronological split approximates deployment: train on the past, evaluate on the future.
- The December test period is not used to choose features, hyperparameters, or the model.

---

# Chapter 3: Data Audit and Preprocessing

## 3.1 Data-quality audit

The audit was computed with DuckDB directly from the raw files. The full report is `reports/data_inventory.md`.

| Check | Result | Consequence |
|---|---|---|
| Duplicate business keys in train | 0 | No deduplication needed |
| Duplicate site/timestamp weather keys | 0 | Safe many-to-one merge |
| Electricity rows | 12,060,910 | Enough data for nonlinear models |
| Electricity buildings | 1,413 | 36 buildings have no electricity meter |
| Zero electricity readings | 530,169 (4.40%) | Retained; zero is ambiguous |
| Negative target values | 0 | Clipping lower bound is zero |
| `year_built` missing (full metadata) | 53.42% | Retained as missing |
| `floor_count` missing (full metadata) | 75.50% | Retained as missing |
| `square_feet` missing | 0% | Usable as a primary size feature |
| `primary_use` missing | 0% | Usable as a categorical feature |

The local subset has 120 buildings and 1,030,967 rows. Its metadata missingness is `year_built` 51.76% and `floor_count` 70.90%, close to the full-data pattern.

## 3.2 Missing values

Missing values were handled according to their meaning, not replaced by arbitrary constants.

**Building metadata.** `year_built` and `floor_count` are missing for many buildings. Models that can handle missing values receive them directly; Linear Regression uses median imputation plus an indicator for missingness. No building's age is invented. `building_age` is missing when `year_built` is missing or would be negative.

**Weather.** The raw weather data have substantial missingness: cloud cover 49.49%, precipitation 35.98%, and sea-level pressure 7.60% in the full training weather file. Air temperature, dew temperature, and wind speed are nearly complete (0.04%, 0.08%, and 0.22% missing, respectively).

The cleaning rules are:

1. Sort by site and timestamp.
2. Treat negative precipitation as the documented missing sentinel (`-1`), not as rainfall.
3. Treat negative wind speed as missing.
4. Treat sea-level pressure outside 800–1100 hPa as missing.
5. Forward-fill each weather variable **within site**, with a **three-hour limit**, so only past observations are used.
6. Create missing-indicator columns before filling for cloud cover, precipitation, and pressure.
7. Leave unresolved values as NaN for models with native missing-value support; Linear Regression imputes with the training median.

After forward fill in the local processed dataset, unresolved missingness is 24.68% for cloud cover, 25.32% for precipitation, 7.57% for pressure, and 0.41% for air temperature, dew temperature, and wind speed. The missing-flag prevalence is 45.70%, 29.72%, and 8.74%, respectively; flags record pre-fill gaps, so they are not identical to post-fill NaN rates.

**Merge gaps.** If a building-site has no weather row for an hour, the weather fields remain NaN and the missing flags are set to 1. This prevents "no weather record" from being silently interpreted as calm, clear, or dry conditions.

## 3.3 Duplicates

There are no duplicate `(building_id, meter, timestamp)` keys in training and no duplicate `(site_id, timestamp)` keys in weather. Repeated timestamps across different buildings are expected and are not duplicates. No rows were dropped for duplication.

## 3.4 Outliers and zero readings

The electricity target is strongly right-skewed: median 62.832, mean 170.826, 99th percentile 1,629.61, and maximum 79,769 in the full electricity data. The local subset has median 59.05, mean 188.78, 99th percentile 2,454, and maximum 41,552.

Large peaks were **not automatically deleted**. A high reading can be a legitimate central plant, a simultaneous HVAC startup, or a real event; deleting it would bias the model against the conditions that matter for demand planning. Negative predictions are clipped to zero at inference, consistent with the physical lower bound. Zero readings are retained because zero can mean no demand, a meter outage, or a reporting gap; the dataset does not distinguish these causes. The limitation is discussed in Chapter 7.

## 3.5 Data types and categories

- `building_id` and `site_id` are categorical identifiers, not continuous quantities. They are one-hot encoded for Linear Regression and ordinal encoded for tree models. Ordinal codes are not interpreted as numeric distances by the trees; they simply identify categories.
- `primary_use` is a 16-level categorical variable.
- `hour`, `dayofweek`, and `month` are calendar integers; cyclical sine/cosine versions preserve the wrap-around structure.
- Weather variables are numeric and remain on their original scales for tree models. Linear Regression uses standardization after imputation.

## 3.6 Data merging

The processed dataset is built in one DuckDB query:

1. Read the typed Parquet electricity rows.
2. Join building metadata on `building_id`.
3. Join cleaned weather on `(site_id, timestamp)`.
4. Derive calendar, cyclical, and building-age features.
5. Assign each row to train, validation, or test by timestamp.
6. Write a ZSTD-compressed Parquet file.

This avoids loading the 20-million-row training CSV into pandas and avoids repeated large copies.

## 3.7 Feature engineering

| Feature group | Features | Rationale |
|---|---|---|
| Identity | `building_id`, `site_id`, `primary_use` | Capture persistent building and site effects |
| Size | `square_feet`, `floor_count`, `year_built`, `building_age` | Proxy for scale, envelope, and vintage |
| Daily cycle | `hour`, `hour_sin`, `hour_cos`, `is_weekend`, `dayofweek` | Occupancy and HVAC schedules |
| Annual cycle | `month`, `month_sin`, `month_cos` | Seasonal weather and operational patterns |
| Weather | Air temperature, dew temperature, wind, cloud, precipitation, pressure | HVAC load and ambient conditions |
| Data quality | Three missing flags | Distinguish a true zero/clear condition from missing data |

Cyclical encodings are important because hour 23 and hour 0 are adjacent, and December and January are adjacent. A raw integer encoding would make those pairs appear far apart to a linear model.

## 3.8 Chronological split

| Split | Period | Rows | Buildings | Mean reading | Zero readings |
|---|---|---:|---:|---:|---:|
| Train | 2016-01-01 to 2016-09-30 | 769,122 | 120 | 186.76 | 44,796 |
| Validation | 2016-10-01 to 2016-11-30 | 174,537 | 120 | 194.07 | 662 |
| Test | 2016-12-01 to 2016-12-31 | 87,308 | 120 | 196.00 | 126 |

The split is strictly chronological. No random shuffle is used. This mimics deployment: the model is fitted on earlier data and evaluated on later data.

## 3.9 Leakage prevention

The following controls were used:

1. **Time-ordered split.** Training never contains rows later than validation, and validation never contains rows later than test.
2. **No target-derived features.** The model does not use lagged readings, rolling means of the target, or any transformation computed from future readings.
3. **Past-only weather fill.** Forward fill uses only earlier timestamps within the same site, with a three-hour limit. No backward fill is used.
4. **Pipeline-fitted preprocessing.** Median imputation, scaling, and categorical encoding are fitted inside each model pipeline on training data only.
5. **No test-driven selection.** The test period is not used to choose the model, features, or hyperparameters.
6. **No future-year labels.** The 2017–2018 Kaggle test features have no labels and are not used for supervised evaluation.

A subtle leakage risk remains: `building_id` is a feature, and the main test buildings are present in training. This is legitimate for the interpolation task but means the main result is **seen-building** performance. The building-holdout experiment isolates this issue.

## 3.10 Local subset selection

The full electricity dataset has 12.06 million rows and 1,413 buildings. The local machine has 4 CPU cores and 3.6 GB RAM. To keep the workflow reproducible and within memory, a site-stratified subset was selected:

- Candidate buildings have at least 80% electricity coverage (at least 7,027 of 8,784 hours).
- Each site receives a proportional allocation with a minimum of two buildings.
- Within each site, primary-use groups are allocated proportionally with a minimum of one building where possible.
- The selection seed is 42.
- The result is 120 buildings across all 16 sites and all 16 primary-use categories, with 1,030,967 hourly rows.

This is a **documented subset**, not a random tiny sample. It preserves site and use diversity but changes the building mix relative to the full data. All results are labelled accordingly.

## 3.11 Raw versus processed records

The raw CSVs are never modified. The processed dataset is reproducible from:

```bash
python -m src.data_loading
python -m src.preprocessing
```

The raw-to-processed row accounting is:

- Full training: 20,216,100 rows.
- Electricity filter: 12,060,910 rows.
- Local subset: 1,030,967 rows.
- Split: 769,122 + 174,537 + 87,308 = 1,030,967 rows.

No generated or synthetic readings are substituted for missing data.

---

# Chapter 4: Exploratory Data Analysis

## 4.1 Descriptive statistics

| Statistic | Value |
|---|---:|
| Observations | 1,030,967 |
| Buildings | 120 |
| Sites | 16 |
| Median reading | 59.05 |
| Mean reading | 188.78 |
| 95th percentile | 696.65 |
| 99th percentile | 2,454 |
| Maximum | 41,552 |
| Zero share | 4.42% |

The distribution is right-skewed. The mean is about 3.2 times the median, so aggregate averages are influenced by high-load buildings and peak hours.

## 4.2 Temporal patterns

- Weekday mean: 195.44; weekend mean: 172.24; ratio: 1.135.
- Peak weekday hour: 13:00; trough: 03:00.
- Monthly means range from 166.60 (April) to 215.14 (August).
- Monthly mean outdoor temperature ranges from 4.45 °C (January) to 25.38 °C (July).
- Daily mean reading ranges from 146.87 (15 May) to 249.89 (29 August).

The midday peak and weekend reduction are consistent with occupancy and HVAC schedules. August's high demand is consistent with cooling load, while January remains relatively high because heating and baseline loads persist.

Figures: `reports/figures/temporal_patterns.png`.

## 4.3 Temperature relationship

The pooled Pearson correlation between air temperature and reading is **-0.074**, which is weak and can be misread as "temperature does not matter." The site-level analysis shows a range from **-0.134** (site 10) to **+0.406** (site 11).

| Site | Pearson r | Mean reading below 10 °C | Mean reading 10–25 °C | Mean reading above 25 °C |
|---|---:|---:|---:|---:|
| 0 | +0.234 | 22.07 | 92.29 | 151.67 |
| 6 | -0.068 | 357.22 | 338.12 | 324.72 |
| 7 | -0.065 | 1,213.99 | 1,107.74 | 959.17 |
| 10 | -0.134 | 123.27 | 112.94 | 105.57 |
| 11 | +0.406 | 265.74 | 335.82 | 377.17 |

Pooling sites with opposite responses cancels the effect. This is a key engineering insight: a single global temperature coefficient is not appropriate.

Figures: `reports/figures/temperature_relationship.png`; `reports/figures/temperature_response_by_site.png`.

## 4.4 Primary use and site

Median reading by primary use:

| Primary use | Median | Mean |
|---|---:|---:|
| Religious worship | 3.75 | 4.59 |
| Manufacturing/industrial | 12.21 | 13.06 |
| Warehouse/storage | 23.17 | 56.26 |
| Parking | 24.06 | 37.04 |
| Services | 32.34 | 72.26 |
| Office | 36.83 | 72.03 |
| Other | 38.67 | 114.09 |
| Retail | 44.13 | 56.93 |
| Healthcare | 49.29 | 168.61 |
| Utility | 52.90 | 66.34 |
| Technology/science | 54.30 | 105.78 |
| Food sales and service | 56.10 | 74.92 |
| Lodging/residential | 78.45 | 114.98 |
| Public services | 89.25 | 156.92 |
| Entertainment/public assembly | 116.25 | 204.26 |
| Education | 157.20 | 485.24 |

Site mean readings range from 10.24 (site 5) to 1,156.11 (site 7). Site 7 is represented by only three buildings with a mean of 1,156.11, so its ranking is sensitive to those buildings. Education dominates the subset with 24 buildings and 208,307 rows.

Figures: `reports/figures/primary_use_consumption.png`; `reports/figures/site_comparison.png`.

## 4.5 Building size

Across the 120 buildings, the correlation between log10 floor area and mean reading is **0.750**. Median floor area is 45,264.5 ft². This is the strongest simple building-level relationship in the EDA, although it is associational and does not capture equipment intensity or occupancy.

Figure: `reports/figures/building_size_relationship.png`.

## 4.6 Five engineering insights

The five evidence-backed insights are documented in detail in `reports/engineering_insights.md`. In summary:

1. **Demand is right-skewed and building-dominated.** Median 59.05 vs mean 188.78; size log-correlation 0.750; `square_feet` is the top permutation feature. Benchmark within building/use groups and normalize by floor area.
2. **Occupancy schedules create a weekday/midday cycle.** Weekday mean 195.44 vs weekend 172.24; peak at 13:00, trough at 03:00. Target setback and scheduling measures.
3. **Temperature response is site-specific.** Pooled r = -0.074, but site r ranges from -0.134 to +0.406. Do not use one global linear temperature coefficient.
4. **Primary use and site dominate.** Median reading ranges from 3.75 to 157.20 by use and site means range from 10.24 to 1,156.11. Segment reporting and modeling.
5. **Seen-building accuracy does not transfer to unseen buildings.** HGB test R² 0.970 on seen buildings vs -0.555 on held-out buildings. Do not deploy to new buildings without calibration or richer descriptors.

## 4.7 Correlation and feature relevance

The correlation matrix shows that floor area (0.434 with reading), floors (0.295), dew temperature (-0.053), air temperature (-0.072), and building age (0.028) have different simple relationships with the raw target. These are not partial effects; they mix building composition, time, and weather. The model and permutation importance provide a more useful multivariate view.

Figure: `reports/figures/correlation_matrix.png`.

---

# Chapter 5: AI/ML Methodology

## 5.1 Model choices

### Mean baseline

A `DummyRegressor(strategy="mean")` predicts the training mean for every row. It is not a candidate for deployment; it establishes the error scale and confirms that the learned models add value.

### Linear Regression

Linear Regression is the interpretable baseline. It assumes additive effects and a linear relationship between the transformed features and the target. Categorical variables are one-hot encoded, numeric variables are median-imputed and standardized, and missingness indicators are added by `SimpleImputer(add_indicator=True)`. The design matrix is sparse.

**Suitability:** transparent coefficients and a clear reference. **Limitation:** cannot represent nonlinear temperature response or complex interactions without manual feature crosses, and the target's skew makes squared-error fitting sensitive to peaks.

### Random Forest Regressor

Random Forest is an ensemble of decorrelated decision trees. It captures nonlinearities and interactions and is robust to irrelevant features. The configuration is:

- 80 trees
- maximum depth 16
- minimum samples per leaf 20
- `max_features=0.5`
- `n_jobs=2`
- seed 42
- 250,000-row seeded training sample on the local subset

**Suitability:** strong tabular baseline, handles nonlinearity and missing values in the installed scikit-learn version. **Limitation:** large memory footprint, slower training, and less efficient than boosting at this scale.

### HistGradientBoosting Regressor

HistGradientBoosting is a histogram-based gradient-boosting model designed for large tabular datasets. It builds trees sequentially, each correcting the previous ensemble's errors. The configuration is:

- maximum 200 iterations
- learning rate 0.08
- 31 maximum leaves
- minimum samples per leaf 20
- L2 regularization 1.0
- early stopping with a 10% internal validation fraction and 15-iteration patience
- seed 42
- all 769,122 training rows on the local subset

**Suitability:** efficient on large data, handles missing values natively, captures nonlinearities and interactions, and is the best performer on validation RMSE. **Limitation:** less interpretable than Linear Regression; the internal early-stopping split is random within training, not a replacement for the chronological validation split.

## 5.2 Preprocessing pipelines

| Model | Categorical handling | Numeric handling | Missing values |
|---|---|---|---|
| Mean baseline | None | None | Not used |
| Linear Regression | One-hot encoding, ignore unknown | Median imputation, missing indicator, standardization | Imputed |
| Random Forest | Ordinal encoding, unknown = -1 | Passthrough | Native support |
| HistGradientBoosting | Ordinal encoding, unknown = -1 | Passthrough | Native support |

All preprocessing is inside the scikit-learn `Pipeline`/`ColumnTransformer`, so it is fitted on training data and applied unchanged to validation and test. This prevents validation leakage through imputation or scaling.

## 5.3 Training and validation design

- Train: January–September 2016.
- Validation: October–November 2016.
- Test: December 2016.
- Model selection: lowest validation RMSE. The rule was fixed before looking at test results.
- Test evaluation: once per model, on identical rows.
- Metrics: MAE, RMSE, R², and RMSLE.
- Random Forest sampling: seeded 250,000 rows from the training split; Linear Regression and HGB use all training rows locally.

The validation period is autumn, while the test period is winter. This is a deliberate temporal holdout, but it also means the test set is a different season. The December test has only 126 zero readings versus 44,796 in training (5.83% vs 0.02% of rows). This distribution shift is discussed in Chapter 7.

## 5.4 Evaluation metrics

For actual values \(y_i\) and predictions \(\hat{y}_i\):

\[
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

\[
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

\[
R^2 = 1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
\]

\[
\mathrm{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log(1+\hat{y}_i)-\log(1+y_i)\right)^2}
\]

Predictions are clipped to zero before metrics are computed. MAE is in dataset units and is easy to explain to a facilities team. RMSE penalizes large errors more heavily, which matters for peak demand. R² measures variance explained relative to the mean. RMSLE emphasizes relative error and is useful for the wide range of building sizes, but it becomes unstable when actual values are near zero.

## 5.5 Computational considerations

The local machine has 4 CPU cores and 3.6 GB RAM. Training times on the subset were approximately:

| Model | Fit time (s) | Predict time (s) |
|---|---:|---:|
| Mean baseline | 0.59 | 0.02 |
| Linear Regression | 53.39 | 4.35 |
| Random Forest | 232.40 | 5.94 |
| HistGradientBoosting | 53.91 | 7.32 |

Random Forest took about 4.3 times longer to fit than HGB and produced a larger model. HGB's efficiency is one reason it is attractive for a larger run, in addition to its lower validation RMSE.

---

# Chapter 6: Results and Comparison

## 6.1 Quantitative comparison

| Model | Split | Rows | MAE | RMSE | R² | RMSLE |
|---|---|---:|---:|---:|---:|---:|
| Mean baseline | Validation | 174,537 | 205.21 | 411.67 | -0.0003 | 1.922 |
| Mean baseline | Test | 87,308 | 209.57 | 406.05 | -0.0005 | 1.920 |
| Linear Regression | Validation | 174,537 | 68.40 | 223.03 | 0.706 | 0.981 |
| Linear Regression | Test | 87,308 | 61.39 | 154.51 | 0.855 | 0.942 |
| Random Forest | Validation | 174,537 | 29.35 | 166.92 | 0.836 | 0.426 |
| Random Forest | Test | 87,308 | 32.05 | 78.90 | 0.962 | 0.448 |
| HistGradientBoosting | Validation | 174,537 | 32.66 | **165.74** | **0.838** | 0.578 |
| HistGradientBoosting | Test | 87,308 | 33.20 | **69.78** | **0.970** | 0.591 |

All learned models beat the mean baseline by a wide margin. The baseline's negative R² is expected: the test-period mean differs slightly from the training mean.

## 6.2 Model selection

The pre-declared rule is lowest validation RMSE. HistGradientBoosting wins with 165.74 versus Random Forest's 166.92 (a difference of about 0.7%). The test results are consistent with this choice on RMSE and R²: HGB has RMSE 69.78 and R² 0.970 versus RF's 78.90 and 0.962.

The selection is not unanimous across metrics:

- Random Forest has a lower validation MAE (29.35 vs 32.66) and a much lower validation RMSLE (0.426 vs 0.578).
- On test, RF still has lower MAE (32.05 vs 33.20) and RMSLE (0.448 vs 0.591).
- HGB has lower RMSE and higher R² on both validation and test.

This trade-off is physically meaningful. MAE treats all hours equally, while RMSE penalizes large errors at high-load hours more heavily. For demand planning, large peak errors are costly, so RMSE is a defensible selection criterion. HGB also trains much faster than RF. If the operational objective were median absolute accuracy across many low-load hours, RF would be the better choice. The report states this explicitly rather than implying that one model dominates on every metric.

## 6.3 Model performance plots

- `reports/figures/model_comparison.png` — test MAE, RMSE, and R² by model.
- `reports/figures/actual_vs_predicted.png` — linear and log-scale actual-versus-predicted scatter for the selected model.
- `reports/figures/residual_analysis.png` — residual distribution and residual-versus-predicted plot.
- `reports/figures/feature_importance.png` — permutation importance for the selected model.

The actual-versus-predicted plot shows tight clustering around the identity line for moderate and high loads, with more scatter at low loads. The residual plot shows that errors are not homoscedastic: large positive residuals occur at high actual loads, and the model has a tendency to overpredict some low-load observations.

## 6.4 Error analysis by reading band

| Actual reading band | Rows | MAE | RMSE | R² | RMSLE |
|---|---:|---:|---:|---:|---:|
| 0–10 | 12,015 | 12.35 | 27.99 | -111.61 | 1.230 |
| 10–50 | 28,334 | 14.44 | 22.01 | -2.585 | 0.528 |
| 50–100 | 12,016 | 19.78 | 29.13 | -2.952 | 0.308 |
| 100–250 | 18,312 | 32.67 | 46.70 | -0.543 | 0.262 |
| 250–500 | 7,442 | 68.49 | 96.38 | -1.505 | 0.343 |
| 500–1000 | 6,546 | 92.55 | 148.00 | -0.112 | 0.399 |
| 1000–5000 | 2,643 | 147.23 | 229.58 | 0.924 | 0.147 |

Absolute error grows with load, which is expected because higher loads have more absolute variability. R² within narrow bands is often negative even when MAE is modest, because the band's variance is small; R² is not a stable within-stratum measure here. RMSLE is largest in the 0–10 band, where small absolute errors become large relative errors.

This matters for monitoring: a 12-unit error at a near-zero building may be operationally significant relative to its baseline, while a 147-unit error at a 2,000-unit building is proportionally smaller.

## 6.5 Error analysis by site

| Site | Rows | MAE | RMSE | R² | Actual mean | Predicted mean |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 2,232 | 134.94 | 210.12 | 0.938 | 1,447.27 | 1,383.71 |
| 0 | 6,696 | 79.91 | 140.57 | 0.463 | 168.30 | 129.30 |
| 11 | 1,486 | 74.01 | 89.42 | -1.728 | 243.99 | 315.46 |
| 14 | 5,930 | 45.81 | 83.55 | 0.840 | 209.18 | 209.65 |
| 13 | 8,179 | 38.19 | 84.83 | 0.990 | 475.33 | 481.98 |
| 5 | 5,952 | 5.05 | 7.42 | 0.430 | 10.94 | 11.19 |

Site 7 has the largest absolute errors because its loads are an order of magnitude above the others. Site 11's predictions are biased high (315.46 vs 243.99), and its R² is negative. Site 0 is biased low (129.30 vs 168.30). Site-level systematic bias suggests that site-specific calibration or site-temperature interactions would help.

## 6.6 Generalization to unseen buildings

The main model uses `building_id` and all test buildings are present in training. To test transfer, 20% of buildings were held out entirely, stratified by site, and HGB was retrained on the remaining buildings' January–September data.

| Group | Rows | MAE | RMSE | R² | RMSLE | Actual mean | Predicted mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unseen buildings, test | 18,937 | 150.51 | 321.33 | **-0.555** | 1.525 | 176.99 | 242.49 |
| Seen buildings, test | 68,371 | 33.86 | 74.15 | 0.971 | 0.595 | 201.27 | 204.98 |
| Seen buildings, validation | 136,713 | 33.03 | 183.88 | 0.831 | 0.576 | 200.12 | 203.77 |

The unseen-building model overpredicts on average (242.49 vs 176.99) and explains less variance than the mean. This is the most important result in the report. It shows that the high seen-building R² is largely building-specific interpolation, not a general physical model.

Figure: `reports/figures/unseen_buildings_actual_vs_predicted.png`.

## 6.7 Unseen-case demonstrations

Two held-out rows were selected from the test predictions and passed through the reusable `ElectricityPredictor` interface. These are previously unseen input cases, not training rows.

| Case | Building | Timestamp | Site | Primary use | Actual | Predicted | Absolute error | Percentage error |
|---|---|---:|---:|---|---:|---:|---:|---:|
| High demand | 1088 | 2016-12-19 15:00 | 13 | Education | 3,427.60 | 3,417.33 | 10.27 | 0.30% |
| Typical demand | 386 | 2016-12-09 06:00 | 3 | Entertainment/public assembly | 41.16 | 41.16036 | 0.00036 | 0.0009% |

The first case is physically plausible: a large education building (287,594 ft²) at -10.6 °C has high winter demand, and the model's error is small. The second case is almost exact; this is better than expected and should not be interpreted as evidence of general accuracy. Tree models can reproduce a leaf value closely when the feature combination is similar to training rows. The building-holdout result is the appropriate generalization test.

The demonstration also verifies the feature-consistency assertion between the SQL-built training frame and the Python inference frame.

---

# Chapter 7: Engineering Discussion

## 7.1 Feature interpretation

Permutation importance for the selected model (increase in MAE when a feature is permuted) is:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `square_feet` | 178.43 |
| 2 | `primary_use` | 83.33 |
| 3 | `building_id` | 53.68 |
| 4 | `site_id` | 35.79 |
| 5 | `floor_count` | 19.49 |
| 6 | `year_built` | 12.36 |
| 7 | `hour_cos` | 8.12 |
| 8 | `is_weekend` | 4.45 |
| 9 | `dew_temperature` | 3.45 |
| 10 | `building_age` | 1.75 |

The ranking says that building scale, function, identity, and location dominate. Weather variables have smaller individual permutation importance because they are correlated with each other and because building identity already absorbs much of the between-building variation. This does **not** mean weather is unimportant for a single building; the site-level temperature analysis shows strong positive or negative responses that are masked in the global ranking.

Permutation importance is also not causal. Correlated features can share or hide importance, and permuting `building_id` creates unrealistic combinations of building identity and other features.

## 7.2 Physical plausibility

Several results are physically sensible:

- Larger buildings have higher mean demand; floor area is the strongest building-level correlate.
- Education and entertainment buildings have high demand relative to religious buildings and parking.
- Weekday midday demand exceeds weekend and nighttime demand, consistent with occupancy and HVAC schedules.
- Cold sites show heating-driven demand in some cases, while warm sites show cooling-driven demand in others.
- A large education building at -10.6 °C has a high predicted winter load.

Some results are less comfortable:

- Site 11 is overpredicted and has negative R².
- The model overpredicts low-load observations in several bands.
- The pooled temperature correlation is weak and can mislead if not segmented.
- Unseen buildings are poorly predicted.

These are not cosmetic issues. They define the conditions under which the model can be used.

## 7.3 Operational implications

**Monitoring.** For a building with a known history, the model can provide an expected hourly reading. A persistent positive residual (actual above predicted) may indicate a control fault, a schedule change, or additional equipment; a persistent negative residual may indicate a setback, an outage, or a data problem. Residual thresholds should be building-specific and validated against known events.

**Demand planning.** The model can support what-if analysis for weather and calendar conditions, but it uses observed weather rather than forecasts. A day-ahead use case would require forecast weather and a proper forecast-error evaluation.

**Benchmarking.** Primary use and floor area should be used to form peer groups. A raw mean reading across all buildings is not a fair efficiency benchmark.

**Scheduling.** The 13:00 peak and 03:00 trough suggest opportunities for HVAC setback and pre-cooling studies, but the dataset cannot prove savings. Any scheduling change should be measured with a controlled before/after design.

## 7.4 Limitations

1. **Subset scope.** Results use 120 of 1,413 electricity buildings. The building mix and site weights differ from the full dataset.
2. **Seen-building dependence.** `building_id` is a feature; unseen-building R² is -0.555.
3. **Seasonal and zero-reading shift.** The test month is December, with only 126 zero readings versus 44,796 in training. Test metrics may be optimistic for periods with missing or zero data.
4. **Single year.** The model is trained and tested within 2016. It cannot capture year-to-year drift, retrofits, or operational changes.
5. **Weather proxy.** Weather is site-level, not building-level. Microclimate, urban heat islands, and sensor placement are unobserved.
6. **Metadata missingness.** `year_built` is missing for 51.76% of subset buildings and `floor_count` for 70.90%. Missingness is handled, but the information is genuinely absent.
7. **No occupancy or schedules.** The calendar features are proxies. Holidays, tenant changes, and equipment schedules are not available.
8. **Target ambiguity.** Zero readings may be true zeros or missing data; the dataset does not distinguish them. Units are documented but not independently verified.
9. **No causal claim.** Feature importance and correlations are associational. The model does not estimate the effect of a retrofit or a control change.
10. **Dashboard status.** No Tableau workbook has been authored or published; the project supplies data and a guide. The Streamlit app is supplementary.
11. **Full-data metrics pending.** The Colab workflow is prepared but not executed; no full-data numbers are claimed.

## 7.5 Consequences of incorrect predictions

**Underprediction at peak.** If the model predicts less demand than occurs, a facility may under-procure capacity, miss a demand-charge threshold, or fail to pre-cool before a peak. The consequence is financial and possibly a comfort/reliability risk.

**Overprediction.** If the model predicts more demand than occurs, a facility may over-schedule chillers or over-procure capacity, wasting energy and maintenance effort. Persistent overprediction can also mask a genuine reduction caused by an efficiency measure.

**False anomaly alarms.** A model that overpredicts low-load hours may generate false alarms, eroding trust in the monitoring system. The 0–10 band's RMSLE of 1.23 shows that relative errors are large precisely where loads are small.

**Unseen-building deployment.** Using the model on a new building without calibration could produce errors larger than the building's mean demand. The unseen-building MAE of 150.51 versus an actual mean of 176.99 is a warning, not a marginal degradation.

## 7.6 Practical recommendations

1. Use the model only for buildings with sufficient historical data, and report seen-building metrics separately from unseen-building metrics.
2. Add building-level history features (lagged readings, rolling medians) computed strictly from the past; these are likely to improve accuracy but require careful leakage control.
3. Add heating/cooling degree hours and site-by-temperature interactions rather than one global temperature coefficient.
4. Add holiday, semester, and occupancy proxies where available; education buildings are a large share of the subset.
5. Treat zero and missing readings explicitly; evaluate a classifier for zero/non-zero and a regressor for non-zero loads.
6. Calibrate residuals by building and by load band before using them for anomaly detection.
7. Validate on a future year and on a building-holdout split before any deployment claim.
8. Use the full data in Colab and compare subset versus full-data metrics; do not extrapolate the subset R² to the full portfolio.
9. Build the Tableau workbook and verify filters/interactions before claiming a dashboard deliverable.
10. Preserve the distinction between predicted meter readings, energy savings, and efficiency; do not label a sum of readings as instantaneous power.

---

# Chapter 8: Dashboard

## 8.1 Architecture

The dashboard layer has three parts:

1. **Dashboard-ready CSVs** in `data/dashboard/tableau_data/`, generated by `src/build_dashboard_data.py`.
2. **Tableau specification and build guide** in `dashboard/`.
3. **Supplementary Streamlit app** in `dashboard/streamlit_app.py`.

The main electricity extract contains 1,030,967 rows. Model test predictions contain 349,232 rows (four models on 87,308 test rows). The data dictionary is `data/dashboard/README.md`.

## 8.2 KPI definitions

| KPI | Definition | Caveat |
|---|---|---|
| Total electricity reading | Sum of `meter_reading` over selected observations | Dataset energy units (documented kWh); not power, efficiency, or savings |
| Average reading per observation | Mean of selected hourly readings | Sensitive to building mix and peak hours |
| Number of buildings | Distinct `building_id` in selection | Subset has 120; full electricity has 1,413 |
| Number of sites | Distinct `site_id` in selection | 16 |
| Selected period | Minimum/maximum timestamp after filters | Test dashboard is December 2016 |
| Model MAE / RMSE / R² | Held-out test metrics from `model_comparison.csv` | Four models, same observations |

## 8.3 Tableau dashboards

**Dashboard 1 — Energy Overview.** KPI cards for total reading, average reading, building count, site count, and selected period; hourly/daily trend; site and primary-use comparisons. Filters: date range, site, primary use. The total is explicitly labelled as a sum of readings, not instantaneous power.

**Dashboard 2 — Weather and Building Analysis.** Consumption versus outdoor temperature, hourly and monthly patterns, primary-use comparison, building-size relationship, and a building-level table. Filters: date, site, building, primary use.

**Dashboard 3 — Model Performance.** Model comparison table, MAE/RMSE/R², actual-versus-predicted scatter, residual or absolute-error distribution, error by site/use/load band, selected model explanation, and generalization warning. Filters: model, site, primary use, reading band.

The full design, field mappings, chart types, filter actions, and layout guidance are in `dashboard/dashboard_specification.md`. The step-by-step build sequence is in `dashboard/tableau_build_guide.md`.

## 8.4 Streamlit application

The supplementary Streamlit app has four pages:

1. **Energy Overview** — KPIs, hourly/daily trends, site and use comparison.
2. **Weather & Buildings** — temperature response, building-size relationship, metadata filters.
3. **Model Performance** — model comparison, actual-vs-predicted, residuals, error tables.
4. **Predict** — single-record prediction from the saved model and weather lookup.

It was smoke-tested with `streamlit.testing.v1.AppTest` across all four pages after replacing deprecated `use_container_width` calls with `width="stretch"`. It is not a substitute for the Tableau Public dashboard.

Run:

```bash
streamlit run dashboard/streamlit_app.py
```

## 8.5 Interaction design and key findings

The intended interaction is filter-first: select a date range, site, primary use, and building, then read the KPIs and trends. The model page should keep the comparison table global (all models, all test rows) while the scatter and residual plots respond to filters. This avoids the misleading impression that a filtered subset is a new model evaluation.

Key dashboard messages:

- Demand is concentrated in a subset of large, high-use buildings.
- Weekday midday demand is higher than weekend/night demand.
- Temperature sensitivity differs by site; a single global curve is misleading.
- The selected model is accurate for seen buildings but not for unseen buildings.
- Total readings are not power or savings.

## 8.6 Access instructions and honesty

No Tableau workbook has been authored, published, or tested. To complete the dashboard deliverable:

1. Open Tableau Public/Cloud.
2. Connect to `data/dashboard/tableau_data/electricity_hourly.csv` and the model tables.
3. Follow `dashboard/tableau_build_guide.md`.
4. Validate each filter and interaction.
5. Publish and record the verified URL in this report and the README.

Until those steps are completed, the dashboard status is **data and build guide complete; workbook pending user authoring**.

---

# Chapter 9: Conclusion and Future Work

## 9.1 Summary of findings

The project built a reproducible pipeline for hourly electricity demand prediction on a 120-building ASHRAE subset. Three models were compared under a chronological split. HistGradientBoosting was selected by validation RMSE and achieved test MAE 33.20, RMSE 69.78, R² 0.970, and RMSLE 0.591. Random Forest had a lower MAE and RMSLE but a higher RMSE and lower R². The mean baseline confirmed that the learned models capture substantial structure.

The EDA produced five engineering insights: demand is right-skewed and building-dominated; occupancy schedules create a weekday/midday cycle; temperature response is site-specific; primary use and site dominate; and seen-building accuracy does not transfer to unseen buildings.

The unseen-building experiment is the most important caution. With 20% of buildings held out, R² fell to -0.555 and MAE rose to 150.51. The model is a seen-building monitoring/interpolation tool, not a validated predictor for new buildings.

## 9.2 Achievement of objectives

| Objective | Status | Evidence |
|---|---|---|
| Audit dataset | Complete | `reports/data_inventory.md` |
| Reproducible preprocessing | Complete for subset | `src/preprocessing.py`; `data/processed/` |
| Three models | Complete | `src/train_models.py`; `models/model_metadata.json` |
| Validation/test comparison | Complete for subset | `reports/tables/model_comparison.csv` |
| Error and generalization analysis | Complete for subset | `reports/tables/generalization_building_holdout.csv` |
| Two unseen cases | Complete | `reports/tables/unseen_case_predictions.csv` |
| Insights | Complete | `reports/engineering_insights.md` |
| Tableau-ready data | Complete | `data/dashboard/tableau_data/` |
| Tableau workbook | Pending user authoring | `dashboard/tableau_build_guide.md` |
| Streamlit supplementary app | Complete and smoke-tested | `dashboard/streamlit_app.py` |
| Technical report | Complete | This file |
| Full-data run | Pending user execution | Notebook and README |
| Dataset approval | Pending teacher | `reports/dataset_approval_proposal.md` |

## 9.3 Main limitations

The results are subset results. The model depends on building identity and does not generalize to unseen buildings. The test month is winter and has a different zero-reading distribution from training. Weather is site-level and incomplete, metadata are substantially missing, and target units are documented rather than independently verified. The Tableau workbook and full-data metrics are not yet produced.

## 9.4 Future work

1. Run the full-data pipeline in a high-memory Colab runtime and compare subset vs full metrics.
2. Add lagged and rolling building-level demand features with strict past-only construction.
3. Add degree-hour features and site-by-temperature interactions.
4. Incorporate holiday, academic-calendar, and occupancy proxies.
5. Model zero/non-zero readings separately (two-stage model).
6. Test time-series cross-validation across multiple months and years.
7. Evaluate calibration by building, site, and load band.
8. Build and validate the Tableau workbook and publish it.
9. Prototype an anomaly-detection threshold using building-specific residual distributions.
10. If sub-meter data become available, extend the analysis to end-use disaggregation and HVAC-specific benchmarking.

---

# References

Amasyali, K., & El-Gohary, N. M. (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81, 1192–1205.

ASHRAE. (2019). *Great Energy Predictor III*. Kaggle. https://www.kaggle.com/c/ashrae-energy-prediction

Kaggle dataset re-upload. (n.d.). *ASHRAE Great Energy Predictor III Dataset*. https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset

McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.

Miller, C., Arjunan, P., Kathirgamanathan, A., Fu, C., Roth, J., Park, J. Y., Balbach, C., Das, K., Atkinson, T., & Selvaraj, A. (2020). The ASHRAE Great Energy Predictor III competition: Overview and results. *Science and Technology for the Built Environment*, 26(10), 1427–1447.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Streamlit. (n.d.). *Streamlit documentation*. https://docs.streamlit.io/

Tableau. (n.d.). *Tableau Public/Cloud documentation*. https://help.tableau.com/

U.S. Energy Information Administration. (2018). *Commercial Buildings Energy Consumption Survey (CBECS)*. https://www.eia.gov/consumption/commercial/

---

# Appendix A: Reproducibility commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_loading
python -m src.inspect_data
python -m src.preprocessing
python -m src.train_models
python -m src.eda
python -m src.generalization
python -m src.predict
python -m src.build_dashboard_data

pytest -q
streamlit run dashboard/streamlit_app.py
```

The notebook `notebooks/ME446_ASHRAE_Complete_Analysis.ipynb` runs the same workflow and supports a full-data Colab mode. The seed is 42. Raw files are retained in `data/raw/`.

# Appendix B: Key output files

| Output | Path |
|---|---|
| Data audit | `reports/data_inventory.md` |
| Approval proposal | `reports/dataset_approval_proposal.md` |
| Model dataset | `data/processed/model_dataset.parquet` |
| Final model | `models/final_model.joblib` |
| Model metadata | `models/model_metadata.json` |
| Model comparison | `reports/tables/model_comparison.csv` |
| Generalization | `reports/tables/generalization_building_holdout.csv` |
| Unseen cases | `reports/tables/unseen_case_predictions.csv` |
| Feature importance | `reports/tables/feature_importance_histgradientboosting.csv` |
| EDA figures | `reports/figures/` |
| Tableau data | `data/dashboard/tableau_data/` |
| Tableau guide | `dashboard/tableau_build_guide.md` |
| Streamlit app | `dashboard/streamlit_app.py` |
| Presentation outline | `presentation/presentation_outline.md` |
| Presentation deck | `presentation/ME446_CEP_Presentation.pptx` |
| Poster content | `presentation/poster_content.md` |
| A1 poster | `presentation/ME446_CEP_Poster.pdf` |
| Poster source | `presentation/poster.tex` |
| Rubric checklist | `reports/rubric_checklist.md` |
