# Dataset Approval Proposal — ME-446 Complex Engineering Problem

**Status: AWAITING TEACHER APPROVAL.** Dataset uniqueness has not been confirmed by the course teacher.

| Item | Detail |
|---|---|
| University | NED University of Engineering and Technology, Karachi |
| Department | Mechanical Engineering |
| Course | ME-446 — AI & Internet of Things, Fall 2026 |
| Assessment | Complex Engineering Problem (CLO-2, PLO-2) |
| Group | _To be supplied_ |
| Students | _Names and roll numbers to be supplied_ |
| Instructor | _To be supplied_ |
| Submission date | 4 November 2026 |

## Proposed title

**AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications**

## Dataset

- **Kaggle identifier:** `sumit261124/ashrae-great-energy-predictor-iii-dataset`
- **URL:** https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset
- **Original source:** ASHRAE Great Energy Predictor III competition, hosted on Kaggle (2019).
- **Accessed:** 20 October 2026 via KaggleHub.
- **Files:** `train.csv` (20,216,100 rows), `test.csv` (41,697,600 rows, features only), `building_metadata.csv` (1,449 buildings), `weather_train.csv`, `weather_test.csv`, `sample_submission.csv`.
- **License/terms:** The files were obtained through Kaggle. The original competition data is governed by the Kaggle competition rules and the dataset's Kaggle terms; the re-upload does not state a separate open-data license. The data will be used for academic coursework and will not be redistributed beyond the course submission.

## Engineering application

Hourly building electricity demand is influenced by weather, building characteristics, and operating schedules. A model that predicts meter readings from these inputs supports building energy monitoring, demand planning, and screening for unusual consumption. This is a mechanical-engineering problem because HVAC and building services dominate electricity use in many commercial buildings, and weather-driven load behaviour is central to HVAC analysis.

## Objective

Develop and compare interpretable regression models that predict hourly **electricity** meter readings for buildings observed in 2016, using weather, calendar, and building metadata, and communicate the results through an interactive dashboard and report.

## Regression task

- **Target:** `meter_reading` (dataset energy units; the competition documentation defines the target as energy in kWh).
- **Filter:** `meter == 0` (electricity). Verified from the data: meter 0 has 12,060,910 rows across 1,413 buildings and is the dominant meter type. Other meters are not mixed into the target.
- **Inputs:** `building_id`, `site_id`, `primary_use`, `square_feet`, `year_built`, `floor_count`, `air_temperature`, `dew_temperature`, `wind_speed`, `cloud_coverage`, `precip_depth_1_hr`, `sea_level_pressure`, plus hour, day of week, month, weekend indicator, and cyclical hour/month encodings.
- **Excluded:** any target-derived variable, future readings, and test labels (not provided).

## Candidate models

1. Linear Regression — interpretable baseline.
2. Random Forest Regressor — nonlinear interactions and feature importance.
3. HistGradientBoostingRegressor — efficient boosting for tabular data.

## Planned preprocessing

- Preserve raw files; build typed Parquet intermediates.
- Audit missingness: `year_built` is missing for 53.4% of buildings and `floor_count` for 75.5%, so no arbitrary imputation; missing values are handled natively by the models or flagged.
- Weather gaps: limited forward-fill within site (no future information); highly incomplete `cloud_coverage` and `precip_depth_1_hr` are retained with missing indicators rather than treated as zero.
- Duplicate business keys: none found.
- Outliers: investigate but do not automatically delete legitimate peaks.
- Time-aware split: train January–September 2016, validate October–November 2016, test December 2016.
- All learned preprocessing fitted on training data only within scikit-learn pipelines.
- Local development uses a documented site-stratified subset because the machine has limited RAM; the full-data run is prepared for Google Colab and will be labelled separately.

## Evaluation strategy

- Metrics: MAE, RMSE, R², and RMSLE where valid.
- Validation and test metrics reported separately on identical held-out observations.
- Actual-versus-predicted, residual, and subgroup error analysis.
- At least two previously unseen held-out cases demonstrated with actual and predicted readings.
- Model selection based on measured performance, runtime, interpretability, and engineering usefulness.

## Expected dashboard

- **Tableau Cloud/Public:** Energy Overview, Weather & Building Analysis, and Model Performance dashboards, built from exported CSV datasets with a documented specification.
- **Streamlit:** supplementary interactive app for local model exploration and prediction.

## Suitability for ME-446

The dataset is large, time-stamped, engineering-relevant, and requires genuine preprocessing decisions (missing metadata, weather gaps, zero readings, temporal leakage). It supports regression, feature engineering, model comparison, and dashboard communication while remaining grounded in building energy and HVAC practice.

## Request

We request approval of this dataset and confirmation of its uniqueness for ME-446. Until approval is received, this proposal remains marked **awaiting approval**, and no claim of approval or uniqueness is made.
