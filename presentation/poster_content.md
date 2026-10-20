# Poster Content and Layout Specification

**Format:** A1 portrait (594 × 841 mm), three columns, 24 mm outer margin, 10 mm gutter.  
**Style:** white background; navy header (#14213D); teal accents (#2A9D8F); orange highlights (#E76F51); dark grey body (#333333). Use Arial/Calibri or Helvetica. Keep body text at least 24 pt when printed at A1.  
**Status:** generated as `presentation/ME446_CEP_Poster.pdf` from `presentation/poster.tex`; group/student/instructor placeholders must still be replaced.

---

## Header

**Title:** AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications

**Subtitle:** ME-446 AI & Internet of Things — Complex Engineering Problem

**Institution:** NED University of Engineering and Technology, Department of Mechanical Engineering

**Authors:** _[Student Name — Roll No.]_, _[Student Name — Roll No.]_  
**Group:** _[Group No.]_ | **Instructor:** _[Instructor Name]_ | **Fall 2026**

---

## Column 1

### 1. Problem and motivation

Buildings and HVAC systems account for a major share of electricity use. A model of normal hourly demand can support monitoring, demand planning, and anomaly screening. The engineering question is not only “how accurate is the model?” but “accurate for which buildings, and under what conditions?”

**Target:** hourly electricity `meter_reading` for meter 0.  
**Inputs:** building metadata, calendar, weather, and building identity.  
**Unit caveat:** dataset energy units (documented as kWh); not independently verified; not power, efficiency, or savings.

### 2. Objectives

- Audit the ASHRAE dataset and document data-quality decisions.
- Build a leakage-safe preprocessing and feature pipeline.
- Compare Linear Regression, Random Forest, and HistGradientBoosting.
- Select a model on validation data and report test metrics separately.
- Quantify unseen-building generalization and demonstrate two unseen cases.
- Produce engineering insights, Tableau-ready data, and a build guide.

### 3. Dataset and scope

| Item | Value |
|---|---|
| Source | ASHRAE Great Energy Predictor III (Kaggle) |
| Full train rows | 20,216,100 |
| Electricity rows | 12,060,910 |
| Electricity buildings | 1,413 |
| Local subset | 120 buildings, 16 sites, 16 uses |
| Local rows | 1,030,967 |
| Split | Jan–Sep / Oct–Nov / Dec 2016 |

**Local scope warning:** all reported metrics are subset metrics, not full-dataset metrics. Dataset approval is pending.

---

## Column 2

### 4. Methodology workflow

**Raw CSVs → typed Parquet → electricity filter → metadata/weather joins → feature engineering → chronological split → pipelines → validation → test → error/generalization analysis → dashboard extracts.**

**Preprocessing decisions**

- Raw files retained unchanged.
- Electricity only (`meter == 0`); zero readings retained.
- Metadata missingness retained: `year_built` 51.76%, `floor_count` 70.90% in subset.
- Weather: past-only forward fill, 3-hour limit; negative precipitation treated as missing; pressure range 800–1100 hPa; missing flags.
- 24 features: identity, size, calendar, cyclical hour/month, weather, missing flags.
- No target lags, no future fill, no test-driven selection.

**Models**

| Model | Key configuration |
|---|---|
| Linear Regression | one-hot, median imputation, scaling |
| Random Forest | 80 trees, depth 16, 250k sample |
| HistGradientBoosting | 200 iterations, lr 0.08, 31 leaves, all train rows |

### 5. Results

| Model | Test MAE | Test RMSE | Test R² | Test RMSLE |
|---|---:|---:|---:|---:|
| Mean baseline | 209.57 | 406.05 | −0.001 | 1.920 |
| Linear Regression | 61.39 | 154.51 | 0.855 | 0.942 |
| Random Forest | **32.05** | 78.90 | 0.962 | **0.448** |
| HistGradientBoosting | 33.20 | **69.78** | **0.970** | 0.591 |

**Selected:** HistGradientBoosting, by the pre-declared lowest-validation-RMSE rule (165.74 vs 166.92). Random Forest has lower MAE and RMSLE; the trade-off is reported.

### 6. Key figures (place as three panels)

1. `reports/figures/reading_distribution.png` — right-skewed demand.
2. `reports/figures/temporal_patterns.png` — weekday/midday cycle.
3. `reports/figures/temperature_response_by_site.png` — site-specific temperature response.

---

## Column 3

### 7. Critical finding: unseen buildings

| Group | MAE | RMSE | R² | RMSLE |
|---|---:|---:|---:|---:|
| Seen buildings, test | 33.86 | 74.15 | 0.971 | 0.595 |
| Unseen buildings, test | 150.51 | 321.33 | **−0.555** | 1.525 |

The main model uses `building_id`. Its high seen-building R² is largely building-specific interpolation, not a universal physical model. The unseen-building predicted mean is biased high (242.49 vs actual 176.99). Do not deploy to new buildings without calibration or richer descriptors.

**Unseen-case demonstrations**

| Case | Actual | Predicted | Error |
|---|---:|---:|---:|
| Building 1088, 19 Dec 15:00 | 3,427.60 | 3,417.33 | 0.30% |
| Building 386, 9 Dec 06:00 | 41.16 | 41.16036 | 0.0009% |

### 8. Five engineering insights

1. **Demand is right-skewed and building-dominated.** Median 59.05 vs mean 188.78; floor-area log-correlation 0.750. Benchmark within use/size groups.
2. **Occupancy schedules create a weekday/midday cycle.** Weekday mean 195.44 vs weekend 172.24; peak 13:00, trough 03:00. Target setback studies.
3. **Temperature response is site-specific.** Pooled r = −0.074, but site r ranges −0.134 to +0.406. Use degree hours or site interactions.
4. **Primary use and site dominate.** Median reading 3.75–157.20 by use; site means 10.24–1,156.11. Segment reporting.
5. **Seen-building accuracy does not transfer.** R² 0.970 seen vs −0.555 unseen. Recalibrate before new-building use.

### 9. Recommendations

- Add past-only lagged/rolling demand features.
- Add heating/cooling degree hours and site-by-temperature interactions.
- Model zero/non-zero readings separately.
- Calibrate residuals by building and load band.
- Validate on a future year and a building holdout.
- Build and verify the Tableau workbook before claiming a dashboard deliverable.

### 10. Conclusion and limitations

The model is a useful **seen-building monitoring and interpolation tool**, not a validated new-building predictor. Results are from a 120-building subset, one year, a winter test month, site-level weather, and substantially missing metadata. The Tableau workbook and full-data Colab metrics remain pending. No efficiency or savings claim is made.

### 11. References

- ASHRAE. (2019). *Great Energy Predictor III*. Kaggle.
- Miller, C., et al. (2020). The ASHRAE Great Energy Predictor III competition: Overview and results. *Science and Technology for the Built Environment*, 26(10), 1427–1447.
- Amasyali, K., & El-Gohary, N. M. (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81, 1192–1205.
- scikit-learn, pandas, Tableau, and Streamlit documentation.

---

## Footer

**Status:** Local subset analysis complete. Tableau workbook pending. Full-data run pending. Dataset approval pending.  
**Code and report:** `README.md`, `reports/technical_report.md`, `reports/engineering_insights.md`.

## Print/layout notes

- Place the university logo at top left only if an official file is supplied; do not fabricate one.
- Keep a clear visual hierarchy: title 80–96 pt, section headings 40–44 pt, body 24–28 pt.
- Use the three figures at equal width in a single row if space allows; otherwise stack them with short captions.
- Put the unseen-building warning in an orange-bordered box; it is the most important engineering message.
- Do not add invented photographs of buildings or fabricated dashboard screenshots. Use actual figures or a clearly labelled architecture diagram.
