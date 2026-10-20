# Presentation Outline

**Project:** AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications  
**Course:** ME-446 AI & Internet of Things, Fall 2026  
**Target duration:** 10–12 minutes + 2–3 minutes questions  
**Group / students / instructor:** placeholders until supplied

**Core message:** A gradient-boosted model predicts hourly electricity demand accurately for buildings it has seen, but it fails on unseen buildings. The project is a monitoring and interpolation tool, not a validated new-building predictor.

**Visual identity:** white background, navy titles (#14213D), teal accents (#2A9D8F), orange highlight (#E76F51), grey body text. Use one figure or one table per slide. Avoid dense paragraphs.

---

## Slide 1 — Title

**Content**

- AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications
- ME-446 AI & Internet of Things — Complex Engineering Problem
- NED University of Engineering and Technology, Department of Mechanical Engineering
- Fall 2026
- Group: _[Group No.]_
- Students: _[Name — Roll No.]_
- Instructor: _[Instructor Name]_
- Submission: 4 November 2026

**Speaker note:** Introduce the team and state the one-sentence takeaway: the model is accurate for known buildings but not for new ones.

---

## Slide 2 — Engineering problem and motivation

**Content**

- Buildings and HVAC consume a large share of electricity.
- A normal-demand model can support monitoring, demand planning, and anomaly screening.
- The target is hourly electricity `meter_reading` for meter 0.
- The engineering question is not only “how accurate?” but “accurate for whom, and when?”
- Key tension: high aggregate accuracy can hide poor transfer to new buildings.

**Speaker note:** Explain why a mechanical engineer cares: peak demand, HVAC scheduling, and avoiding false alarms. Do not claim savings.

---

## Slide 3 — Dataset and scope

**Content**

- ASHRAE Great Energy Predictor III (Kaggle, accessed 20 October 2026).
- Full data: 20,216,100 train rows; 1,449 buildings; 1,413 electricity buildings; 12,060,910 electricity rows.
- Local subset: 120 buildings, 16 sites, 16 primary uses, 1,030,967 rows.
- Split: Jan–Sep train (769,122), Oct–Nov validation (174,537), Dec test (87,308).
- Units: dataset energy units (documented as kWh), not independently verified.
- Approval status: awaiting teacher approval; uniqueness not claimed.

**Visual:** Small dataset table or three KPI cards.

**Speaker note:** Be explicit that all local results are subset results. The subset preserves all sites and uses but not the full building mix.

---

## Slide 4 — Preprocessing and leakage control

**Content**

- Raw CSVs retained unchanged; typed Parquet intermediates.
- Electricity only (`meter == 0`); zero readings retained.
- Metadata missingness: `year_built` 51.76%, `floor_count` 70.90% in subset; no invented values.
- Weather: past-only forward fill, 3-hour limit; negative precipitation treated as missing; pressure range check; missing flags.
- Features: building, size, calendar, cyclical hour/month, weather, missing flags — 24 features.
- Chronological split; no test-driven model selection; pipeline-fitted preprocessing.

**Speaker note:** Emphasize that weather fill uses only the past, and that the test month is never used to choose the model.

---

## Slide 5 — EDA: demand is right-skewed and building-dominated

**Content**

- Median 59.05; mean 188.78; p99 2,454; maximum 41,552.
- Weekday mean 195.44 vs weekend 172.24; peak 13:00, trough 03:00.
- Floor area log-correlation 0.750; `square_feet` is the top permutation feature.
- Education median 157.2; Religious worship 3.75.
- Implication: benchmark within building/use groups; normalize by size.

**Visual:** `reports/figures/reading_distribution.png` or `building_size_relationship.png`.

**Speaker note:** The mean is not the typical building. A single global average is a poor benchmark.

---

## Slide 6 — EDA: occupancy and seasonal cycles

**Content**

- Weekday/weekend ratio 1.135.
- Peak 13:00; low 03:00.
- Monthly mean: April 166.60 low; August 215.14 high.
- Daily mean range: 146.87 (15 May) to 249.89 (29 Aug).
- Implication: target HVAC setback and scheduling studies at low-occupancy hours.

**Visual:** `reports/figures/temporal_patterns.png`.

**Speaker note:** The cycle is consistent with occupancy and cooling, but the data have no occupancy labels.

---

## Slide 7 — EDA: temperature response is site-specific

**Content**

- Pooled correlation −0.074 looks weak.
- Site-level correlation ranges from −0.134 (site 10) to +0.406 (site 11).
- Site 11 mean reading rises from 265.74 below 10 °C to 377.17 above 25 °C.
- Sites 6, 7, 10 are negative; sites 0, 11, 13 are positive.
- Implication: one global linear temperature coefficient is inadequate; use site interactions or degree hours.

**Visual:** `reports/figures/temperature_response_by_site.png`.

**Speaker note:** This is a classic pooled-correlation trap. Opposite heating and cooling responses cancel.

---

## Slide 8 — Models and validation design

**Content**

- Linear Regression: interpretable baseline; one-hot, median imputation, scaling.
- Random Forest: 80 trees, depth 16, 250,000-row training sample.
- HistGradientBoosting: 200 iterations, learning rate 0.08, 31 leaves, all 769,122 training rows.
- Mean baseline for reference.
- Selection rule: lowest validation RMSE; test used once.
- Metrics: MAE, RMSE, R², RMSLE.

**Speaker note:** Explain why RMSE: large peak errors are costly for demand planning. Mention the models were compared on identical split rows.

---

## Slide 9 — Results: HGB selected, RF trade-off

**Content**

| Model | Val RMSE | Test MAE | Test RMSE | Test R² | Test RMSLE |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 411.67 | 209.57 | 406.05 | −0.001 | 1.920 |
| Linear Regression | 223.03 | 61.39 | 154.51 | 0.855 | 0.942 |
| Random Forest | 166.92 | **32.05** | 78.90 | 0.962 | **0.448** |
| HistGradientBoosting | **165.74** | 33.20 | **69.78** | **0.970** | 0.591 |

- HGB wins validation RMSE and test RMSE/R².
- RF wins MAE and RMSLE; this trade-off is reported honestly.
- HGB is also about 4.3× faster to fit than RF.

**Visual:** Table plus `model_comparison.png`.

**Speaker note:** Do not say HGB dominates every metric. The choice depends on whether peak RMSE or typical MAE is the operational priority.

---

## Slide 10 — Where the model fails

**Content**

- Seen-building test: MAE 33.20, R² 0.970.
- Unseen-building holdout (20% of buildings): MAE 150.51, RMSE 321.33, R² **−0.555**, RMSLE 1.525.
- Unseen predictions are biased high: mean 242.49 vs actual 176.99.
- Error grows with load; low-load bands have large relative error.
- December has 126 zeros vs 44,796 in training — distribution shift.

**Visual:** `reports/figures/unseen_buildings_actual_vs_predicted.png` or a two-bar R² comparison.

**Speaker note:** This is the most important slide. The high R² is building-specific interpolation, not a universal physical model.

---

## Slide 11 — Unseen-case demonstrations

**Content**

| Case | Building | Time | Actual | Predicted | Error |
|---|---|---:|---:|---:|---:|
| High demand | 1088 | 19 Dec 15:00 | 3,427.60 | 3,417.33 | 0.30% |
| Typical demand | 386 | 9 Dec 06:00 | 41.16 | 41.16036 | 0.0009% |

- Building 1088: 287,594 ft² education building at −10.6 °C; physically plausible winter load.
- Near-exact second case is not proof of generalization; trees can match a leaf closely.
- Feature consistency between training SQL and Python inference passed.

**Speaker note:** Explain that these are held-out rows, but the building is still known to the main model. The holdout experiment is the real generalization test.

---

## Slide 12 — Dashboard and deliverables

**Content**

- Tableau-ready data: 1,030,967-row electricity extract; 349,232 model prediction rows.
- Three dashboard designs: Energy Overview, Weather & Building, Model Performance.
- Detailed Tableau build guide; workbook pending user authoring.
- Supplementary Streamlit app with four pages, smoke-tested.
- Reproducible notebook, tests (22 passed), audit, report, insights.

**Visual:** Screenshot of Streamlit Energy Overview or dashboard architecture diagram.

**Speaker note:** Be explicit: no Tableau workbook has been published. The data and instructions are complete; authoring is the remaining step.

---

## Slide 13 — Engineering insights and recommendations

**Content**

1. Demand is right-skewed and building-dominated — benchmark within peer groups.
2. Weekday/midday cycle — target setback and scheduling.
3. Temperature response is site-specific — add degree hours/site interactions.
4. Primary use and site dominate — segment models and dashboards.
5. Seen-building accuracy does not transfer — calibrate before new-building use.

**Recommendations**

- Add past-only lagged readings and rolling statistics.
- Model zero/non-zero readings separately.
- Calibrate residuals by building and load band.
- Validate on a future year and a building holdout.
- Do not claim savings or efficiency from predictions alone.

**Speaker note:** Connect each insight to a concrete action. Avoid claiming causal savings.

---

## Slide 14 — Limitations and conclusion

**Content**

- Local results use 120 of 1,413 electricity buildings.
- Main model depends on `building_id`; unseen-building R² is negative.
- Single year; winter-only test; zero-reading shift; site-level weather; missing metadata.
- No Tableau workbook yet; full-data Colab run pending; approval pending.
- Conclusion: useful seen-building monitoring tool; not a validated new-building predictor.
- Next: full-data run, history features, degree hours, two-stage zero model, Tableau workbook.

**Speaker note:** End with the conditional conclusion. Invite questions about generalization and operational use.

---

## Slide 15 — References and questions

**Content**

- ASHRAE Great Energy Predictor III, Kaggle (2019).
- Miller et al. (2020), ASHRAE competition overview.
- Amasyali & El-Gohary (2018), data-driven building energy prediction review.
- scikit-learn, pandas, Tableau, Streamlit documentation.
- Project repository paths: `reports/technical_report.md`, `reports/engineering_insights.md`.

**Speaker note:** Thank the audience; invite questions. Keep the final slide visible during Q&A.

---

## Timing plan

| Slide | Topic | Time |
|---|---|---:|
| 1 | Title | 0:30 |
| 2 | Problem and motivation | 0:45 |
| 3 | Dataset and scope | 0:50 |
| 4 | Preprocessing and leakage | 0:50 |
| 5–7 | EDA and insights | 2:30 |
| 8 | Models and validation | 1:00 |
| 9 | Results | 1:15 |
| 10 | Failure and generalization | 1:15 |
| 11 | Unseen cases | 0:45 |
| 12 | Dashboard | 0:50 |
| 13 | Recommendations | 0:50 |
| 14 | Limitations and conclusion | 0:40 |
| 15 | References/questions | 0:20 |
| **Total** | | **~12:00** |

## Anticipated questions

**Q: Why is R² negative for unseen buildings?**  
A: The model is worse than predicting the unseen group's mean. It learned building-specific levels; without `building_id` history, it transfers poorly. The predicted mean is also biased high (242.49 vs 176.99).

**Q: Why select HGB if RF has lower MAE?**  
A: The pre-declared rule is validation RMSE, which penalizes large peak errors and selected HGB. RF wins MAE/RMSLE; the trade-off is reported. For a typical-hour objective, RF would be defensible.

**Q: Are the units kWh?**  
A: The competition documentation describes energy in kWh, but the CSV does not state units. We label “dataset units (documented as kWh)” and do not claim independent verification.

**Q: Does the dashboard exist?**  
A: Tableau-ready data and a detailed build guide exist; the actual Tableau workbook has not been authored or published. The Streamlit app is supplementary and was smoke-tested.

**Q: Why not use 2017–2018 test data?**  
A: Kaggle provides features only, no labels, so supervised evaluation is impossible. The local test is December 2016.
