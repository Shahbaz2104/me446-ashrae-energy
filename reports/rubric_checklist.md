# Final Assessment Rubric Checklist

**Purpose:** quality gate, not a self-award of marks. Each item lists the evidence that exists and the work that remains. The project does not claim teacher approval, a Tableau workbook, or full-data metrics.

**Current status:** local subset analysis complete; notebook executed in artifact-reuse mode; Tableau workbook pending; presentation deck and A1 poster generated; full-data Colab run pending.

---

## Criterion 1 — Dataset Selection and Engineering Problem Formulation

| Requirement | Status | Evidence / gap |
|---|---|---|
| Mechanical engineering relevance is explicit | Met | HVAC/building-energy motivation in `reports/technical_report.md` Ch. 1 and `reports/dataset_approval_proposal.md` |
| Dataset source documented | Met | Kaggle identifier, URL, access date, original competition, raw-file audit |
| Features and target clearly defined | Met | `src/config.py`; report Ch. 2; 24 features; `meter_reading` for meter 0 |
| Objectives justified | Met | Five objectives and research questions in report Ch. 1–2 |
| Teacher approval not misrepresented | Met | Proposal repeatedly marked **AWAITING TEACHER APPROVAL**; no approval claimed |
| Dataset uniqueness confirmed | **Pending** | Must be confirmed by the teacher; not asserted |

**Verdict:** Evidence complete except external approval. Do not claim dataset uniqueness until the teacher confirms it.

---

## Criterion 2 — Data Preprocessing and Feature Preparation

| Requirement | Status | Evidence / gap |
|---|---|---|
| Raw data preserved | Met | `data/raw/` unchanged; hardlinked from KaggleHub cache |
| Data quality audited | Met | `reports/data_inventory.md`; row counts, duplicates, meters, missingness, coverage |
| Missing values addressed | Met | Median imputation + indicators for linear model; native missing handling for trees; weather flags |
| Duplicates addressed | Met | Zero business-key duplicates verified; no inappropriate row deletion |
| Outliers handled defensibly | Met | Legitimate peaks retained; negatives clipped only at prediction; zero ambiguity documented |
| Categories handled | Met | One-hot for linear; ordinal for trees; 16 primary-use levels |
| Reproducible transformations | Met | `src/data_loading.py`, `src/preprocessing.py`, `src/feature_engineering.py`; seed 42 |
| Leakage avoided | Met | Chronological split; past-only weather fill; pipeline-fitted preprocessing; test untouched during selection |
| Subset documented | Met | 120 buildings, 16 sites, 16 uses; `data/processed/subset_selection.json`; site/use stratification |

**Verdict:** Met for the local subset. Full-data preprocessing is implemented but not yet executed.

---

## Criterion 3 — Visualization, Dashboard, and Insights

| Requirement | Status | Evidence / gap |
|---|---|---|
| At least three meaningful visualizations | Met | 13 figures in `reports/figures/` (distribution, temporal, temperature, site, use, size, correlation, model, residuals, importance, unseen buildings) |
| Dashboard is interactive or fully documented | **Partial** | Tableau-ready CSVs + detailed specification/build guide; Tableau workbook **not authored/published** |
| Supplementary Streamlit app | Met | `dashboard/streamlit_app.py`; all four pages smoke-tested with `AppTest` |
| KPIs appropriate | Met | Definitions and unit caveats in `data/dashboard/README.md`, report Ch. 8 |
| Filters appropriate | Met in specification | Date, site, primary use, building, model, reading band; actual workbook behavior still requires validation |
| Three to five evidence-backed insights | Met | Five insights in `reports/engineering_insights.md`; each with evidence, interpretation, action, limitation |

**Verdict:** Visualization and insights are met. Dashboard is partial until the Tableau workbook is built and validated by the user. The Streamlit app does not replace Tableau.

---

## Criterion 4 — AI/ML Implementation

| Requirement | Status | Evidence / gap |
|---|---|---|
| At least three suitable models | Met | Linear Regression, Random Forest, HistGradientBoosting; mean baseline also reported |
| Model choices justified | Met | Report Ch. 5: linearity, nonlinearity, efficiency, missing-value support |
| Pipeline reproducible | Met | `src/train_models.py`; fixed seed 42; scikit-learn pipelines; saved metadata |
| Training executes correctly | Met | Local training completed; metrics in `reports/tables/model_comparison.csv`; 22 tests pass |
| Inference executes correctly | Met | `ElectricityPredictor`; feature-consistency assertion; two unseen cases; prediction test |

**Verdict:** Met for the local subset.

---

## Criterion 5 — Validation, Comparison, and Selection

| Requirement | Status | Evidence / gap |
|---|---|---|
| Training/testing strategy justified | Met | Chronological Jan–Sep / Oct–Nov / Dec split; deployment rationale in report Ch. 3 |
| Temporal leakage avoided | Met | No shuffle; no target lags; past-only weather fill; test used once |
| Metrics appropriate | Met | MAE, RMSE, R², RMSLE; units and metric behavior explained |
| Models compared fairly | Met | Same split and held-out rows; fit/predict times; validation and test reported separately |
| Errors and generalization discussed | Met | Error by band/site; unseen-building holdout; negative R² reported |
| At least two unseen cases demonstrated | Met | Building 1088 and 386 in `reports/tables/unseen_case_predictions.csv` |
| Selection rule followed | Met | Lowest validation RMSE selected HGB; RF trade-off explicitly discussed |
| Full-data validation | **Pending** | Colab workflow prepared; no full-data metrics claimed |

**Verdict:** Met for the subset. Full-data comparison remains pending.

---

## Criterion 6 — Engineering Interpretation

| Requirement | Status | Evidence / gap |
|---|---|---|
| Feature effects and model behavior discussed | Met | Permutation importance; site-specific temperature; building size; report Ch. 7 |
| Predictions checked for physical plausibility | Met | High-demand winter education case; site/load patterns; limitations noted |
| Limitations and error causes identified | Met | Subset, seen-building dependence, seasonal/zero shift, site weather, metadata, units, no occupancy |
| Practical recommendations supported by evidence | Met | Ten recommendations in report Ch. 7.6; five insight actions |
| Consequences of incorrect predictions discussed | Met | Peak underprediction, overprediction, false alarms, unseen-building risk |
| No causal overclaim | Met | Correlations/importance labelled associational; no savings claim |

**Verdict:** Met for the available evidence.

---

## Criterion 7 — Reproducibility and Communication

| Requirement | Status | Evidence / gap |
|---|---|---|
| Notebook can be reproduced | **Partial** | `notebooks/ME446_ASHRAE_Complete_Analysis.ipynb` executed top-to-bottom in reuse mode with no errors; full fresh execution not run because it would retrain/rebuild |
| Data and code organized | Met | `data/`, `src/`, `models/`, `reports/`, `dashboard/`, `tests/`, `presentation/` |
| Technical report complete | Met | `reports/technical_report.md` (all suggested chapters, references, appendices) |
| Dashboard materials prepared | **Partial** | Data + specification + guide + Streamlit; Tableau workbook pending |
| Presentation materials prepared | **Met** | 15-slide `.pptx`, A1 `.pdf`/`.tex`, slide outline, and poster content; replace placeholders |
| References, assumptions, limitations documented | Met | Report references; assumptions Ch. 2; limitations Ch. 7 and README |
| Results traceable to computations | Met | CSV/JSON/figure outputs and saved metadata; notebook displays them |
| Dependencies documented | Met | `requirements.txt`; README installation |
| Tests exist and pass | Met | 22 tests; `pytest -q` passed |

**Verdict:** Partial only because the notebook has not been freshly executed in a full-data environment and the Tableau workbook is pending. Presentation and poster artifacts are complete as templates; placeholders and teacher approval remain user actions. Local execution is reproducible from the documented artifacts.

---

## Required Final Deliverables Checklist

| # | Deliverable | Status | Path |
|---:|---|---|---|
| 1 | Reproducible notebook | Complete (reuse execution) | `notebooks/ME446_ASHRAE_Complete_Analysis.ipynb` |
| 2 | Inspection/cleaning scripts | Complete | `src/inspect_data.py`, `src/preprocessing.py` |
| 3 | Raw dataset retained | Complete | `data/raw/` |
| 4 | Cleaned/transformed dataset | Complete | `data/processed/model_dataset.parquet` |
| 5 | Data-quality audit report | Complete | `reports/data_inventory.md` |
| 6 | At least three visualizations | Complete | `reports/figures/` |
| 7 | Three to five insights | Complete | `reports/engineering_insights.md` |
| 8 | Three trained/evaluated models | Complete | `src/train_models.py`; `models/` |
| 9 | Model comparison table | Complete | `reports/tables/model_comparison.csv` |
| 10 | Validation and test metrics | Complete (subset) | `reports/tables/model_comparison.csv` |
| 11 | Actual-vs-predicted and error plots | Complete | `reports/figures/actual_vs_predicted.png`, `residual_analysis.png` |
| 12 | Two unseen predictions | Complete | `reports/tables/unseen_case_predictions.csv` |
| 13 | Saved model and inference function | Complete | `models/final_model.joblib`, `src/predict.py` |
| 14 | Tableau-ready datasets | Complete | `data/dashboard/tableau_data/` |
| 15 | Tableau workbook | **Pending** | User must author/validate in Tableau |
| 16 | Tableau construction guide | Complete | `dashboard/tableau_build_guide.md` |
| 17 | Technical report | Complete | `reports/technical_report.md` |
| 18 | Dataset approval proposal | Complete (pending approval) | `reports/dataset_approval_proposal.md` |
| 19 | Presentation content/file | Complete (template) | `presentation/ME446_CEP_Presentation.pptx`; `presentation/presentation_outline.md` |
| 20 | Poster content/file | Complete (template) | `presentation/ME446_CEP_Poster.pdf`; `presentation/poster.tex`; `presentation/poster_content.md` |
| 21 | README and dependency file | Complete | `README.md`, `requirements.txt` |
| 22 | Tests | Complete | `tests/`; 22 passed |
| 23 | Reproducibility and limitations documentation | Complete | `README.md`, report Ch. 7, this checklist |

---

## Final Quality Gate

Before submission:

- [ ] Teacher approval/uniqueness confirmed and recorded.
- [ ] State clearly that reported metrics are subset metrics.
- [ ] Do not claim a Tableau workbook until it is published and verified.
- [x] Complete presentation and poster content (templates generated; replace placeholders).
- [ ] Optionally run the full-data notebook in Colab and add a clearly labelled full-data section.
- [x] Re-run `pytest -q` (22 passed).
- [ ] Re-run the notebook in a fresh environment, or document the reuse-mode validation honestly.
- [x] Verify all file paths and links in the report (45 project paths checked).
- [ ] Replace group/student/instructor placeholders.
- [ ] Confirm no credentials or personal information are committed.

**Marks are not self-assigned in this checklist.** The evidence above is provided for the instructor's assessment.
