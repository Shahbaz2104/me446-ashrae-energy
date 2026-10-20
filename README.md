# ME-446 Complex Engineering Problem: ASHRAE Building Energy Prediction

**Course:** ME-446 AI & Internet of Things, Fall 2026  
**Department:** Mechanical Engineering, NED University of Engineering and Technology  
**Project:** AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications  
**Submission date:** 4 November 2026  
**Status:** Local subset analysis complete; full-data Colab run pending. Dataset approval is pending.

## 1. Project objective

The project predicts hourly **electricity meter readings** for commercial buildings from weather, calendar, building metadata, and building identity. It compares an interpretable linear model, a Random Forest, and HistGradientBoosting, then communicates the results through a technical report, Tableau-ready datasets, a Tableau build guide, and a supplementary Streamlit application.

The engineering motivation is building energy monitoring and HVAC demand analysis. A model that predicts normal hourly demand can support benchmarking, demand planning, and screening for unusual consumption. It is not an efficiency rating, a savings measurement, or a substitute for metered verification.

## 2. Dataset

- **Kaggle dataset:** `sumit261124/ashrae-great-energy-predictor-iii-dataset`
- **URL:** https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset
- **Original source:** ASHRAE Great Energy Predictor III competition (2019)
- **Accessed:** 20 October 2026
- **Raw files retained unchanged in:** `data/raw/`

Key audit facts:

| Item | Value |
|---|---|
| Training observations | 20,216,100 |
| Buildings | 1,449 |
| Training period | 2016-01-01 to 2016-12-31 |
| Electricity (meter 0) rows | 12,060,910 |
| Electricity buildings | 1,413 |
| Electricity zero readings | 4.40% |
| Test features rows | 41,697,600 (2017–2018, no labels) |
| Duplicate business keys | 0 |

`meter_reading` units are not stated in the downloadable CSV. The competition documentation describes the target as energy in kWh; this project therefore uses the label **dataset units (documented as kWh)** and does not claim independent unit verification. Summed readings are not instantaneous power, efficiency, or savings.

## 3. Scope and evaluation design

- **Target:** `meter_reading`
- **Filter:** `meter == 0` electricity only
- **Local modeling scope:** documented site-stratified subset of **120 buildings**, all 16 sites and 16 primary-use categories, **1,030,967 hourly rows**
- **Split:** chronological, never shuffled
  - Train: January–September 2016 — 769,122 rows
  - Validation: October–November 2016 — 174,537 rows
  - Test: December 2016 — 87,308 rows
- **Selection rule:** lowest validation RMSE; the test period is used only once after selection
- **Metrics:** MAE, RMSE, R², and RMSLE
- **Leakage controls:** time-ordered split, past-only weather forward fill (limit 3 hours), no target-derived features, all learned preprocessing fitted inside training pipelines

The main task predicts readings for **buildings already observed in training**. A separate building-holdout experiment tests generalization to unseen buildings and is reported as a limitation.

## 4. Results summary (local subset only)

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Test R² | Test RMSLE |
|---|---:|---:|---:|---:|---:|---:|
| Mean baseline | 205.21 | 411.67 | 209.57 | 406.05 | -0.001 | 1.920 |
| Linear Regression | 68.40 | 223.03 | 61.39 | 154.51 | 0.855 | 0.942 |
| Random Forest | 29.35 | 166.92 | 32.05 | 78.90 | 0.962 | 0.448 |
| HistGradientBoosting | 32.66 | **165.74** | 33.20 | **69.78** | **0.970** | 0.591 |

HistGradientBoosting was selected by the pre-declared validation-RMSE rule. Random Forest had a lower validation MAE and better RMSLE; this trade-off is discussed in the report.

**Critical limitation:** in a 20% building-holdout experiment, the selected model had MAE 150.51 and R² **-0.555** on unseen buildings, versus R² 0.971 on seen buildings. The model should not be deployed on new buildings without recalibration or additional building-level information.

All numbers above are subset results. They are not full-dataset results.

## 5. Repository layout

```text
AI IOT CEP/
├── data/
│   ├── raw/                    # original CSVs, unchanged
│   ├── interim/                # typed Parquet, filled weather
│   ├── processed/              # model dataset, subset, unseen cases
│   └── dashboard/tableau_data/ # dashboard-ready CSV extracts
├── dashboard/
│   ├── streamlit_app.py
│   ├── dashboard_specification.md
│   └── tableau_build_guide.md
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── ME446_ASHRAE_Complete_Analysis.ipynb
├── reports/
│   ├── data_inventory.md
│   ├── dataset_approval_proposal.md
│   ├── engineering_insights.md
│   ├── technical_report.md
│   ├── rubric_checklist.md
│   ├── figures/
│   └── tables/
├── src/
├── tests/
├── presentation/
├── requirements.txt
└── README.md
```

## 6. Installation

Python 3.12 was used. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For notebook editing locally, install a notebook front end separately if it is not already available:

```bash
pip install "jupyterlab>=4,<5"
```

## 7. Dataset setup

The raw files are already present in `data/raw/` on the development machine. To reproduce the download:

```python
import os
import shutil
from pathlib import Path
import kagglehub

os.environ["KAGGLE_API_TOKEN"] = "YOUR_TOKEN"  # or use ~/.kaggle/kaggle.json
cache = Path(kagglehub.dataset_download(
    "sumit261124/ashrae-great-energy-predictor-iii-dataset"
))
for name in [
    "train.csv", "test.csv", "building_metadata.csv",
    "weather_train.csv", "weather_test.csv", "sample_submission.csv",
]:
    shutil.copy2(cache / name, Path("data/raw") / name)
```

Never commit `kaggle.json`, API tokens, or `.env` files. They are ignored by `.gitignore`.

Convert the CSVs to typed Parquet:

```bash
python -m src.data_loading
```

## 8. Execution order

```bash
# 1. Structural audit
python -m src.inspect_data

# 2. Select the documented subset and build the model dataset
python -m src.preprocessing

# 3. Train, evaluate, and save models and plots
python -m src.train_models

# 4. EDA figures and tables
python -m src.eda

# 5. Building-holdout generalization experiment
python -m src.generalization

# 6. Two unseen-case demonstrations
python -m src.predict

# 7. Tableau/Streamlit CSV extracts
python -m src.build_dashboard_data
```

The subset is selected with seed 42 and a documented site/use stratification rule. `src/train_models.py` uses a 250,000-row random sample for Random Forest and the full 769,122-row training split for Linear Regression and HistGradientBoosting on the local subset.

Optional environment overrides for a larger run:

```bash
ME446_SCOPE=full \
ME446_RF_SAMPLE_ROWS=500000 \
ME446_LINEAR_SAMPLE_ROWS=2000000 \
python -m src.train_models
```

The full-data workflow is documented in the notebook. Full-data metrics must be generated by the user and are not claimed in this repository.

## 9. Outputs

- `models/final_model.joblib` — selected fitted pipeline
- `models/model_metadata.json` — features, split counts, selection rule, test metrics
- `reports/data_inventory.md` — data-quality audit
- `reports/dataset_approval_proposal.md` — proposal marked awaiting approval
- `reports/technical_report.md` — full report
- `reports/engineering_insights.md` — five evidence-backed insights
- `reports/rubric_checklist.md` — honest evidence map for the seven criteria
- `reports/figures/` — 13 EDA/model/generalization figures
- `reports/tables/` — model comparison, feature importance, error tables, predictions
- `data/dashboard/tableau_data/` — Tableau-ready CSVs
- `dashboard/streamlit_app.py` — supplementary interactive app
- `presentation/` — 15-slide deck (`.pptx`), A1 poster (`.pdf`/`.tex`), slide outline, and poster content

## 10. Streamlit dashboard

Run from the project root:

```bash
streamlit run dashboard/streamlit_app.py
```

Pages:

1. **Energy Overview** — KPIs, hourly/daily trends, site and primary-use comparison
2. **Weather & Buildings** — temperature response, building-size relationship, metadata filters
3. **Model Performance** — model comparison, actual-vs-predicted, residuals, error tables
4. **Predict** — single-record prediction with the saved final model

The app was smoke-tested with Streamlit's `AppTest` across all four pages. It is supplementary and does **not** replace the required Tableau Public dashboard.

### Live deployment

- GitHub repository: <https://github.com/Shahbaz2104/me446-ashrae-energy>
- Streamlit Cloud entrypoint: `dashboard/streamlit_app.py`
- Deploy URL: <https://share.streamlit.io/deploy?repository=Shahbaz2104/me446-ashrae-energy&branch=main&mainModule=dashboard/streamlit_app.py>
- The repository includes the compact model dataset, prediction table, weather artifact, building metadata, saved model, and small dashboard tables needed by Streamlit Cloud. Raw data and the 100 MB+ Tableau extracts remain excluded.

## 11. Tableau dashboard

No Tableau workbook has been authored or published in this environment. The project supplies:

- `data/dashboard/tableau_data/` — CSV extracts
- `data/dashboard/README.md` — data dictionary and KPI definitions
- `dashboard/dashboard_specification.md` — three-dashboard design
- `dashboard/tableau_build_guide.md` — step-by-step build and interaction instructions

Follow the build guide in Tableau Cloud/Public, then update the report with the verified published URL. Do not claim a workbook exists until it has been opened and validated.

## 12. Notebook and full-data run

`notebooks/ME446_ASHRAE_Complete_Analysis.ipynb` runs the workflow end to end. It supports:

- **Subset mode** for local execution using the documented 120-building scope.
- **Full mode** in Google Colab using all electricity rows. Full mode uses a 2,000,000-row sample for Linear Regression and a 500,000-row sample for Random Forest to remain within Colab memory; HistGradientBoosting uses all available training rows. Sampling is documented and seeded.

Run the notebook from the project root or upload the project to Colab. Add `KAGGLE_API_TOKEN` as a Colab secret. The notebook does not contain credentials.

## 13. Tests

```bash
pytest -q
```

Tests cover weather sanitization, calendar features, feature-frame construction, proportional allocation, split boundaries, metrics, and saved-model inference.

## 14. Reproducibility notes

- Seed: 42 for subset selection, sampling, model fitting, and plots.
- Raw CSVs are retained unchanged.
- Parquet intermediates are reproducible from the raw CSVs.
- The local subset and all sampling are documented.
- The test period is never used for model selection.
- Package versions used for the reported results are pinned in `requirements.txt`.
- The local machine has 4 cores, 3.6 GB RAM, and no GPU; full-data training is not local.

## 15. Known limitations

1. Results are from a 120-building subset, not the full 1,413 electricity buildings.
2. The main model relies on `building_id`; unseen-building generalization is poor (R² -0.555).
3. December test data contain far fewer zero readings than the training period, so test metrics may be optimistic relative to periods with missing or zero readings.
4. Weather covariates are site-level, not building-level, and cloud/precipitation coverage is incomplete.
5. Building metadata has substantial missingness (`year_built` 51.76%, `floor_count` 70.90% in the subset).
6. Target units are documented but not independently verified.
7. No Tableau workbook has been authored or published; only data and instructions exist.
8. Full-data metrics are pending a user-run Colab execution.
9. Dataset uniqueness and teacher approval are pending.
10. Correlations and permutation importance are associational, not causal.

## 16. References

- ASHRAE. (2019). *Great Energy Predictor III*. Kaggle. https://www.kaggle.com/c/ashrae-energy-prediction
- Kaggle dataset re-upload: https://www.kaggle.com/datasets/sumit261124/ashrae-great-energy-predictor-iii-dataset
- Miller, C., Arjunan, P., Kathirgamanathan, A., et al. (2020). The ASHRAE Great Energy Predictor III competition: Overview and results. *Science and Technology for the Built Environment*, 26(10), 1427–1447.
- Amasyali, K., & El-Gohary, N. M. (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81, 1192–1205.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*.
- Tableau. *Tableau Public/Cloud documentation*. https://help.tableau.com/
- Streamlit. *Streamlit documentation*. https://docs.streamlit.io/
