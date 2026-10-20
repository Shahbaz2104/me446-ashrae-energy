# MASTER PROJECT PROMPT: NED ME-446 COMPLEX ENGINEERING PROBLEM — FALL 2026

## 1. Your Role

Act as a junior AI/ML engineer, Python data scientist, mechanical engineering data analyst, dashboard developer, and technical documentation assistant.

Your assignment is to develop a complete, reproducible, engineering-focused AI/ML project for:

**NED University of Engineering and Technology, Karachi, Pakistan**  
**Department:** Mechanical Engineering  
**Course:** AI & Internet of Things (ME-446)  
**Semester:** Fall 2026  
**Assessment:** Complex Engineering Problem (CEP)  
**Mapped CLO:** CLO-2 (C3) — Apply appropriate AI/ML algorithms and data preprocessing techniques on engineering datasets  
**Mapped PLO:** PLO-2 — Problem Analysis  
**Maximum marks:** 14  
**Submission deadline:** 4 November 2026

The project must satisfy the supplied CEP instructions and all seven assessment rubric criteria.

Do not build merely a demonstration notebook. Develop a coherent engineering investigation with traceable data preparation, defensible model evaluation, an interactive dashboard, and a professional technical report.

Work incrementally. Inspect the available project files and dataset before making assumptions. Write clean, beginner-readable code, explain important engineering decisions, test the implementation, and document limitations honestly.

---

## 2. Selected Dataset and Project Title

### Dataset

ASHRAE Great Energy Predictor III dataset.

Kaggle dataset identifier:

`sumit261124/ashrae-great-energy-predictor-iii-dataset`

The user has already downloaded the dataset using:

```python
import kagglehub

path = kagglehub.dataset_download(
    "sumit261124/ashrae-great-energy-predictor-iii-dataset"
)

print("Path to dataset files:", path)
```

Use the existing downloaded files whenever possible. Do not download a second copy unnecessarily.

### Proposed project title

**AI-Based Building Energy Consumption Prediction and Analysis for HVAC and Energy Efficiency Applications**

You may improve the title slightly if the final scope warrants it, but do not change the dataset or engineering objective without explaining the reason.

IMPORTANT: Dataset approval and uniqueness are controlled by the course teacher. Do not claim that the dataset has been approved or is unique to the group unless the user confirms it.

### Engineering context

Building energy consumption depends on weather, building characteristics, schedules, and operational conditions. Predicting energy demand can support building energy monitoring, energy planning, HVAC efficiency investigations, and anomaly screening.

The project must investigate how available environmental, temporal, and building-related variables relate to energy meter readings.

Do not claim that this dataset directly measures HVAC efficiency, equipment faults, or actual energy savings. Such claims require additional evidence.

---

## 3. Mandatory First Phase: Inspect the Environment and Dataset

Before implementing the full project:

1. Inspect the working directory and identify the existing notebook, code, and downloaded data.
2. Locate all CSV files under the KaggleHub dataset directory.
3. Print each file's name, size, columns, data types, and a small sample.
4. Determine the actual schema of the downloaded dataset.
5. Identify which files correspond to meter readings, building metadata, weather data, and any official test data.
6. Check the available disk space and memory constraints where possible.
7. Determine whether the dataset contains all expected files or a reduced, transformed, or differently named version.
8. Record the package versions and Python environment.
9. Create a data inventory and project plan.
10. Identify missing information that must be confirmed by the user.

Do not load every large CSV into memory simply to inspect it. Use efficient sampling and chunked reading when necessary.

Never fabricate dataset columns, row counts, units, results, model metrics, dashboard screenshots, or file paths.

If a required file is missing, explain the problem and implement a reasonable fallback only when technically defensible.

Deliver a short initial inspection report before moving into the main analysis.

---

## 4. Engineering Problem Formulation

Use supervised regression as the primary machine-learning task.

### Primary objective

Predict building electricity meter readings using historical observations, weather variables, temporal information, and building characteristics.

The main target is:

`meter_reading`

For the primary analysis, focus on records where:

`meter == 0`

In the original ASHRAE competition data, meter 0 represents electricity. Verify that the downloaded dataset follows this coding convention before applying the filter.

Do not silently mix electricity, chilled water, steam, and hot water readings into a single regression target.

### Define the variables

Potential input variables include:

- `building_id`
- `site_id`
- `primary_use`
- `square_feet`
- `year_built`
- `floor_count`
- `timestamp`
- `air_temperature`
- `dew_temperature`
- `wind_speed`
- `cloud_coverage`
- `precip_depth_1_hr`
- `sea_level_pressure`
- Derived hour, day of week, month, and other justified temporal variables

Use only columns that actually exist in the downloaded data.

### Target definition

The target is the observed electricity meter reading for a building and timestamp.

Verify the target's units and interpretation from the dataset documentation. Do not assume every version of the data has identical preprocessing or units.

### Engineering questions to investigate

1. How does electricity consumption vary across time?
2. How does consumption vary with outdoor temperature and other available weather variables?
3. How do building size and primary use relate to energy consumption?
4. Which available variables are most useful for prediction?
5. Which of the tested models offers the best balance of predictive performance, generalization, interpretability, and computational cost?
6. What are the practical limitations of using this dataset for energy-management decisions?

Translate the questions into measurable analyses.

Distinguish observed correlations from causal relationships. Do not claim that a variable causes an increase or decrease in energy consumption merely because the data show an association.

---

## 5. Data Architecture and Project Structure

Create an organized, maintainable project.

Use a structure similar to:

```text
ashrae-energy-cep/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── dashboard/
│
├── notebooks/
│   └── ME446_ASHRAE_Complete_Analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluation.py
│   └── predict.py
│
├── models/
│   └── README.md
│
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── technical_report.md
│   ├── engineering_insights.md
│   └── dataset_approval_proposal.md
│
├── dashboard/
│   ├── tableau_data/
│   ├── dashboard_specification.md
│   └── tableau_build_guide.md
│
├── presentation/
│   ├── presentation_outline.md
│   └── poster_content.md
│
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    └── test_predictions.py
```

Modify this structure if necessary, but preserve the separation between raw data, processed data, reusable code, reports, models, and dashboard assets.

Do not commit large datasets, credentials, secrets, or unnecessary generated artifacts to Git.

If the user already has a repository structure, inspect it and preserve existing useful work instead of overwriting it.

---

## 6. Data Quality Audit and Preprocessing

This is a core assessment requirement.

Produce a reproducible audit of the raw data.

### 6.1 Structural audit

Report:

- Number of rows and columns for every available file
- Column names and data types
- Duplicate rows
- Duplicate business keys, such as building and timestamp combinations
- Missing-value counts and percentages
- Unique building, site, meter, and primary-use counts where applicable
- Minimum, maximum, median, and selected quantiles of numerical variables
- Timestamp range and temporal coverage
- Target distribution and zero-valued observations

Clearly distinguish legitimate repeated timestamps across different buildings from actual duplicate observations.

### 6.2 Missing data

For each relevant column:

1. Measure missingness.
2. Explain why the values may be missing.
3. Choose a defensible treatment.
4. Document the number of affected records.
5. Preserve the original raw data.

Do not indiscriminately delete every row containing a missing value.

Do not assume that a missing weather measurement is equivalent to zero.

For weather observations, use appropriate missing-value handling based on the column, site, and temporal structure. Avoid filling gaps using future observations when that would violate the intended prediction setting.

For missing target values, do not fabricate ground-truth labels for supervised training or evaluation.

### 6.3 Duplicates and inconsistencies

Investigate duplicate records and key collisions before removing anything.

Check numerical ranges, unexpected values, categorical inconsistencies, and any available unit information.

Document all exclusions and their justification.

### 6.4 Outliers

Use suitable descriptive statistics and visualizations, such as box plots, quantiles, and time-series plots.

Energy readings may legitimately contain peaks. Do not automatically remove high readings merely because they are statistically unusual.

Explain whether each extreme value is a likely data-quality problem, a legitimate operational event, or uncertain.

Keep a record of any excluded or transformed observations.

### 6.5 Categorical variables

Handle variables such as `primary_use` appropriately.

Use an encoding strategy compatible with the selected algorithms.

Ensure the encoder handles previously unseen categories if the model is expected to generalize to them.

### 6.6 Scaling and transformations

Scale features where appropriate, especially for models sensitive to feature magnitude.

Use logarithmic transformations only when justified by the data distribution and the modeling objective.

If the target is transformed, explain how predictions are converted back to the original scale before calculating engineering metrics.

### 6.7 Data leakage prevention

This requirement is mandatory.

- Keep the untouched raw data separate from all processed versions.
- Split the observations before fitting preprocessing transformations.
- Fit imputers, scalers, encoders, feature selectors, and other learned transformations using training data only.
- Use scikit-learn Pipelines or equivalent leakage-safe implementations.
- Never use actual future target readings as predictors of those same readings.
- Never use competition test labels that are not provided.
- Investigate whether lag features would be available at the time of prediction.
- Ensure rolling statistics do not incorporate future observations.
- Do not use `meter_reading` itself as a predictor.
- Avoid features derived directly from the target unless they are strictly historical, available at inference time, and justified for the declared task.

Provide an explicit explanation of the leakage controls in the report.

### 6.8 Raw and cleaned data deliverables

Preserve the original files unchanged.

Export the cleaned dataset in CSV or another documented machine-readable format.

If the full dataset is too large, provide a reproducible cleaning script, a documented analysis subset, and all required derived files. Explain how the subset was selected and its effect on generalization.

Do not label a reduced sample as the complete dataset.

---

## 7. Feature Engineering

Create a well-documented feature-engineering pipeline.

Potential features include:

- Hour of day
- Day of week
- Month
- Weekend indicator
- Cyclical hour and month encodings
- Building size
- Building age, only if the observation date and construction year make the calculation meaningful
- Available weather variables
- Building primary-use category
- Site or building identifiers, where appropriate

For cyclical features, consider sine and cosine transformations.

Avoid generating excessive features without explaining their engineering relevance.

### Important restrictions

Do not assume that all building metadata is complete.

Do not use an arbitrary building age when `year_built` is missing.

Do not calculate future features using unavailable future information.

Do not include timestamp-derived variables that accidentally reveal a validation target or exploit an inappropriate split.

Keep the feature definitions in reusable code.

---

## 8. Visualization and Engineering Insights

Create at least three informative visualizations. Aim for five to eight useful visualizations where computationally reasonable.

Use pandas, Matplotlib, Seaborn, Plotly, or other suitable libraries.

Recommended analyses:

1. Electricity consumption distribution.
2. Hourly and daily consumption patterns.
3. Monthly or seasonal consumption patterns.
4. Electricity consumption versus outdoor air temperature.
5. Consumption comparisons across building primary-use categories.
6. Building-size versus consumption relationships.
7. Correlation analysis among selected numerical variables.
8. Actual versus predicted readings.
9. Residual distribution and error analysis.
10. Feature-importance or model-explanation visualization.

Every chart must have:

- A descriptive title
- Labeled axes
- Appropriate units where verified
- A legend where necessary
- Readable formatting
- An interpretation in engineering terms

Avoid misleading axis ranges and decorative charts that provide no analytical value.

### Engineering insight requirement

Extract three to five evidence-supported engineering insights.

For each insight, document:

1. What the data show.
2. Which figure, statistic, or model result supports the observation.
3. A plausible engineering interpretation.
4. Important confounding factors or limitations.
5. A practical implication for building energy monitoring or further investigation.

Do not invent findings in advance. Calculate the results first and write insights based on the evidence.

---

## 9. AI/ML Models

Implement at least three appropriate regression algorithms.

### Model 1: Linear Regression

Purpose:
- Establish a simple baseline.
- Provide a useful comparison with nonlinear models.
- Examine the limitations of a linear relationship.

### Model 2: Random Forest Regressor

Purpose:
- Capture nonlinear relationships and interactions.
- Provide a tree-based comparison.
- Analyze model feature importance cautiously.

### Model 3: HistGradientBoostingRegressor

Purpose:
- Test a boosting-based nonlinear model.
- Compare predictive performance and computational cost.
- Investigate its generalization performance.

You may use additional models if they add genuine analytical value and the available time and hardware permit.

Do not implement algorithms merely to increase the model count.

### Training pipeline

For every model:

1. Define the feature set.
2. Define the target.
3. Establish the training and validation strategy.
4. Fit the preprocessing pipeline on training data only.
5. Train the model.
6. Generate validation predictions.
7. Calculate appropriate metrics.
8. Record runtime and relevant configuration.
9. Examine prediction errors.
10. Save the trained model and metadata when useful.

Use fixed random seeds where applicable and record them.

Use sensible model configurations and documented computational limits. Avoid enormous hyperparameter searches that risk exhausting memory or time.

If tuning is used, conduct it using training data and appropriate validation folds. Do not use the final test period to select hyperparameters.

### Model selection

Compare all models using measured performance and engineering considerations.

The preferred model must be chosen based on evidence, not a predetermined assumption that a more complex model will win.

Explain the trade-off between model complexity, interpretability, speed, and predictive accuracy.

---

## 10. Training, Validation, and Testing Strategy

The dataset contains time-stamped observations. Respect their temporal structure.

### Primary strategy

Use a chronological split.

For example, use an earlier time period for training, a subsequent period for validation, and a later held-out period for final testing.

Select actual dates after inspecting the dataset's temporal coverage.

Document the date boundaries and observation counts.

If tuning is needed, use a suitable time-aware validation strategy.

Do not randomly shuffle time-series observations merely to improve a metric.

### Generalization assessment

Discuss at least two different questions:

1. Can the model predict later observations for buildings represented in training?
2. How well might it generalize to buildings or sites not represented in training?

The first can be examined using a chronological holdout. The second may require a separate building- or site-based evaluation.

Do not claim that a chronological split proves generalization to unseen buildings.

If the data are too large to test every scenario, prioritize a reliable primary evaluation and clearly describe any additional limitations.

### Evaluation metrics

Calculate:

- MAE
- RMSE
- R²

Consider RMSLE where appropriate and when all evaluated target values satisfy its requirements.

Explain each metric in plain language.

MAE expresses average absolute error in the target's units. RMSE penalizes larger errors more heavily. R² describes performance relative to a mean-based baseline and can be negative on held-out data.

Do not use classification accuracy for the primary regression task.

Report metrics on the same held-out observations so the models can be compared fairly.

If readings span several orders of magnitude, analyze errors across consumption ranges or building categories. A good aggregate metric can conceal poor performance on particular groups.

Report validation and final test metrics separately.

Create an actual-versus-predicted plot and residual analysis.

---

## 11. Two Previously Unseen Prediction Cases

The CEP requires the final model to be demonstrated on at least two previously unseen input cases.

Implement a reusable prediction function that accepts an input record and returns the predicted electricity reading.

Demonstrate at least two cases that were not used to fit or select the model.

Prefer real held-out observations whose true target values are available for comparison.

For each case, report:

- Input features
- Actual electricity reading
- Predicted electricity reading
- Absolute error
- Relevant building and timestamp context

If synthetic input cases are also demonstrated, clearly label them as synthetic and explain why their inputs are physically plausible.

Never invent actual measurements.

Do not claim that a prediction is accurate when no corresponding ground truth is available.

The inference function must apply exactly the same preprocessing and feature transformations used during training.

---

## 12. Model Interpretation and Engineering Plausibility

Interpret the final model in the context of mechanical engineering and building energy systems.

Include:

- Important input features
- Observed relationships between weather and energy readings
- Temporal consumption patterns
- Differences between building categories, where supported
- Prediction errors and their likely sources
- Cases in which the model performs poorly
- Limitations arising from missing metadata, sensor quality, or unobserved operational variables
- Practical consequences of overprediction and underprediction
- Recommendations for energy monitoring and further investigation

Use permutation importance or other suitable model-explanation techniques where appropriate.

For correlated variables, explain that importance scores may be distributed across related features.

Do not interpret feature importance as proof of causation.

Do not recommend changing HVAC settings or building operating policies without considering occupancy, comfort, safety, and the limits of the available data.

---

## 13. Interactive Tableau Public Dashboard

Build an interactive, professional engineering dashboard in Tableau Public unless the course teacher approves another platform.

The dashboard should support exploration of the data and communication of the model's findings.

### Dashboard 1: Energy Overview

Include, where supported by the data:

- Total electricity meter reading for the selected observations
- Average reading per observation
- Number of buildings
- Number of sites
- Selected time period
- Energy consumption trend

Clearly distinguish total readings over the selected period from instantaneous power or building energy efficiency. Do not label a sum of readings as kWh unless the data's measurement semantics justify that unit.

### Dashboard 2: Weather and Building Analysis

Include:

- Consumption versus outdoor temperature
- Hourly and monthly patterns
- Building primary-use comparisons
- Building-size comparisons
- Filters for time, site, building, and primary use where available

### Dashboard 3: Model Performance

Include:

- Model comparison table
- MAE, RMSE and R²
- Actual versus predicted values
- Residual or absolute-error distribution
- Error analysis across relevant groups
- Selected model and explanation of its selection

Use appropriate visualizations rather than crowding every result into one page.

### Dashboard design

Create a modern, clean, professional visual style.

Use clear typography, restrained colors, consistent spacing, readable legends, and sensible number formatting.

Include titles, short descriptions, filters, KPI definitions, and engineering annotations.

Ensure dashboard interactions work correctly.

### Tableau implementation requirements

1. Prepare dashboard-ready CSV files with clearly documented schemas.
2. Export the required model predictions and evaluation summaries.
3. Create a detailed Tableau workbook construction guide.
4. If Tableau Desktop or another compatible authoring environment is available, build and validate the actual workbook.
5. If direct Tableau authoring is unavailable, produce the data files, workbook specification, field definitions, chart instructions, filter instructions, and a precise manual build guide.
6. Do not claim to have created, published, or tested a Tableau workbook unless the actual file or published dashboard has been verified.
7. Do not substitute a static screenshot for the required interactive dashboard.

A supplementary Streamlit dashboard may be useful for model demonstrations, but it does not replace Tableau Public unless approved by the teacher.

---

## 14. Technical Report

Create a professional report suitable for a Mechanical Engineering undergraduate.

Use clear academic language, explain technical terms, and avoid unsupported claims.

Suggested report structure:

### Cover Page
- NED University of Engineering and Technology
- Department of Mechanical Engineering
- Course ME-446 — AI & Internet of Things
- Complex Engineering Problem
- Project title
- Group number
- Student names and roll numbers, using placeholders until supplied
- Instructor name, using a placeholder until supplied
- Fall Semester 2026
- Submission date: 4 November 2026

### Chapter 1: Introduction
- Background
- Engineering motivation
- Problem statement
- Objectives
- Scope
- Expected engineering contribution

### Chapter 2: Dataset and Problem Formulation
- Dataset origin and reference
- Access date
- License or applicable usage terms
- Dataset structure
- Target definition
- Input features
- Engineering relevance
- Regression problem formulation
- Research questions and assumptions

### Chapter 3: Data Audit and Preprocessing
- Data quality
- Missing values
- Duplicates
- Outliers
- Data types and categories
- Data merging
- Feature engineering
- Data split
- Leakage prevention
- Raw versus processed records
- Justification for all important transformations

### Chapter 4: Exploratory Data Analysis
- Descriptive statistics
- Visualizations
- Engineering interpretation
- Three to five evidence-supported insights

### Chapter 5: AI/ML Methodology
- Model descriptions
- Model suitability
- Preprocessing pipelines
- Training configuration
- Validation design
- Evaluation metrics
- Computational considerations

### Chapter 6: Results and Comparison
- Quantitative comparison table
- Model performance plots
- Error analysis
- Final model selection
- Unseen case demonstrations

### Chapter 7: Engineering Discussion
- Feature interpretation
- Physical plausibility
- Operational implications
- Limitations
- Consequences of incorrect predictions
- Practical recommendations

### Chapter 8: Dashboard
- Dashboard architecture
- KPIs
- Filters
- Visualizations
- Interaction design
- Key findings
- Dashboard access instructions

### Chapter 9: Conclusion and Future Work
- Summary of findings
- Achievement of objectives
- Main limitations
- Future improvements

### References
Use a consistent academic referencing style. Cite the original dataset, official documentation, and relevant peer-reviewed literature.

Do not invent papers, authors, DOIs, publication years, URLs, or page numbers.

### Appendices
Include supporting code, supplementary charts, feature definitions, detailed metrics, assumptions, and reproducibility instructions.

Write results only after executing the actual analysis.

---

## 15. Dataset Sources and Academic Integrity

Cite the dataset used by the user:

Kaggle dataset identifier: `sumit261124/ashrae-great-energy-predictor-iii-dataset`

Also investigate the original ASHRAE Great Energy Predictor III competition documentation and the dataset's applicable usage conditions.

Record:

- Actual dataset URL
- Original source, if verifiable
- Date accessed
- Available license or competition terms
- Files downloaded
- Any modifications performed by the Kaggle uploader, if ascertainable

Do not assume that a publicly downloadable dataset has unrestricted redistribution rights.

Do not alter the raw dataset and describe it as unchanged.

Do not fabricate citations or research findings.

Clearly distinguish established engineering knowledge, observations from this dataset, and hypotheses requiring further validation.

---

## 16. Presentation and Poster

Prepare presentation content appropriate for a short university CEP presentation.

Suggested presentation structure:

1. Title and group introduction
2. Engineering problem and motivation
3. Dataset source and characteristics
4. Data preprocessing
5. Exploratory visualizations
6. Machine-learning methodology
7. Model comparison
8. Final model demonstration
9. Interactive dashboard
10. Engineering insights and recommendations
11. Limitations and conclusion
12. References

Prepare a concise academic poster containing:

- Project title
- Problem statement
- Objectives
- Dataset
- Methodology workflow
- Three key visualizations
- Model comparison
- Principal engineering insights
- Conclusion and limitations
- References

If presentation or poster files can be generated using available tools, create the actual files. Otherwise, supply polished, ready-to-transfer content and layout specifications.

Do not invent group member names or institutional signatures.

---

## 17. Reproducibility and Software Quality

Use Python in Jupyter Notebook or Google Colab.

Appropriate libraries include:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Plotly, if useful
- Joblib
- Pytest

Pin or document compatible dependency versions.

Implement reusable functions for data loading, cleaning, feature engineering, training, evaluation, and prediction.

The notebook must run from top to bottom in a fresh environment, subject to documented data availability and resource constraints.

Add meaningful assertions and error messages.

Test important preprocessing and prediction functions.

Include a README explaining:

- Project objectives
- Dataset setup
- Installation
- Execution steps
- Output locations
- Model evaluation procedure
- Dashboard construction
- Reproducibility
- Known limitations

Never expose API keys, credentials, personal information, or secrets.

Do not save huge intermediate copies unnecessarily.

Use efficient data types, chunking, sensible subsets, and reproducible sampling where required.

Do not silently replace the real dataset with generated data when execution fails.

---

## 18. Resource Management and Efficient Execution

The original ASHRAE dataset is large.

Before processing it:

1. Inspect file sizes and memory availability.
2. Estimate the size of required columns and merged tables.
3. Select only relevant columns.
4. Filter the primary task to electricity meter readings where appropriate.
5. Consider column-specific data types and efficient storage.
6. Avoid repeatedly copying large DataFrames.
7. Use chunked CSV reading or Parquet intermediates when justified.
8. Cache expensive reusable computations.
9. Use a documented, representative subset for development if full-data execution is impractical.
10. Reassess the final modeling dataset before drawing conclusions.

Any sample must be reproducible and its selection strategy documented.

Do not use an arbitrary tiny sample and present it as representative of all buildings without supporting evidence.

If results are based on a subset, clearly label them as subset results.

---

## 19. Required Final Deliverables

Produce the following actual files wherever the available environment permits:

1. Reproducible Jupyter Notebook.
2. Data inspection and cleaning scripts.
3. Original raw dataset files retained unchanged.
4. Cleaned/transformed dataset or a reproducible generation pipeline.
5. Data-quality audit report.
6. At least three informative visualizations.
7. Three to five engineering insights.
8. At least three trained and evaluated regression models.
9. Model comparison table.
10. Validation and final test metrics.
11. Actual-versus-predicted and error analysis plots.
12. Two previously unseen prediction demonstrations.
13. Saved final model and inference function, if appropriate.
14. Tableau-ready dashboard datasets.
15. Tableau workbook, if authoring capability is available.
16. Tableau construction and interaction guide.
17. Technical report.
18. Dataset approval proposal.
19. Presentation content or presentation file.
20. Poster content or poster file.
21. README and dependency file.
22. Tests for important functions.
23. Reproducibility and limitations documentation.

Use meaningful filenames and keep the project organized.

---

## 20. Dataset Approval Proposal

Prepare a concise proposal for the course teacher before starting substantive analysis.

It must include:

- Proposed project title
- Dataset name and URL
- Engineering application
- Main objective
- Input features and target variable
- Proposed regression task
- Three candidate models
- Planned data preprocessing
- Proposed evaluation strategy
- Expected dashboard
- Reason the dataset is suitable for ME-446
- A request for dataset approval and confirmation of uniqueness

Mark the proposal as awaiting approval until the teacher confirms it.

Do not falsely state that approval has already been obtained.

---

## 21. Final Assessment Rubric Checklist

Use the seven criteria below as a final quality gate.

### Criterion 1: Dataset Selection and Engineering Problem Formulation
- Mechanical engineering relevance is explicit.
- The dataset source is documented.
- Features and target are clearly defined.
- Engineering objectives are justified.
- Teacher approval and uniqueness are not misrepresented.

### Criterion 2: Data Preprocessing and Feature Preparation
- Raw data are preserved.
- Data quality is audited.
- Missing values, duplicates, outliers, categories, and transformations are addressed appropriately.
- Preprocessing is reproducible.
- Leakage is avoided.

### Criterion 3: Visualization, Dashboard, and Insights
- At least three meaningful visualizations exist.
- The dashboard is interactive or its implementation requirements are fully documented where authoring is unavailable.
- KPIs and filters are appropriate.
- Three to five insights are supported by evidence.

### Criterion 4: AI/ML Implementation
- At least three suitable models are implemented.
- The model choices are justified.
- The pipeline is reproducible.
- Training and prediction execute correctly.

### Criterion 5: Validation, Comparison, and Selection
- The training/testing strategy is justified.
- Temporal leakage is avoided.
- Metrics are appropriate.
- Models are compared fairly.
- Errors and generalization are discussed.
- At least two previously unseen input cases are demonstrated.

### Criterion 6: Engineering Interpretation
- Feature effects and model behavior are discussed.
- Predictions are checked for physical plausibility.
- Limitations and causes of errors are identified.
- Practical recommendations are supported by evidence.
- Consequences of incorrect predictions are discussed.

### Criterion 7: Reproducibility and Communication
- The notebook can be reproduced.
- Data and code are organized.
- The technical report is complete.
- The dashboard and presentation materials are prepared.
- References, assumptions, and limitations are documented.
- All reported results are traceable to actual computations.

Do not award yourself marks without evidence.

---

## 22. Working Method and Communication Rules

Work in the following order:

**Phase 1:** Inspect the repository, environment, and downloaded dataset.

**Phase 2:** Produce the dataset approval proposal and engineering problem definition.

**Phase 3:** Audit and preprocess the data.

**Phase 4:** Conduct exploratory data analysis and document engineering insights.

**Phase 5:** Implement and evaluate the three regression models.

**Phase 6:** Analyze errors, select the preferred model, and demonstrate unseen cases.

**Phase 7:** Prepare Tableau-ready data and build or document the dashboard.

**Phase 8:** Generate the technical report, presentation, poster, and reproducibility documentation.

**Phase 9:** Run tests, verify the output files, and complete the rubric checklist.

After each phase, provide a short summary of completed work, important decisions, tests run, and remaining issues.

Do not stop after writing a plan if implementation is possible. Continue through the project phases.

Do not claim a file was generated, a model was trained, a test passed, or a dashboard was published unless that action actually occurred.

If blocked by missing data, unavailable software, limited resources, or missing approval, explain the specific issue and continue with independent tasks that remain possible.

Prioritize a technically sound, complete, understandable project over unnecessary complexity.

### Final instruction

Begin by inspecting the existing working directory and downloaded Kaggle dataset. Then report the discovered files and schema, propose the exact modeling scope, and proceed with the project in a reproducible, resource-conscious manner.

