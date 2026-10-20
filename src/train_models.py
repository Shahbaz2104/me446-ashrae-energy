"""Train, evaluate, and persist the three required regression models."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config, evaluation
from src.feature_engineering import assert_feature_consistency, build_feature_frame

MODEL_NAMES = ["Mean baseline", "Linear Regression", "Random Forest", "HistGradientBoosting"]


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


RF_SAMPLE_ROWS = env_int("ME446_RF_SAMPLE_ROWS", config.LOCAL_MODEL_SAMPLE_ROWS)
LINEAR_SAMPLE_ROWS = env_int("ME446_LINEAR_SAMPLE_ROWS", 0)
SCOPE = os.getenv("ME446_SCOPE", "local_subset")


def load_dataset() -> pd.DataFrame:
    columns = [*config.FEATURES, config.TARGET, "split", "timestamp"]
    frame = pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=columns)
    frame["primary_use"] = frame["primary_use"].astype("category")
    for column in config.NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    frame[config.TARGET] = frame[config.TARGET].astype("float64")
    return frame


def build_linear_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype="float64")
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, config.NUMERIC_FEATURES),
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )


def build_tree_preprocessor() -> ColumnTransformer:
    categorical = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype="float32"
    )
    return ColumnTransformer(
        [
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
            ("numeric", "passthrough", config.NUMERIC_FEATURES),
        ]
    )


def build_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_tree_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=16,
                    min_samples_leaf=20,
                    max_features=0.5,
                    n_jobs=2,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_tree_preprocessor()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def save_importance(
    pipeline: Pipeline, X_sample: pd.DataFrame, y_sample: pd.Series, prefix: str
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X_sample,
        y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=config.RANDOM_SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(config.TABLES_DIR / f"feature_importance_{prefix}.csv", index=False)
    return importance


def main() -> None:
    config.ensure_directories()
    frame = load_dataset()
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]

    X_train = train[config.FEATURES]
    y_train = train[config.TARGET]
    X_validation = validation[config.FEATURES]
    y_validation = validation[config.TARGET]
    X_test = test[config.FEATURES]
    y_test = test[config.TARGET]

    if len(train) > RF_SAMPLE_ROWS:
        rf_indices = train.sample(RF_SAMPLE_ROWS, random_state=config.RANDOM_SEED).index
        X_rf = train.loc[rf_indices, config.FEATURES]
        y_rf = train.loc[rf_indices, config.TARGET]
    else:
        X_rf, y_rf = X_train, y_train

    if LINEAR_SAMPLE_ROWS and len(train) > LINEAR_SAMPLE_ROWS:
        linear_indices = train.sample(
            LINEAR_SAMPLE_ROWS, random_state=config.RANDOM_SEED
        ).index
        X_linear = train.loc[linear_indices, config.FEATURES]
        y_linear = train.loc[linear_indices, config.TARGET]
    else:
        X_linear, y_linear = X_train, y_train

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": build_linear_pipeline(),
        "Random Forest": build_random_forest(),
        "HistGradientBoosting": build_hist_gradient_boosting(),
    }

    metric_records: list[dict] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for name in MODEL_NAMES:
        model = models[name]
        fit_start = time.perf_counter()
        if name == "Mean baseline":
            model.fit(X_train[["square_feet"]], y_train)
            fit_rows = len(X_train)
        elif name == "Random Forest":
            model.fit(X_rf, y_rf)
            fit_rows = len(X_rf)
        elif name == "Linear Regression":
            model.fit(X_linear, y_linear)
            fit_rows = len(X_linear)
        else:
            model.fit(X_train, y_train)
            fit_rows = len(X_train)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        if name == "Mean baseline":
            validation_pred = model.predict(X_validation[["square_feet"]])
            test_pred = model.predict(X_test[["square_feet"]])
        else:
            validation_pred = model.predict(X_validation)
            test_pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - predict_start
        predictions[name] = {"validation": validation_pred, "test": test_pred}

        for split_name, y_true, y_pred in (
            ("validation", y_validation, validation_pred),
            ("test", y_test, test_pred),
        ):
            metrics = evaluation.regression_metrics(y_true, y_pred)
            metric_records.append(
                {
                    "model": name,
                    "split": split_name,
                    **metrics,
                    "fit_seconds": round(fit_seconds, 2),
                    "predict_seconds": round(predict_seconds, 3),
                    "fit_rows": fit_rows,
                }
            )
        print(
            f"{name}: fit {fit_seconds:.1f}s | "
            f"validation MAE {metric_records[-2]['mae']:.2f} | "
            f"test MAE {metric_records[-1]['mae']:.2f}",
            flush=True,
        )
        gc.collect()

    metrics = evaluation.comparison_table(metric_records)
    metrics.to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)
    (config.TABLES_DIR / "model_comparison.md").write_text(
        metrics.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8"
    )

    validation_metrics = metrics[metrics["split"] == "validation"].sort_values("rmse")
    best_name = validation_metrics.iloc[0]["model"]
    best_model = models[best_name]
    print(f"Selected model by validation RMSE: {best_name}")

    model_path = config.MODELS_DIR / config.MODEL_FILENAME
    joblib.dump(best_model, model_path)

    best_test_pred = predictions[best_name]["test"]
    best_validation_pred = predictions[best_name]["validation"]
    evaluation.plot_actual_vs_predicted(
        y_test, best_test_pred, best_name, config.FIGURES_DIR / "actual_vs_predicted.png"
    )
    evaluation.plot_residuals(
        y_test, best_test_pred, best_name, config.FIGURES_DIR / "residual_analysis.png"
    )
    evaluation.plot_model_comparison(metrics, config.FIGURES_DIR / "model_comparison.png")

    test_frame = test[
        ["building_id", "timestamp", "site_id", "primary_use", config.TARGET]
    ].copy()
    test_frame = test_frame.rename(columns={config.TARGET: "actual"})
    test_frame["predicted"] = np.clip(best_test_pred, 0, None)
    test_frame["residual"] = test_frame["actual"] - test_frame["predicted"]
    test_frame["absolute_error"] = test_frame["residual"].abs()
    test_frame.to_csv(config.TABLES_DIR / "test_predictions_best_model.csv", index=False)

    long_frames = []
    for name in MODEL_NAMES:
        if name not in predictions:
            continue
        model_frame = test[["building_id", "timestamp", "site_id", "primary_use"]].copy()
        model_frame["model"] = name
        model_frame["actual"] = y_test.to_numpy()
        model_frame["predicted"] = np.clip(predictions[name]["test"], 0, None)
        model_frame["residual"] = model_frame["actual"] - model_frame["predicted"]
        model_frame["absolute_error"] = model_frame["residual"].abs()
        long_frames.append(model_frame)
    pd.concat(long_frames, ignore_index=True).to_csv(
        config.DATA_DASHBOARD / "model_test_predictions.csv", index=False
    )

    by_use = evaluation.segment_metrics(test_frame, "primary_use")
    by_site = evaluation.segment_metrics(test_frame, "site_id")
    by_band = evaluation.reading_band_metrics(test_frame)
    by_use.to_csv(config.TABLES_DIR / "error_by_primary_use.csv", index=False)
    by_site.to_csv(config.TABLES_DIR / "error_by_site.csv", index=False)
    by_band.to_csv(config.TABLES_DIR / "error_by_reading_band.csv", index=False)

    importance_sample = test.sample(
        min(10_000, len(test)), random_state=config.RANDOM_SEED
    )
    importance = save_importance(
        best_model,
        importance_sample[config.FEATURES],
        importance_sample[config.TARGET],
        prefix=best_name.lower().replace(" ", "_"),
    )
    evaluation.plot_feature_importance(
        importance, config.FIGURES_DIR / "feature_importance.png"
    )

    case_indices = (
        test_frame[test_frame["actual"] > 0]
        .sort_values("absolute_error")
        .iloc[[0, len(test_frame[test_frame["actual"] > 0]) // 2]]
        .index
    )
    unseen = test.loc[case_indices].copy()
    unseen.insert(0, "case_id", ["case_1_median_error", "case_2_typical"][: len(unseen)])
    unseen.to_csv(config.DATA_PROCESSED / "unseen_cases.csv", index=False)

    try:
        consistency = build_feature_frame(
            unseen.drop(columns=[config.TARGET, "split"]),
            buildings=pd.read_parquet(config.BUILDINGS_PARQUET),
        )
        assert_feature_consistency(consistency, unseen[config.FEATURES])
        consistency_status = "passed"
    except Exception as exc:  # pragma: no cover - reported, not hidden
        consistency_status = f"failed: {exc}"

    metadata = {
        "model_name": best_name,
        "selection_rule": "Lowest validation RMSE; test period held out until after selection.",
        "scope": SCOPE,
        "target": config.TARGET,
        "meter": config.ELECTRICITY_METER,
        "features": config.FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "train_period": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "validation_period": [
            str(validation["timestamp"].min()),
            str(validation["timestamp"].max()),
        ],
        "test_period": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "random_seed": config.RANDOM_SEED,
        "random_forest_fit_rows": int(len(X_rf)),
        "linear_fit_rows": int(len(X_linear)),
        "best_validation_rmse": float(
            validation_metrics.iloc[0]["rmse"]
        ),
        "test_metrics": validation_metrics.iloc[0].to_dict(),
        "feature_consistency_check": consistency_status,
        "packages": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata["best_validation_rmse"] = float(
        metrics[
            (metrics["model"] == best_name) & (metrics["split"] == "validation")
        ]["rmse"].iloc[0]
    )
    metadata["test_metrics"] = {
        key: float(value)
        for key, value in metrics[
            (metrics["model"] == best_name) & (metrics["split"] == "test")
        ].iloc[0].to_dict().items()
        if key not in {"model", "split"}
    }
    (config.MODELS_DIR / config.METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print(metrics.to_string(index=False))
    print(f"Saved model to {model_path}")
    print(f"Feature consistency check: {consistency_status}")


if __name__ == "__main__":
    main()
