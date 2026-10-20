"""Regression metrics, error tables, and evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

MODEL_COLORS = {
    "Linear Regression": "#4C72B0",
    "Random Forest": "#DD8452",
    "HistGradientBoosting": "#55A868",
    "Mean baseline": "#8C8C8C",
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, R², and RMSLE after clipping impossible negatives."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))),
        "n_rows": int(len(y_true)),
        "actual_mean": float(np.mean(y_true)),
        "predicted_mean": float(np.mean(y_pred)),
    }


def comparison_table(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    ordered = [
        "model",
        "split",
        "n_rows",
        "mae",
        "rmse",
        "r2",
        "rmsle",
        "fit_seconds",
        "predict_seconds",
    ]
    return frame[[column for column in ordered if column in frame.columns]]


def segment_metrics(
    frame: pd.DataFrame, group_column: str, min_rows: int = 50
) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_column, observed=True):
        if len(subset) < min_rows:
            continue
        metrics = regression_metrics(subset["actual"], subset["predicted"])
        rows.append({group_column: group, **metrics})
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def reading_band_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.001, 10, 50, 100, 250, 500, 1000, 5000, np.inf]
    labels = ["0–10", "10–50", "50–100", "100–250", "250–500", "500–1000", "1000–5000", ">5000"]
    banded = frame.copy()
    banded["reading_band"] = pd.cut(
        banded["actual"], bins=bins, labels=labels, include_lowest=True
    )
    return segment_metrics(banded, "reading_band", min_rows=20)


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    destination: Path,
    sample_size: int = 25_000,
    seed: int = config.RANDOM_SEED,
) -> None:
    frame = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": np.clip(y_pred, 0, None)})
    if len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=seed)
    positive = frame[(frame["actual"] > 0) & (frame["predicted"] > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(frame["actual"], frame["predicted"], s=4, alpha=0.18, color="#4C72B0")
    limit = max(frame["actual"].quantile(0.995), frame["predicted"].quantile(0.995))
    axes[0].plot([0, limit], [0, limit], color="#C44E52", linewidth=1, label="Perfect prediction")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Actual electricity reading (dataset units)")
    axes[0].set_ylabel("Predicted electricity reading (dataset units)")
    axes[0].set_title("Linear scale (99.5th percentile)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].scatter(positive["actual"], positive["predicted"], s=4, alpha=0.18, color="#55A868")
    low = max(positive[["actual", "predicted"]].min().min(), 0.1)
    high = positive[["actual", "predicted"]].max().max()
    axes[1].plot([low, high], [low, high], color="#C44E52", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Actual reading (log scale)")
    axes[1].set_ylabel("Predicted reading (log scale)")
    axes[1].set_title("Log scale (positive readings)")
    fig.suptitle(f"Actual vs predicted — {model_name} (test period, December 2016)")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    destination: Path,
    sample_size: int = 25_000,
    seed: int = config.RANDOM_SEED,
) -> None:
    y_true = y_true.to_numpy(dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    residual = y_true - y_pred
    frame = pd.DataFrame({"predicted": y_pred, "residual": residual})
    if len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=seed)
    clipped = frame["residual"].clip(
        frame["residual"].quantile(0.005), frame["residual"].quantile(0.995)
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(clipped, bins=60, color="#4C72B0", alpha=0.85)
    axes[0].axvline(0, color="#C44E52", linewidth=1)
    axes[0].set_xlabel("Residual = actual − predicted (dataset units)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual distribution (0.5–99.5% shown)")

    axes[1].scatter(frame["predicted"], frame["residual"], s=4, alpha=0.15, color="#DD8452")
    axes[1].axhline(0, color="#C44E52", linewidth=1)
    axes[1].set_xlabel("Predicted reading (dataset units)")
    axes[1].set_ylabel("Residual (dataset units)")
    axes[1].set_title("Residual vs predicted")
    fig.suptitle(f"Residual analysis — {model_name} (test period, December 2016)")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, destination: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    models = test["model"].tolist()
    colors = [MODEL_COLORS.get(model, "#8172B3") for model in models]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for axis, column, title, ylabel in (
        (axes[0], "mae", "Mean absolute error", "MAE (dataset units)"),
        (axes[1], "rmse", "Root mean squared error", "RMSE (dataset units)"),
        (axes[2], "r2", "Coefficient of determination", "R²"),
    ):
        axis.bar(models, test[column], color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Model comparison on the held-out December 2016 test period")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance: pd.DataFrame, destination: Path, top_n: int = 15
) -> None:
    top = importance.sort_values("importance_mean", ascending=False).head(top_n)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(
        top["feature"][::-1],
        top["importance_mean"][::-1],
        xerr=top["importance_std"][::-1],
        color="#55A868",
        alpha=0.9,
    )
    axis.set_xlabel("Increase in MAE when feature is permuted (dataset units)")
    axis.set_title("Permutation importance — selected model (test sample)")
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
