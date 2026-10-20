"""Exploratory analysis of the documented local electricity subset."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
SCOPE_LABEL = "site-stratified subset (120 buildings, 16 sites, 2016)"


def load_eda_frame() -> pd.DataFrame:
    columns = [
        "timestamp",
        "meter_reading",
        "building_id",
        "site_id",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "air_temperature",
        "dew_temperature",
        "wind_speed",
        "cloud_coverage",
        "precip_depth_1_hr",
        "sea_level_pressure",
    ]
    frame = pd.read_parquet(config.MODEL_DATASET_PARQUET, columns=columns)
    frame["primary_use"] = frame["primary_use"].astype("category")
    return frame


def figure_distribution(frame: pd.DataFrame) -> dict:
    reading = frame["meter_reading"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    clipped = reading.clip(upper=reading.quantile(0.999))
    axes[0].hist(clipped, bins=70, color="#4C72B0", alpha=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Electricity reading (dataset units)")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[0].set_title("Linear scale, clipped at 99.9th percentile")
    axes[0].axvline(reading.median(), color="#C44E52", linestyle="--", linewidth=1)
    axes[0].annotate(
        f"median = {reading.median():.1f}",
        xy=(reading.median(), axes[0].get_ylim()[1] * 0.5),
        xytext=(reading.median() * 1.25, axes[0].get_ylim()[1] * 0.5),
        fontsize=8,
        color="#C44E52",
    )

    axes[1].hist(np.log1p(reading), bins=70, color="#55A868", alpha=0.9)
    axes[1].set_xlabel("log(1 + electricity reading)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Log scale highlights the long right tail")
    zero_share = (reading == 0).mean() * 100
    fig.suptitle(f"Electricity reading distribution — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "reading_distribution.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "median": float(reading.median()),
        "mean": float(reading.mean()),
        "p95": float(reading.quantile(0.95)),
        "p99": float(reading.quantile(0.99)),
        "max": float(reading.max()),
        "zero_pct": float(zero_share),
    }


def figure_temporal(frame: pd.DataFrame) -> dict:
    hourly = (
        frame.groupby(["hour", "is_weekend"], observed=True)["meter_reading"]
        .mean()
        .reset_index()
    )
    daily = frame.set_index("timestamp")["meter_reading"].resample("D").mean()
    rolling = daily.rolling(7, min_periods=1).mean()
    monthly = frame.groupby("month", observed=True)["meter_reading"].mean()
    monthly_temp = frame.groupby("month", observed=True)["air_temperature"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for weekend, label, color in ((0, "Weekday", "#4C72B0"), (1, "Weekend", "#DD8452")):
        subset = hourly[hourly["is_weekend"] == weekend]
        axes[0, 0].plot(subset["hour"], subset["meter_reading"], marker="o", label=label, color=color)
    axes[0, 0].set_title("Mean reading by hour of day")
    axes[0, 0].set_xlabel("Hour of day")
    axes[0, 0].set_ylabel("Mean reading (dataset units)")
    axes[0, 0].set_xticks(range(0, 24, 3))
    axes[0, 0].legend()

    dow_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    by_dow = frame.groupby("dayofweek", observed=True)["meter_reading"].mean()
    axes[0, 1].bar([dow_labels[i] for i in by_dow.index], by_dow.to_numpy(), color="#55A868")
    axes[0, 1].set_title("Mean reading by day of week")
    axes[0, 1].set_ylabel("Mean reading (dataset units)")

    axes[1, 0].plot(daily.index, daily.to_numpy(), color="#B0B0B0", linewidth=0.7, label="Daily mean")
    axes[1, 0].plot(rolling.index, rolling.to_numpy(), color="#C44E52", linewidth=1.6, label="7-day rolling mean")
    axes[1, 0].set_title("Daily mean electricity reading through 2016")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Mean reading (dataset units)")
    axes[1, 0].legend(fontsize=8)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axis = axes[1, 1]
    axis.bar([month_names[i - 1] for i in monthly.index], monthly.to_numpy(), color="#8172B3", alpha=0.85)
    axis.set_title("Monthly mean reading and outdoor temperature")
    axis.set_xlabel("Month")
    axis.set_ylabel("Mean reading (dataset units)", color="#8172B3")
    axis.tick_params(axis="y", labelcolor="#8172B3")
    twin = axis.twinx()
    twin.plot([month_names[i - 1] for i in monthly_temp.index], monthly_temp.to_numpy(), color="#C44E52", marker="o")
    twin.set_ylabel("Mean air temperature (°C)", color="#C44E52")
    twin.tick_params(axis="y", labelcolor="#C44E52")
    twin.spines["right"].set_visible(True)
    twin.grid(False)

    fig.suptitle(f"Temporal consumption patterns — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "temporal_patterns.png", bbox_inches="tight")
    plt.close(fig)

    weekday_mean = hourly[hourly["is_weekend"] == 0]["meter_reading"].mean()
    weekend_mean = hourly[hourly["is_weekend"] == 1]["meter_reading"].mean()
    peak_hour = int(hourly[hourly["is_weekend"] == 0].sort_values("meter_reading").iloc[-1]["hour"])
    trough_hour = int(hourly[hourly["is_weekend"] == 0].sort_values("meter_reading").iloc[0]["hour"])
    return {
        "weekday_hourly_mean": float(weekday_mean),
        "weekend_hourly_mean": float(weekend_mean),
        "weekday_to_weekend_ratio": float(weekday_mean / weekend_mean),
        "peak_weekday_hour": peak_hour,
        "trough_weekday_hour": trough_hour,
        "monthly_mean": {month_names[i - 1]: float(value) for i, value in monthly.items()},
        "monthly_temperature": {
            month_names[i - 1]: float(value) for i, value in monthly_temp.items()
        },
        "daily_max": {"date": str(daily.idxmax().date()), "value": float(daily.max())},
        "daily_min": {"date": str(daily.idxmin().date()), "value": float(daily.min())},
    }


def figure_temperature(frame: pd.DataFrame) -> dict:
    sample = frame.dropna(subset=["air_temperature", "meter_reading"]).sample(
        min(120_000, len(frame)), random_state=config.RANDOM_SEED
    )
    binned = frame.dropna(subset=["air_temperature"]).copy()
    binned["temp_bin"] = (binned["air_temperature"] / 2).round() * 2
    grouped = (
        binned.groupby("temp_bin")["meter_reading"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    quantiles = (
        binned.groupby("temp_bin")["meter_reading"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q25", 0.75: "q75"})
        .reset_index()
    )
    grouped = grouped.merge(quantiles, on="temp_bin")
    grouped = grouped[grouped["count"] >= 100]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hexbin(
        sample["air_temperature"],
        np.log1p(sample["meter_reading"]),
        gridsize=45,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    axes[0].set_xlabel("Outdoor air temperature (°C)")
    axes[0].set_ylabel("log(1 + electricity reading)")
    axes[0].set_title("Building-hour observations (hexbin density)")
    correlation = sample["air_temperature"].corr(sample["meter_reading"])
    axes[0].annotate(f"Pearson r = {correlation:.2f}", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=8)

    axes[1].fill_between(grouped["temp_bin"], grouped["q25"], grouped["q75"], color="#4C72B0", alpha=0.2, label="Interquartile range")
    axes[1].plot(grouped["temp_bin"], grouped["mean"], color="#4C72B0", marker="o", markersize=3, label="Mean")
    axes[1].plot(grouped["temp_bin"], grouped["median"], color="#DD8452", linestyle="--", label="Median")
    axes[1].set_xlabel("Outdoor air temperature (°C, 2°C bins)")
    axes[1].set_ylabel("Electricity reading (dataset units)")
    axes[1].set_title("Binned reading versus temperature")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Electricity consumption and weather — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "temperature_relationship.png", bbox_inches="tight")
    plt.close(fig)

    cold = frame[frame["air_temperature"] < 5]["meter_reading"].mean()
    mild = frame[(frame["air_temperature"] >= 15) & (frame["air_temperature"] < 25)]["meter_reading"].mean()
    hot = frame[frame["air_temperature"] >= 30]["meter_reading"].mean()
    return {
        "pearson_r": float(correlation),
        "mean_reading_below_5C": float(cold),
        "mean_reading_15_to_25C": float(mild),
        "mean_reading_above_30C": float(hot),
        "binned": grouped[["temp_bin", "mean", "median", "count"]].to_dict(orient="records"),
    }


def figure_primary_use(frame: pd.DataFrame) -> dict:
    summary = (
        frame.groupby("primary_use", observed=True)["meter_reading"]
        .agg(["mean", "median", "count"])
        .sort_values("median")
    )
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.barh(summary.index.astype(str), summary["median"], color="#4C72B0", alpha=0.9)
    axis.scatter(summary["mean"], np.arange(len(summary)), color="#C44E52", zorder=3, label="Mean")
    for index, (_, row) in enumerate(summary.iterrows()):
        axis.text(row["median"] + summary["median"].max() * 0.01, index, f"n={int(row['count']):,}", va="center", fontsize=7)
    axis.set_xlabel("Electricity reading (dataset units)")
    axis.set_title("Median and mean electricity reading by primary use")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(axis="y", visible=False)
    fig.suptitle(f"Building category comparison — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "primary_use_consumption.png", bbox_inches="tight")
    plt.close(fig)
    return {
        str(index): {"median": float(row["median"]), "mean": float(row["mean"]), "rows": int(row["count"])}
        for index, row in summary.iterrows()
    }


def figure_building_size(frame: pd.DataFrame) -> dict:
    building = (
        frame.groupby(["building_id", "site_id", "primary_use"], observed=True)
        .agg(mean_reading=("meter_reading", "mean"), square_feet=("square_feet", "first"))
        .reset_index()
    )
    building = building[(building["mean_reading"] > 0) & (building["square_feet"] > 0)]
    top_uses = building["primary_use"].value_counts().head(4).index.tolist()
    fig, axis = plt.subplots(figsize=(8.2, 5))
    for index, use in enumerate(top_uses):
        subset = building[building["primary_use"] == use]
        axis.scatter(
            subset["square_feet"],
            subset["mean_reading"],
            s=18,
            alpha=0.8,
            label=str(use),
            color=PALETTE[index % len(PALETTE)],
        )
    other = building[~building["primary_use"].isin(top_uses)]
    axis.scatter(other["square_feet"], other["mean_reading"], s=14, alpha=0.5, label="Other uses", color="#B0B0B0")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Building floor area (square feet, log scale)")
    axis.set_ylabel("Mean electricity reading per hour (dataset units, log scale)")
    axis.set_title("Building size versus mean electricity demand")
    axis.legend(fontsize=7, loc="upper left")
    log_corr = np.corrcoef(np.log10(building["square_feet"]), np.log10(building["mean_reading"]))[0, 1]
    axis.annotate(f"Pearson r on log scales = {log_corr:.2f}", xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8)
    fig.suptitle(f"Building size relationship — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "building_size_relationship.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "log10_pearson_r": float(log_corr),
        "n_buildings": int(len(building)),
        "median_square_feet": float(building["square_feet"].median()),
    }


def figure_correlation(frame: pd.DataFrame) -> dict:
    columns = {
        "meter_reading": "Reading",
        "air_temperature": "Air temp",
        "dew_temperature": "Dew temp",
        "wind_speed": "Wind speed",
        "cloud_coverage": "Cloud cover",
        "precip_depth_1_hr": "Precipitation",
        "sea_level_pressure": "Pressure",
        "square_feet": "Floor area",
        "year_built": "Year built",
        "floor_count": "Floors",
        "hour": "Hour",
        "month": "Month",
    }
    sample = frame[list(columns)].sample(min(300_000, len(frame)), random_state=config.RANDOM_SEED)
    correlation = sample.corr(numeric_only=True).rename(index=columns, columns=columns)
    fig, axis = plt.subplots(figsize=(8.5, 7))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            value = correlation.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 else "#222222",
            )
    fig.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    axis.set_title(f"Correlation among available numerical variables ({SCOPE_LABEL})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "correlation_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return {
        str(columns[column]): {
            "reading_r": float(correlation.loc["Reading", columns[column]]),
            "air_temp_r": float(correlation.loc["Air temp", columns[column]]),
        }
        for column in columns
        if columns[column] in correlation.columns
    }


def figure_site(frame: pd.DataFrame) -> dict:
    site = (
        frame.groupby("site_id", observed=True)
        .agg(
            mean_reading=("meter_reading", "mean"),
            median_reading=("meter_reading", "median"),
            mean_temp=("air_temperature", "mean"),
        )
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(site["site_id"].astype(str), site["mean_reading"], color="#55A868", alpha=0.85)
    axis.set_xlabel("Site ID")
    axis.set_ylabel("Mean electricity reading (dataset units)", color="#55A868")
    axis.tick_params(axis="y", labelcolor="#55A868")
    twin = axis.twinx()
    twin.plot(site["site_id"].astype(str), site["mean_temp"], color="#C44E52", marker="o")
    twin.set_ylabel("Mean air temperature (°C)", color="#C44E52")
    twin.tick_params(axis="y", labelcolor="#C44E52")
    twin.spines["right"].set_visible(True)
    twin.grid(False)
    axis.set_title(f"Site-level electricity demand and climate — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "site_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return {
        int(row["site_id"]): {
            "mean_reading": float(row["mean_reading"]),
            "median_reading": float(row["median_reading"]),
            "mean_temp": float(row["mean_temp"]),
        }
        for _, row in site.iterrows()
    }


def figure_temperature_by_site(frame: pd.DataFrame) -> dict:
    working = frame.dropna(subset=["air_temperature"]).copy()
    working["temp_band"] = pd.cut(
        working["air_temperature"],
        bins=[-np.inf, 10, 25, np.inf],
        labels=["<10°C", "10–25°C", ">25°C"],
    )
    summary = (
        working.groupby(["site_id", "temp_band"], observed=True)["meter_reading"]
        .mean()
        .unstack()
        .reindex(columns=["<10°C", "10–25°C", ">25°C"])
    )
    correlations = (
        working.groupby("site_id", observed=True)
        .apply(
            lambda group: group["air_temperature"].corr(group["meter_reading"]),
            include_groups=False,
        )
        .rename("pearson_r")
    )
    summary = summary.join(correlations)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    positions = np.arange(len(summary))
    width = 0.27
    for offset, band, color in zip((-width, 0, width), ["<10°C", "10–25°C", ">25°C"], ["#4C72B0", "#55A868", "#DD8452"]):
        axes[0].bar(positions + offset, summary[band], width=width, label=band, color=color)
    axes[0].set_xticks(positions, summary.index.astype(str))
    axes[0].set_xlabel("Site ID")
    axes[0].set_ylabel("Mean electricity reading (dataset units)")
    axes[0].set_title("Mean reading by outdoor temperature band and site")
    axes[0].legend(fontsize=8)

    ordered = summary.sort_values("pearson_r")
    colors = ["#C44E52" if value > 0 else "#4C72B0" for value in ordered["pearson_r"]]
    axes[1].barh(ordered.index.astype(str), ordered["pearson_r"], color=colors)
    axes[1].axvline(0, color="#444444", linewidth=1)
    axes[1].set_xlabel("Pearson r (air temperature vs reading)")
    axes[1].set_ylabel("Site ID")
    axes[1].set_title("Site-specific temperature correlation")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(f"Temperature response differs by site — {SCOPE_LABEL}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "temperature_response_by_site.png", bbox_inches="tight")
    plt.close(fig)
    return {
        int(site): {
            "pearson_r": float(row["pearson_r"]),
            "mean_reading_below_10C": float(row["<10°C"]) if pd.notna(row["<10°C"]) else None,
            "mean_reading_10_to_25C": float(row["10–25°C"]) if pd.notna(row["10–25°C"]) else None,
            "mean_reading_above_25C": float(row[">25°C"]) if pd.notna(row[">25°C"]) else None,
        }
        for site, row in summary.iterrows()
    }


def main() -> None:
    config.ensure_directories()
    frame = load_eda_frame()
    summary = {
        "scope": SCOPE_LABEL,
        "n_rows": int(len(frame)),
        "n_buildings": int(frame["building_id"].nunique()),
        "n_sites": int(frame["site_id"].nunique()),
        "distribution": figure_distribution(frame),
        "temporal": figure_temporal(frame),
        "temperature": figure_temperature(frame),
        "temperature_by_site": figure_temperature_by_site(frame),
        "primary_use": figure_primary_use(frame),
        "building_size": figure_building_size(frame),
        "correlation": figure_correlation(frame),
        "site": figure_site(frame),
    }
    (config.TABLES_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("n_rows", "distribution", "temporal", "temperature")}, indent=2)[:4000])
    print("EDA figures written to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
