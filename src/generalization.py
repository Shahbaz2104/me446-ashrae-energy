"""Secondary evaluation: generalization to buildings not seen during training.

The primary chronological split keeps every building in training, so it cannot
answer whether the model transfers to a new building. This experiment holds out
20% of buildings (stratified by site) entirely, retrains the selected model
family, and compares errors on seen and unseen buildings in the same test month.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config, evaluation
from src.train_models import build_hist_gradient_boosting, load_dataset


def select_holdout_buildings(frame: pd.DataFrame, fraction: float = 0.20) -> list[int]:
    buildings = frame[["building_id", "site_id"]].drop_duplicates()
    rng = np.random.default_rng(config.RANDOM_SEED)
    held_out: list[int] = []
    for _, group in buildings.groupby("site_id"):
        n_holdout = max(1, int(round(len(group) * fraction)))
        n_holdout = min(n_holdout, len(group) - 1) if len(group) > 1 else 1
        chosen = rng.choice(group["building_id"].to_numpy(), size=n_holdout, replace=False)
        held_out.extend(int(value) for value in chosen)
    return sorted(set(held_out))


def main() -> None:
    config.ensure_directories()
    frame = load_dataset()
    holdout = select_holdout_buildings(frame)
    is_holdout = frame["building_id"].isin(holdout)

    train = frame[(frame["split"] == "train") & (~is_holdout)]
    validation = frame[(frame["split"] == "validation") & (~is_holdout)]
    test_unseen = frame[(frame["split"] == "test") & is_holdout]
    test_seen = frame[(frame["split"] == "test") & (~is_holdout)]

    model = build_hist_gradient_boosting()
    model.fit(train[config.FEATURES], train[config.TARGET])

    results = []
    for group_name, subset in (
        ("unseen_buildings_test", test_unseen),
        ("seen_buildings_test", test_seen),
        ("seen_buildings_validation", validation),
    ):
        prediction = np.clip(
            model.predict(subset[config.FEATURES]), 0, None
        )
        metrics = evaluation.regression_metrics(subset[config.TARGET], prediction)
        results.append({"building_group": group_name, **metrics})

    table = pd.DataFrame(results)
    table.to_csv(config.TABLES_DIR / "generalization_building_holdout.csv", index=False)
    (config.TABLES_DIR / "generalization_building_holdout.md").write_text(
        table.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8"
    )

    unseen_prediction = np.clip(
        model.predict(test_unseen[config.FEATURES]), 0, None
    )
    evaluation.plot_actual_vs_predicted(
        test_unseen[config.TARGET],
        unseen_prediction,
        "HGB trained without these buildings",
        config.FIGURES_DIR / "unseen_buildings_actual_vs_predicted.png",
    )

    metadata = {
        "protocol": (
            "20% of buildings held out entirely, stratified by site (seed 42). "
            "Model trained on Jan-Sep 2016 of remaining buildings and evaluated on "
            "December 2016 for held-out and seen buildings."
        ),
        "n_holdout_buildings": len(holdout),
        "n_training_rows": int(len(train)),
        "n_unseen_test_rows": int(len(test_unseen)),
        "n_seen_test_rows": int(len(test_seen)),
        "holdout_building_ids": holdout,
    }
    (config.TABLES_DIR / "generalization_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
