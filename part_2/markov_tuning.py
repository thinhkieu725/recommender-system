"""Validation-only random search for the Markov recommender.

Each trial fits on ``ratings_train`` and uses that same data as recommendation
history. Validation interactions are used only as ground truth. The test set is
intentionally not accepted by this module.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from itertools import product
from io import StringIO
import random
from typing import Any

import pandas as pd

from part_1.part_1 import metric_evaluation
from part_2.markov import fit_markov_models, recommend_markov


_SEARCH_SPACE = {
    "order": (1, 2),
    "history_mode": ("full", "recent"),
    "recent_n": (3, 5, 10),
    "weighting": ("none", "destination", "both"),
    "weight_method": ("linear", "exponential", "positive"),
    "score_weighting": ("none", "rating"),
}


def _valid_configurations() -> list[dict[str, Any]]:
    """Enumerate the valid space, omitting irrelevant full-history recent_n values."""
    configurations = []
    for order, history_mode, recent_n, weighting, weight_method, score_weighting in product(
        _SEARCH_SPACE["order"],
        _SEARCH_SPACE["history_mode"],
        _SEARCH_SPACE["recent_n"],
        _SEARCH_SPACE["weighting"],
        _SEARCH_SPACE["weight_method"],
        _SEARCH_SPACE["score_weighting"],
    ):
        if history_mode == "full" and recent_n != _SEARCH_SPACE["recent_n"][0]:
            continue
        configurations.append(
            {
                "order": order,
                "history_mode": history_mode,
                "recent_n": recent_n if history_mode == "recent" else None,
                "weighting": weighting,
                "weight_method": weight_method,
                "score_weighting": score_weighting,
            }
        )
    return configurations


def tune_markov_hyperparameters(
    ratings_train: pd.DataFrame,
    ratings_validation: pd.DataFrame,
    catalogue: pd.Series,
    trial_budget: int = 30,
    random_state: int = 42,
    top_n: int = 10,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    """Random-search Markov settings using training data and validation NDCG only.

    The evaluated population is fixed to validation users who both occur in
    training and have at least one validation rating. The project evaluator
    assigns a score of zero to users with no recommendations, so they remain
    in the macro average. Users whose validation ratings are all below the
    evaluator's relevance threshold are excluded by that evaluator for every
    configuration.

    ``group_same_timestamp=True``, ``top_n``, and no fallback are fixed across
    trials. ``recent_n`` is recorded as ``None`` for full-history trials.
    """
    if trial_budget < 1:
        raise ValueError("trial_budget must be at least 1.")
    if top_n != 10:
        raise ValueError("This tuning protocol fixes top_n=10.")
    if ratings_train.empty:
        raise ValueError("ratings_train must not be empty.")
    if ratings_validation.empty:
        raise ValueError("ratings_validation must not be empty.")

    candidates = _valid_configurations()
    rng = random.Random(random_state)
    rng.shuffle(candidates)
    selected = candidates[: min(trial_budget, len(candidates))]

    warm_users = set(ratings_train["user_id"])
    evaluation_users = [
        user_id
        for user_id in ratings_validation["user_id"].drop_duplicates()
        if user_id in warm_users
    ]
    validation_truth = ratings_validation[
        ratings_validation["user_id"].isin(evaluation_users)
    ]
    if validation_truth.empty:
        raise ValueError("No warm users have validation interactions to evaluate.")

    records: list[dict[str, Any]] = []
    for trial, config in enumerate(selected, start=1):
        model = fit_markov_models(
            ratings_train,
            weight_method=config["weight_method"],
            weighting=config["weighting"],
            group_same_timestamp=True,
        )
        recommendations = recommend_markov(
            models=model,
            ratings_history=ratings_train,
            user_ids=evaluation_users,
            catalogue=catalogue,
            order=config["order"],
            history_mode=config["history_mode"],
            # recent_n is immaterial in full mode; pass the implementation default.
            recent_n=config["recent_n"] or 5,
            top_n=top_n,
            score_weighting=config["score_weighting"],
            fallback_ranking=None,
        )
        # The existing evaluator returns the same graded NDCG used below and
        # prints per-call output; capture that output to keep tuning concise.
        with redirect_stdout(StringIO()):
            metrics = metric_evaluation(
                predictions=recommendations,
                ground_truth=validation_truth,
                catalogue=catalogue,
                k=top_n,
                relevance_threshold=4,
            )
        records.append(
            {
                "trial": trial,
                **config,
                "validation_ndcg_at_10": float(metrics[f"NDCG@{top_n}"]),
            }
        )

    results = pd.DataFrame(records).sort_values(
        ["validation_ndcg_at_10", "trial"],
        ascending=[False, True],
        ignore_index=True,
    )
    best_row = results.iloc[0]
    best_config = {
        key: best_row[key]
        for key in (
            "order",
            "history_mode",
            "recent_n",
            "weighting",
            "weight_method",
            "score_weighting",
        )
    }
    # Convert NumPy scalar types from DataFrame rows to ordinary Python values.
    best_config["order"] = int(best_config["order"])
    if pd.isna(best_config["recent_n"]):
        best_config["recent_n"] = None
    else:
        best_config["recent_n"] = int(best_config["recent_n"])
    best_score = float(best_row["validation_ndcg_at_10"])

    print(f"Markov configurations evaluated: {len(results)} / budget {trial_budget}")
    print(f"Best validation NDCG@{top_n}: {best_score:.6f}")
    print(f"Best configuration: {best_config}")

    return best_config, best_score, results


def save_tuning_results(results: pd.DataFrame, path: str) -> None:
    """Save the trial table as a CSV file."""
    results.to_csv(path, index=False)
