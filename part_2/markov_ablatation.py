
import pandas as pd

from part_1.part_1 import metric_evaluation
from part_2.markov import fit_markov_models, recommend_markov


def markov_sparsity_test(
    ratings_train_valid,
    ratings_test,
    catalogue,
    best_config,
    seed=42,
):
    """
    Evaluate the Markov recommender using 100%, 50%, and 20%
    of the available training interactions.

    The samples are nested and reproducible.
    Hyperparameters remain fixed across all sparsity levels.
    """

    # One shuffled ordering ensures nested samples:
    # 20% is contained in 50%, which is contained in 100%.
    shuffled_history = ratings_train_valid.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    test_users = ratings_test["user_id"].unique()

    results = []

    for fraction in (1.0, 0.50, 0.20):
        sample_size = int(len(shuffled_history) * fraction)

        sampled_history = shuffled_history.iloc[:sample_size].copy()

        print("\n" + "=" * 60)
        print(f"MARKOV SPARSITY TEST: {fraction:.0%} TRAINING DATA")
        print(f"Training interactions: {len(sampled_history)}")
        print("=" * 60)

        # Fit a fresh model on this subset.
        models = fit_markov_models(
            sampled_history,
            weight_method=best_config["weight_method"],
            weighting=best_config["weighting"],
            group_same_timestamp=True,
        )

        # Generate recommendations using exactly the same
        # subset as the user's available history.
        recommendations = recommend_markov(
            models=models,
            ratings_history=sampled_history,
            user_ids=test_users,
            catalogue=catalogue,
            order=best_config["order"],
            history_mode=best_config["history_mode"],
            recent_n=best_config["recent_n"] or 5,
            top_n=10,
            score_weighting=best_config["score_weighting"],
            fallback_ranking=None,
        )

        metrics = metric_evaluation(
            predictions=recommendations,
            ground_truth=ratings_test,
            catalogue=catalogue,
            k=10,
            relevance_threshold=4,
        )

        results.append({
            "training_fraction": fraction,
            "training_interactions": len(sampled_history),
            "metrics": metrics,
        })

    return results
