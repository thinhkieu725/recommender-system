import argparse
from pathlib import Path
import pandas as pd
import part_1.part_1 as p1

from part_2.markov import (
    fit_markov_models,
    recommend_markov,
)
from part_2.markov_tuning import save_tuning_results, tune_markov_hyperparameters

from part_2.content_based import (
    evaluate_by_user_activity,
    evaluate_by_test_period,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommender system project full run on MovieLens 1M dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing ratings.dat, movies.dat and users.dat",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- PART 1: Data, protocol, baselines ---
    p1.check_data_dir(args.data_dir)
    users, movies, ratings = p1.load_data(args.data_dir)
    # p1.characterize_data(users, movies, ratings)

    ratings_train, ratings_validation, ratings_test = p1.split_data_temporally(ratings)
    item_based_knn_model = p1.item_based_knn_baseline(ratings_train, ratings_validation, movies)

    ratings_train_valid = pd.concat([ratings_train, ratings_validation], ignore_index=True)
    # Most-popular baseline
    # Since the most popular items do not need tuning and validation, both training and validation sets are used to compute the ranking.
    most_rated_ranking = p1.most_rated_baseline(ratings_train_valid, movies)
    most_positive_ranking = p1.most_positively_rated(ratings_train_valid, movies)


    test_users = ratings_test["user_id"].unique()
    # Output to pass to evaluate
    knn_recommendations = p1.item_based_knn_recommend(
        model=item_based_knn_model,
        ratings_history=ratings_train_valid,
        user_ids=test_users,
        catalogue=movies["movie_id"],
        top_n=10,
    )
    most_rated_recommendations = p1.recommend_popular(
        ranking=most_rated_ranking,
        ratings_history=ratings_train_valid,
        user_ids=test_users,
        top_n=10,
    )
    most_positive_recommendations = p1.recommend_popular(
        ranking=most_positive_ranking,
        ratings_history=ratings_train_valid,
        user_ids=test_users,
        top_n=10,
    )

    # Evaluation
    p1.run_final_evaluation(
        knn_recommendations,
        most_rated_recommendations,
        most_positive_recommendations,
        ratings_test,
        movies["movie_id"],
        ratings_history=ratings_train_valid,
        k=10,
        threshold=4
    )

    #============= MARKOV CHAIN EVALUATION =============

    print("\n============= BEGIN OF MARKOV CHAIN EVALUATION =============")
    """
    old markov model without tuning, for experiment

    markov_models = fit_markov_models(
        ratings_train_valid,
        weight_method="linear",
        weighting="destination",
        group_same_timestamp=True,
    )

    markov_recommendations = recommend_markov(
        models=markov_models,
        ratings_history=ratings_train_valid,
        user_ids=test_users,
        catalogue=movies["movie_id"],
        order=2,
        history_mode="full",
        top_n=10,
        score_weighting="rating",
        fallback_ranking=most_positive_ranking,
    )
    """
    # Markov model with hyperparameter tuning
    tuning_results_path = Path("markov_tuning_results.csv")

    if tuning_results_path.exists():
        tuning_results = pd.read_csv(tuning_results_path)
        best_trial = tuning_results.loc[
            tuning_results["validation_ndcg_at_10"].idxmax()
        ]

        best_config = {
            "order": int(best_trial["order"]),
            "history_mode": best_trial["history_mode"],
            "recent_n": (
                None if pd.isna(best_trial["recent_n"])
                else int(best_trial["recent_n"])
            ),
            "weighting": best_trial["weighting"],
            "weight_method": best_trial["weight_method"],
            "score_weighting": best_trial["score_weighting"],
        }

        best_score = best_trial["validation_ndcg_at_10"]

    else:


        best_config, best_score, tuning_results = tune_markov_hyperparameters(
            ratings_train=ratings_train,
            ratings_validation=ratings_validation,
            catalogue=movies["movie_id"],
            trial_budget=30,
            random_state=42,
        )
        save_tuning_results(tuning_results, "markov_tuning_results.csv")

    markov_models = fit_markov_models(
        ratings_train_valid,
        weight_method=best_config["weight_method"],
        weighting=best_config["weighting"],
        group_same_timestamp=True,
    )

    markov_recommendations = recommend_markov(
        models=markov_models,
        ratings_history=ratings_train_valid,
        user_ids=ratings_test["user_id"].unique(),
        catalogue=movies["movie_id"],
        order=best_config["order"],
        history_mode=best_config["history_mode"],
        recent_n=best_config["recent_n"] or 5,
        top_n=10,
        score_weighting=best_config["score_weighting"],
        fallback_ranking=None,
    )

    from part_2.utils import run_final_variablity

    # After generating markov_recommendations and the three baseline recommendations:
    print("\n========== Markov variability report ==========")

    run_final_variablity(
        model_prediction=markov_recommendations,
        kNN_result=knn_recommendations,
        most_rated_result=most_rated_recommendations,
        most_positive_result=most_positive_recommendations,
        ground_truth=ratings_test,
    )

    # Disaggregated evaluation 

    print("\n========== MARKOV: USER ACTIVITY EVALUATION ==========")

    evaluate_by_user_activity(
        ratings_history=ratings_train_valid,
        ratings_test=ratings_test,
        model_recommendations=markov_recommendations,
        catalogue=movies["movie_id"],
    )

    print("\n========== MARKOV: TEST PERIOD EVALUATION ==========")

    evaluate_by_test_period(
        ratings_test=ratings_test,
        model_recommendations=markov_recommendations,
        catalogue=movies["movie_id"],
    )

    print("\n========== Markov model evaluation ==========")
    p1.metric_evaluation(
        predictions=markov_recommendations,
        ground_truth=ratings_test,
        catalogue=movies["movie_id"],
        k=10,
        relevance_threshold=4,
    )

    from part_2.markov_ablatation import markov_sparsity_test

    print("\n========== MARKOV: SPARSITY TEST ==========")
    sparsity_results = markov_sparsity_test(
        ratings_train_valid=ratings_train_valid,
        ratings_test=ratings_test,
        catalogue=movies["movie_id"].unique(),
        best_config=best_config,
        seed=42,
    )

    # ========== END OF MARKOV CHAIN EVALUATION ==========

if __name__ == "__main__":
    main()
