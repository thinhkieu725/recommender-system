
import numpy as np
import pandas as pd
import part_1.part_1 as p1
from pathlib import Path
from part_2.markov import fit_markov_models, recommend_markov
from part_1.part_1 import metric_evaluation_warm_users
from part_3.mmr import rerank_mmr
from part_3.diversity_metrics import evaluate_diversity


# 1. Experiment configuration
CANDIDATE_POOL_SIZE = 100
RECOMMENDATION_SIZE = 10
K = 10

LAMBDA_VALUES = np.round(np.arange(0.0, 1.01, 0.1), 2)

# Best configuration from Part 2
MARKOV_CONFIG = {
    "weight_method": "linear",
    "weighting": "none",
    "group_same_timestamp": True,
}

RECOMMEND_CONFIG = {
    "order": 1,
    "history_mode": "full",
    "recent_n": 5, # not really needed 
    "score_weighting": "rating",
}

CACHE_DIR = Path("part_3/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 2. Generate Markov candidate pool
def generate_candidates(train: pd.DataFrame,movies: pd.DataFrame) -> pd.DataFrame:

    # Fit Markov model using training data only.
    models = fit_markov_models(
        ratings_train=train,
        **MARKOV_CONFIG,
    )

    # Only users with training history can be recommended to.
    user_ids = train["user_id"].unique().tolist()

    # Generate a larger candidate pool than the final list.
    candidates = recommend_markov(
        models=models,
        ratings_history=train,
        user_ids=user_ids,
        catalogue=movies["movie_id"],
        top_n=CANDIDATE_POOL_SIZE,
        fallback_ranking=None,
        **RECOMMEND_CONFIG,
    )

    return candidates

# 3. Run validation experiment
def run_validation_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movies: pd.DataFrame,
):
    # Generate candidates once so every lambda uses
    # exactly the same candidate pool.

    # generate candidate with cache to reduce runtime
    candidate_cache = Path("part_3/cache/validation_candidates.pkl")

    if candidate_cache.exists():
        print("Loading cached validation candidates...")
        candidates = pd.read_pickle(candidate_cache)

    else:
        print("Generating validation candidates...")
        candidates = generate_candidates(
            train=train,
            movies=movies,
        )

        candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_pickle(candidate_cache)

    print("\n========== CANDIDATE GENERATION ==========")
    print(f"Users: {candidates['user_id'].nunique()}")
    print(f"Candidate recommendations: {len(candidates)}")

    if candidates.empty:
        raise ValueError("The Markov model generated no candidates.")

    results = []
    recommendations_by_lambda = {}

    for lambda_ in LAMBDA_VALUES:

        print(f"\n========== LAMBDA = {lambda_:.1f} ==========")

        # Apply MMR to the candidate pool.
        recommendation_cache = Path(
            f"part_3/cache/validation_recommendations_lambda_{lambda_:.1f}.pkl"
        )

        if recommendation_cache.exists():
            print(f"Loading cached recommendations for lambda={lambda_:.1f}")
            recommendations = pd.read_pickle(recommendation_cache)

        else:
            print(f"No cache, generating recommendations for lambda={lambda_:.1f}")

            recommendations = rerank_mmr(
                candidates=candidates,
                movies=movies,
                top_n=RECOMMENDATION_SIZE,
                lambda_=float(lambda_),
            )

            recommendation_cache.parent.mkdir(parents=True, exist_ok=True)
            recommendations.to_pickle(recommendation_cache)

        recommendations_by_lambda[float(lambda_)] = recommendations

        # Evaluate on warm users only.
        # Training history determines which users are warm.
        # Cache the evaluation metrics for this lambda value.
        metrics_cache = Path(f"part_3/cache/validation_metrics_lambda_{lambda_:.1f}.pkl")

        if metrics_cache.exists():
            print(f"Loading cached metrics for lambda={lambda_:.1f}")
            result = pd.read_pickle(metrics_cache)
        else:
            metrics = metric_evaluation_warm_users(
                predictions=recommendations,
                ground_truth=validation,
                ratings_history=train,
                catalogue=movies["movie_id"],
                k=K,
                relevance_threshold=4,
            )

            diversity_metrics = evaluate_diversity(
                predictions=recommendations,
                ratings_history=train,
                movies=movies,
                catalogue=movies["movie_id"],
                k=K,
            )

            result = {
                "lambda": float(lambda_),
                **metrics,
                **diversity_metrics,
            }

            metrics_cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(result, metrics_cache)

        results.append(result)

    results_df = pd.DataFrame(results)

    return candidates, recommendations_by_lambda, results_df


def run_final_test_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    selected_lambda: float,
):
    # Combine training and validation history.
    train_valid = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    # Fit the Markov model using the same protocol
    # as the original final Markov experiment.
    models = fit_markov_models(
        ratings_train=train_valid,
        **MARKOV_CONFIG,
    )

    # Generate 100 candidates for the same test users
    # as the original Markov experiment.

    candidates = recommend_markov(
        models=models,
        ratings_history=train_valid,
        user_ids=test["user_id"].unique().tolist(),
        catalogue=movies["movie_id"],
        top_n=CANDIDATE_POOL_SIZE,
        fallback_ranking=None,
        **RECOMMEND_CONFIG,
    )

    # Original Markov top-10 recommendations.
    original_recommendations = recommend_markov(
        models=models,
        ratings_history=train_valid,
        user_ids=test["user_id"].unique().tolist(),
        catalogue=movies["movie_id"],
        top_n=RECOMMENDATION_SIZE,
        fallback_ranking=None,
        **RECOMMEND_CONFIG,
    )

    original_metrics = p1.metric_evaluation(
        predictions=original_recommendations,
        ground_truth=test,
        catalogue=movies["movie_id"],
        k=K,
        relevance_threshold=4,
    )

    # Apply MMR using the lambda selected on validation.
    recommendations = rerank_mmr(
        candidates=candidates,
        movies=movies,
        top_n=RECOMMENDATION_SIZE,
        lambda_=selected_lambda,
    )

    # Evaluate on the test set.
    metrics = p1.metric_evaluation(
        predictions=recommendations,
        ground_truth=test,
        catalogue=movies["movie_id"],
        k=K,
        relevance_threshold=4,
    )

    return (
        candidates,
        original_recommendations,
        recommendations,
        original_metrics,
        metrics,
    )


# 4. Run the experiment
import argparse

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

if __name__ == "__main__":

    # IF CHANGES ANYTHING MAJOR IN THE LOGIC, PLEASE REMOVE THE OLD CACHE!

    args = parse_args()

    # --- PART 1: Data, protocol, baselines ---
    p1.check_data_dir(args.data_dir)
    users, movies, ratings = p1.load_data(args.data_dir)
    # p1.characterize_data(users, movies, ratings)

    ratings_train, ratings_validation, ratings_test = p1.split_data_temporally(ratings)
    ratings_train_valid = pd.concat([ratings_train, ratings_validation], ignore_index=True)
    
    candidates, recommendations_by_lambda, validation_results = (
        run_validation_experiment(
            train=ratings_train,
            validation=ratings_validation,
            movies=movies,
        )
    )

    missing_metadata = (set(candidates["movie_id"]) - set(movies["movie_id"]))
    
    if missing_metadata:
        raise ValueError("Some candidate movies are missing genre metadata.")
    print(f"Candidates missing movie metadata: {len(missing_metadata)}")
    
    original_top_10 = (
        candidates
        .sort_values(["user_id", "rank"], kind="stable")
        .groupby("user_id", sort=False)
        .head(10)
        )

    mmr_top_10 = rerank_mmr(
        candidates=candidates,
        movies=movies,
        top_n=10,
        lambda_=1.0,
    )

    original_lists = (
        original_top_10.groupby("user_id")["movie_id"].apply(list).to_dict()
    )

    mmr_lists = (
        mmr_top_10.groupby("user_id")["movie_id"].apply(list).to_dict()
    )

    assert original_lists == mmr_lists, (
        "MMR with lambda=1 does not reproduce the original ranking."
    )

    print("Lambda=1 test passed!")
    print("\n========== VALIDATION RESULTS ==========")
    print(validation_results.to_string(index=False))

    Path("part_3").mkdir(parents=True, exist_ok=True)

    validation_results.to_csv("part_3/validation_metrics.csv", index=False)

    print("\nSaved validation metrics to part_3/validation_metrics.csv")

    # Select lambda using an accuracy constraint and a diversity objective.

    NDCG_TOLERANCE = 0.01  # Allow at most 1% relative loss in NDCG.

    best_ndcg = validation_results["NDCG@10"].max()
    minimum_acceptable_ndcg = best_ndcg * (1 - NDCG_TOLERANCE)

    acceptable_results = validation_results[
        validation_results["NDCG@10"] >= minimum_acceptable_ndcg
    ]

    # Among acceptable configurations, select the one with highest ILD.
    best_row = acceptable_results.sort_values(
        by=["ILD@10", "Calibration Error"],
        ascending=[False, True],
    ).iloc[0]

    selected_lambda = float(best_row["lambda"])

    print(f"\nBest validation NDCG: {best_ndcg:.6f}")
    print(f"Minimum acceptable NDCG: {minimum_acceptable_ndcg:.6f}")
    print("\nAcceptable lambda configurations:")
    print(
        acceptable_results[
            ["lambda", "NDCG@10", "ILD@10", "Calibration Error"]
        ].to_string(index=False)
    )

    print(f"\nSelected lambda: {selected_lambda}")

    print(f"\nSelected lambda: {selected_lambda}")
    print("\n========== FINAL MMR TEST EVALUATION ==========")

    (
        test_candidates,
        original_test_recommendations,
        test_recommendations,
        original_test_metrics,
        test_metrics,
    ) = run_final_test_experiment(
        train=ratings_train,
        validation=ratings_validation,
        test=ratings_test,
        movies=movies,
        selected_lambda=selected_lambda,
    )

    print("\nSelected MMR configuration:")
    print(f"Lambda: {selected_lambda}")
    print("\nOriginal Markov test metrics:")
    print(original_test_metrics)
    print("\nMMR test metrics:")
    print(test_metrics)
    test_recommendations.to_csv("part_3/test_recommendations.csv", index=False)