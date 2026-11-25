
import numpy as np
import pandas as pd
import part_1.part_1 as p1
from pathlib import Path
from part_2.markov import fit_markov_models, recommend_markov
from part_1.part_1 import metric_evaluation_warm_users
from part_3.mmr import rerank_mmr


# ==================================================
# 1. Experiment configuration
# ==================================================

CANDIDATE_POOL_SIZE = 100
RECOMMENDATION_SIZE = 10
K = 10

LAMBDA_VALUES = np.round(np.arange(0.0, 1.01, 0.1), 2)


# Best configuration from Part 2
MARKOV_CONFIG = {
    "weight_method": "linear",
    "weighting": "none",
    "group_same_timestamp": False,
}

RECOMMEND_CONFIG = {
    "order": 1,
    "history_mode": "full",
    "recent_n": 5,
    "score_weighting": "rating",
}


# ==================================================
# 2. Generate Markov candidate pool
# ==================================================

def generate_candidates(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:

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


# ==================================================
# 3. Run validation experiment
# ==================================================

def run_validation_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movies: pd.DataFrame,
):
    # Generate candidates once so every lambda uses
    # exactly the same candidate pool.
    candidates = generate_candidates(
        train=train,
        movies=movies,
    )

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
        recommendations = rerank_mmr(
            candidates=candidates,
            movies=movies,
            top_n=RECOMMENDATION_SIZE,
            lambda_=float(lambda_),
        )

        recommendations_by_lambda[float(lambda_)] = recommendations

        # Evaluate on warm users only.
        # Training history determines which users are warm.
        metrics = metric_evaluation_warm_users(
            predictions=recommendations,
            ground_truth=validation,
            ratings_history=train,
            catalogue=movies["movie_id"],
            k=K,
            relevance_threshold=4,
        )

        results.append({
            "lambda": float(lambda_),
            **metrics,
        })

    results_df = pd.DataFrame(results)

    return candidates, recommendations_by_lambda, results_df


# ==================================================
# 4. Run the experiment
# ==================================================

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

    missing_metadata = (
        set(candidates["movie_id"])
        - set(movies["movie_id"])
    )
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

    validation_results.to_csv(
        "part_3/validation_metrics.csv",
        index=False,
    )

    print("\nSaved validation metrics to part_3/validation_metrics.csv")
