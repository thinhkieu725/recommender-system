import argparse
from pathlib import Path
from part_2.utils import compare_test_ndcg
import pandas as pd
import part_1.part_1 as p1
import part_2.content_based as p2

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
    # item_based_knn_model = p1.item_based_knn_baseline(ratings_train, ratings_validation, movies)

    ratings_train_valid = pd.concat([ratings_train, ratings_validation], ignore_index=True)
    # Most-popular baseline
    # Since the most popular items do not need tuning and validation, both training and validation sets are used to compute the ranking.
    # most_rated_ranking = p1.most_rated_baseline(ratings_train_vaild, movies)
    most_positive_ranking = p1.most_positively_rated(ratings_train_valid, movies)

    test_users = ratings_test["user_id"].unique()
    # Output to pass to evaluate
    # knn_recommendations = p1.item_based_knn_recommend(
    #     model=item_based_knn_model,
    #     ratings_history=ratings_train_vaild,
    #     user_ids=test_users,
    #     catalogue=movies["movie_id"],
    #     top_n=10,
    # )
    # most_rated_recommendations = p1.recommend_popular(
    #     ranking=most_rated_ranking,
    #     ratings_history=ratings_train_vaild,
    #     user_ids=test_users,
    #     top_n=10
    # )
    most_positive_recommendations = p1.recommend_popular(
        ranking=most_positive_ranking,
        ratings_history=ratings_train_valid,
        user_ids=test_users,
        top_n=10,
    )

    # Evaluation
    # p1.run_final_evaluation(
    #     knn_recommendations,
    #     most_rated_recommendations,
    #     most_positive_recommendations,
    #     ratings_test,
    #     movies["movie_id"],
    #     ratings_history=ratings_train_vaild,
    #     k=10,
    #     threshold=4
    # )

    # --- PART 2: Three models---

    # -- Content-based recommender --
    valid_users = ratings_validation["user_id"].unique()
    # Multi-hot genres encoding and One-hot decades encoding
    features = p2.encoding_movie_features(movies)

    # Evaluate recommended movies to validation users using full history
    content_based_recommendations = p2.recommend_for_users(
        user_ids=test_users,
        ratings_history=ratings_train_valid,
        movie_features=features,
        top_n=10,
    )
    # Evaluate the full-history recommendations against test ratings
    print("\n========== Content-based final ==========")
    p1.metric_evaluation(
        predictions=content_based_recommendations,
        ground_truth=ratings_test,
        catalogue=movies["movie_id"],
        k=10,
        relevance_threshold=4,
    )

    # Variability report
    print("\n========== Variablity report ==========")
    paired_comparison = compare_test_ndcg(
        content_based_recommendations,
        most_positive_recommendations,
        ratings_test
    )
    print(paired_comparison)

    # Disaggregated results - low vs high activity users
    p2.evaluate_by_user_activity(
        ratings_history = ratings_train_valid,
        ratings_test = ratings_test,
        model_recommendations = content_based_recommendations,
        catalogue = movies["movie_id"]
    )
    p2.evaluate_by_test_period(
        ratings_test = ratings_test,
        model_recommendations = content_based_recommendations,
        catalogue = movies["movie_id"]
    )

    # Sparsity-ablation
    ablation_seed = 42
    shuffled_history = ratings_train_valid.sample(frac=1, random_state=ablation_seed)
    p2.sparsity_test(test_users = test_users,
                     shuffled_history= shuffled_history,
                     ground_truth = ratings_test,
                     catalogue = movies["movie_id"],
                     features = features
    )
if __name__ == "__main__":
    main()
