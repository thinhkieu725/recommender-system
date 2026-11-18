import argparse
from pathlib import Path
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
    #
    ratings_train_valid = pd.concat([ratings_train, ratings_validation], ignore_index=True)

    # Most-popular baseline
    # Since the most popular items do not need tuning and validation, both training and validation sets are used to compute the ranking.
    # most_rated_ranking = p1.most_rated_baseline(ratings_train_vaild, movies)
    most_positive_ranking = p1.most_positively_rated(ratings_train_valid, movies)
    #
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
    # Multi-hot genres encoding and One-hot decades encoding
    features = p2.encoding_movie_features(movies)

    # Adding weight for decade feature
    decade_weight = 2
    for column in features.columns:
        if column.startswith("decade_"):
            features[column] = features[column] * decade_weight

    # Evaluate recommended movies to validation users using full history
    valid_users = ratings_validation["user_id"].unique()
    content_based_recommendations = p2.recommend_for_users(
        user_ids=test_users,
        ratings_history=ratings_train_valid,
        movie_features=features,
        top_n=10,
    )
    # Evaluate the full-history recommendations against test ratings
    # print("\n========== Content-based test ==========")
    # p1.metric_evaluation(
    #     predictions=content_based_recommendations,
    #     ground_truth=ratings_test,
    #     catalogue=movies["movie_id"],
    #     k=10,
    #     relevance_threshold=4,
    # )

    # Variability report
    # paired_comparison = p2.compare_test_ndcg(content_based_recommendations,
    #                      most_positive_recommendations,
    #                      ratings_test
    #                     )
    # print(paired_comparison)

    # Disaggregated results - low vs high activity users
    # history_counts = (
    #     ratings_train_valid.groupby("user_id")
    #     .size()
    #     .reindex(test_users, fill_value=0)
    # )
    #
    # # Define the cut-off point using median
    # median_count = history_counts[history_counts > 0].median()
    # print(f"User activity cutoff: {median_count} historical ratings")
    #
    # activity_groups = {
    #     "No history": history_counts == 0,
    #     "Lower activity": (history_counts > 0) & (history_counts <= median_count),
    #     "Higher activity": history_counts > median_count,
    # }
    #
    # for label, group_mask in activity_groups.items():
    #     group_users = history_counts.index[group_mask]
    #     if len(group_users) == 0:
    #         continue
    #
    #     test_slice = ratings_test[ratings_test["user_id"].isin(group_users)]
    #     p2.evaluate_test_slice(
    #         label,
    #         test_slice,
    #         content_based_recommendations,
    #         movies["movie_id"]
    #     )
    #
    # # Disaggregated results - early vs test period
    # sorted_timestamps = ratings_test["timestamp"].sort_values()
    #
    # # Define cut-off point using timestamp
    # middle_timestamp = sorted_timestamps.iloc[len(sorted_timestamps) // 2]
    #
    # test_periods = {
    #     "Early test": ratings_test[ratings_test["timestamp"] < middle_timestamp],
    #     "Late test": ratings_test[ratings_test["timestamp"] >= middle_timestamp],
    # }
    #
    # for label, test_slice in test_periods.items():
    #     p2.evaluate_test_slice(
    #         label,
    #         test_slice,
    #         content_based_recommendations,
    #         movies["movie_id"]
    #     )

if __name__ == "__main__":
    main()
