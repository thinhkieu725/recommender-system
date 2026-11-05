import argparse
from pathlib import Path
import pandas as pd
import part_1.part_1 as p1

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
    p1.characterize_data(users, movies, ratings)

    ratings_train, ratings_validation, ratings_test = p1.split_data_temporally(ratings)
    item_based_knn_model = p1.item_based_knn_baseline(ratings_train, ratings_validation, ratings_test)

    rating_columns = ["user_id", "movie_id", "rating"]
    test_set = list(ratings_test[rating_columns].itertuples(index=False, name=None))
    test_predictions = item_based_knn_model.test(test_set)

    test_predictions_df = pd.DataFrame(
    {
        "user_id": [p.uid for p in test_predictions],
        "movie_id": [p.iid for p in test_predictions],
        "rating": [p.r_ui for p in test_predictions],
        "predicted_rating": [p.est for p in test_predictions],
        "was_impossible": [p.details["was_impossible"] for p in test_predictions],
    }
    )
    test_rankings = test_predictions_df[~test_predictions_df["was_impossible"]].sort_values(
        ["user_id", "predicted_rating", "movie_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    test_rankings["rank"] = test_rankings.groupby("user_id").cumcount() + 1

    # TODO: Most popular baseline
    most_rated_ranking = p1.most_rated_baseline(ratings_train, movies)
    most_positive_ranking = p1.most_positively_rated(ratings_train, movies)

    print(
        most_rated_ranking.head(10).merge(
            movies[["movie_id", "title"]],
            on="movie_id",
            how="left",
        )
    )

    print(
        most_positive_ranking.head(10).merge(
        movies[["movie_id", "title"]],
        on="movie_id",
        how="left",
        )
    )

    validation_users = ratings_validation["user_id"].unique()
    # Output to pass to evaluate
    most_rated_recommendations = p1.recommend_popular(
        ranking=most_rated_ranking,
        ratings_history=ratings_train,
        user_ids=validation_users,
        top_n=10,
    )
    most_positive_recommendations = p1.recommend_popular(
        ranking=most_positive_ranking,
        ratings_history=ratings_train,
        user_ids=validation_users,
        top_n=10,
    )
    # TODO: Evaluation

    p1.metric_evaluation(
        predictions=test_rankings,
        ground_truth=ratings_validation,
        catalogue=movies["movie_id"],
        k=10,
        relevance_threshold=4,
    )

if __name__ == "__main__":
    main()
