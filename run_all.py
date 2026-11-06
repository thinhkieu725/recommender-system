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
    ib_knn_test_rankings = p1.item_based_knn_predict(item_based_knn_model, ratings_test)

    # Most-popular baseline
    most_rated_ranking = p1.most_rated_baseline(ratings_train, movies)
    most_positive_ranking = p1.most_positively_rated(ratings_train, movies)

    test_users = ratings_test["user_id"].unique()
    # Output to pass to evaluate
    most_rated_recommendations = p1.recommend_popular(
        ranking=most_rated_ranking,
        ratings_history=ratings_train,
        user_ids=test_users,
        top_n=10,
    )
    most_positive_recommendations = p1.recommend_popular(
        ranking=most_positive_ranking,
        ratings_history=ratings_train,
        user_ids=test_users,
        top_n=10,
    )

    # Evaluation
    p1.run_final_evaluation(
        ib_knn_test_rankings,
        most_rated_recommendations,
        most_positive_recommendations,
        ratings_test,
        movies["movie_id"],
        k=10,
        threshold=4
    )

if __name__ == "__main__":
    main()
