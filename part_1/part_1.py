from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, KNNBasic, Prediction, Reader, accuracy
from part_1.utils import read_dat_file

REQUIRED_DATA_FILES = ("ratings.dat", "movies.dat", "users.dat")

# Tuned hyperparameters
NEIGHBORHOOD_SIZE_K = 40
SIMILARITY_MEASURE = "cosine"  # "cosine", "msd", "pearson", "pearson_baseline"

def check_data_dir(data_dir: Path) -> None:
    missing_files = [
        file_name
        for file_name in REQUIRED_DATA_FILES
        if not (data_dir / file_name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{data_dir} is missing required files: {', '.join(missing_files)}"
        )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the users, movies, and ratings data from the specified directory.
    """
    users = read_dat_file(
        data_dir / "users.dat",
        {
            "user_id": int,
            "gender": str,
            "age": int,
            "occupation": int,
            "zip_code": str,
        },
    )
    movies = read_dat_file(
        data_dir / "movies.dat",
        {"movie_id": int, "title": str, "genres": str},
    )
    ratings = read_dat_file(
        data_dir / "ratings.dat",
        {"user_id": int, "movie_id": int, "rating": int, "timestamp": int},
    )
    return users, movies, ratings


def characterize_data(users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """
    Print basic statistics about the users, movies, and ratings data.
    Illustrate the data statistics and characteristics with plots.
    """

    # 0. Input validation
    datasets = {
        "users": users,
        "movies": movies,
        "ratings": ratings,
    }

    required_columns = {
        "users": {"user_id"},
        "movies": {"movie_id"},
        "ratings": {"user_id", "movie_id", "rating", "timestamp"},
    }

    for name, df in datasets.items():

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

        if df.empty:
            raise ValueError(
                f"Cannot characterize data: {name} DataFrame is empty."
            )

        missing = required_columns[name] - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} DataFrame is missing columns: {missing}"
            )

    # 1. Data quality overview

    print("\n========== DATA QUALITY ==========")

    for name, df in datasets.items():
        print(f"\n{name.upper()}")

        print("Shape:", df.shape)
        print("Missing values:")
        print(df.isnull().sum())
        print("Duplicate rows:", df.duplicated().sum())

    # 2. Users, items, interactions and density

    n_users = users["user_id"].nunique()
    n_movies = movies["movie_id"].nunique()
    n_ratings = len(ratings)

    if n_users == 0 or n_movies == 0:
        raise ValueError(
            "Cannot calculate density: no valid users or movies."
        )

    density = n_ratings / (n_users * n_movies)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Number of users: {n_users}")
    print(f"Number of movies: {n_movies}")
    print(f"Number of interactions: {n_ratings}")
    print(f"Dataset density: {density:.4%}")
    print(f"Dataset sparsity: {1 - density:.4%}")

    # 3. Interactions per user and movie

    interactions_per_user = ratings.groupby("user_id").size()
    interactions_per_movie = ratings.groupby("movie_id").size()

    print("\n========== INTERACTIONS ==========")

    print("\nInteractions per user:")
    print(interactions_per_user.describe())

    print("\nInteractions per movie:")
    print(interactions_per_movie.describe())

    # Long-tail distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(interactions_per_user, bins=50)
    axes[0].set_title("Interactions per User")
    axes[0].set_xlabel("Number of ratings")
    axes[0].set_ylabel("Number of users")
    axes[0].set_yscale("log")

    axes[1].hist(interactions_per_movie, bins=50)
    axes[1].set_title("Interactions per Movie")
    axes[1].set_xlabel("Number of ratings")
    axes[1].set_ylabel("Number of movies")
    axes[1].set_yscale("log")

    fig.tight_layout()
    plt.show()

    # 4. Rating distribution

    print("\n========== RATING DISTRIBUTION ==========")

    valid_ratings = pd.to_numeric(
        ratings["rating"], errors="coerce"
    ).dropna()

    invalid_ratings = (
        len(ratings) - len(valid_ratings)
        + (~valid_ratings.between(1, 5)).sum()
    )

    print("Invalid ratings:", invalid_ratings)

    if valid_ratings.empty:
        print("No valid ratings available.")
    else:
        print(valid_ratings.describe())

        rating_counts = valid_ratings.value_counts().sort_index()
        print("\nRating counts:")
        print(rating_counts)

        plt.figure(figsize=(8, 5))
        rating_counts.plot(kind="bar")

        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Number of ratings")
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

    # 5. Temporal distribution

    print("\n========== TEMPORAL DISTRIBUTION ==========")

    numeric_timestamps = pd.to_numeric(
        ratings["timestamp"], errors="coerce"
    )

    timestamps = pd.to_datetime(
        numeric_timestamps,
        unit="s",
        errors="coerce",
        utc=True
    ).dropna()

    invalid_timestamps = len(ratings) - len(timestamps)

    print("Invalid timestamps:", invalid_timestamps)

    if timestamps.empty:
        print("No valid timestamps available.")
    else:
        print("First rating:", timestamps.min())
        print("Last rating:", timestamps.max())
        print("Time span:", timestamps.max() - timestamps.min())

        # Monthly interaction activity
        monthly_activity = (
            timestamps
            .dt.tz_localize(None)
            .dt.to_period("M")
            .value_counts()
            .sort_index()
        )

        plt.figure(figsize=(14, 5))
        monthly_activity.plot()

        plt.title("Rating Activity Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of ratings")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def split_data_temporally(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the ratings data 80/10/10 into training, validation, and test sets
    using two global timestamp cuts.
    """
    sorted_timestamps = ratings["timestamp"].sort_values().to_numpy()
    validation_start = sorted_timestamps[int(len(sorted_timestamps) * 0.8)]
    test_start = sorted_timestamps[int(len(sorted_timestamps) * 0.9)]

    train = ratings[ratings["timestamp"] < validation_start]
    validation = ratings[
        (ratings["timestamp"] >= validation_start) & (ratings["timestamp"] < test_start)
    ]
    test = ratings[ratings["timestamp"] >= test_start]
    return train, validation, test


def print_prediction_quality(split_name: str, predictions: list[Prediction]) -> None:
    rmse = accuracy.rmse(predictions, verbose=False)
    # Surprise predicts the global mean when a user or item is unknown or has no
    # neighbors. A temporal split creates many such cases, and they affect the RMSE.
    n_impossible = sum(
        prediction.details["was_impossible"] for prediction in predictions
    )

    print(f"{split_name} RMSE: {rmse:.4f}")
    print(f"{split_name} impossible predictions: {n_impossible} of {len(predictions)}")


def item_based_knn_baseline(ratings_train: pd.DataFrame, ratings_validation: pd.DataFrame, ratings_test: pd.DataFrame) -> None:
    """
    Implement an item-based kNN baseline recommender system.

    Fit on train and evaluate on validation, then refit on train plus validation
    and evaluate on test, so test ratings stay unseen and later history is used.
    """
    k = NEIGHBORHOOD_SIZE_K
    similarity = SIMILARITY_MEASURE

    print("\n========== ITEM-BASED KNN ==========")
    print(f"k: {k}")
    print(f"Similarity: {similarity}")

    rating_columns = ["user_id", "movie_id", "rating"]

    reader = Reader(rating_scale=(1, 5))
    training_set = Dataset.load_from_df(ratings_train[rating_columns], reader)
    training_set = training_set.build_full_trainset()

    # user_based=False makes the similarities item-item instead of user-user.
    model = KNNBasic(
        k=k,
        sim_options={"name": similarity, "user_based": False},
        verbose=False,
    )
    model.fit(training_set)

    validation_set = list(ratings_validation[rating_columns].itertuples(index=False, name=None))
    validation_predictions = model.test(validation_set)
    print_prediction_quality("Validation", validation_predictions)

    # Neighborhood size k and similarity measure are chosen before reading this result, so test stays unbiased.
    # TODO: to be adapted with the evaluation protocol to be implemented.
    test_set = list(ratings_test[rating_columns].itertuples(index=False, name=None))
    test_predictions = model.test(test_set)
    print_prediction_quality("Test", test_predictions)

    return model

def most_rated_baseline(
        ratings_train: pd.DataFrame,
        movies: pd.DataFrame,
) -> pd.DataFrame:
    """Rank catalogue movies by their number of training ratings."""
    counts = ratings_train.groupby("movie_id").size()

    ranking = movies[["movie_id"]].drop_duplicates().copy()
    ranking["score"] = (
        ranking["movie_id"].map(counts).fillna(0).astype(int)
    )

    return ranking.sort_values(
        ["score", "movie_id"],
        ascending=[False, True],
        ignore_index=True,
    )


def most_positively_rated(
        ratings_train: pd.DataFrame,
        movies: pd.DataFrame,
        positive_threshold: int = 4,
) -> pd.DataFrame:
    """Rank catalogue movies by their number of positive training ratings."""
    positive_ratings = ratings_train[
        ratings_train["rating"] >= positive_threshold
        ]
    counts = positive_ratings.groupby("movie_id").size()

    ranking = movies[["movie_id"]].drop_duplicates().copy()
    ranking["score"] = (
        ranking["movie_id"].map(counts).fillna(0).astype(int)
    )

    return ranking.sort_values(
        ["score", "movie_id"],
        ascending=[False, True],
        ignore_index=True,
    )

def recommend_popular(
        ranking: pd.DataFrame,
        ratings_history: pd.DataFrame,
        user_ids,
        top_n: int = 10,
) -> pd.DataFrame:
    """Recommend the highest-ranked movies each user has not rated."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Include every historical rating, including low ratings.
    seen_by_user = (
        ratings_history.groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    recommendations = []

    for user_id in pd.Series(user_ids).unique():
        seen_movies = seen_by_user.get(user_id, set())

        user_recommendations = (
            ranking.loc[
                ~ranking["movie_id"].isin(seen_movies),
                ["movie_id", "score"],
            ]
            .head(top_n)
            .copy()
        )

        user_recommendations.insert(0, "user_id", user_id)
        user_recommendations["rank"] = range(
            1, len(user_recommendations) + 1
        )
        recommendations.append(user_recommendations)

    columns = ["user_id", "movie_id", "rank", "score"]

    if not recommendations:
        return pd.DataFrame(columns=columns)

    return pd.concat(recommendations, ignore_index=True)[columns]

import math

def _precision_at_k(
        recommended: list[int], 
        relevant: set[int], 
        k: int = 10
    )-> float:
    """Fraction of the top-k recommendations that are relevant."""
    hits = len(set(recommended[:k]) & relevant)
    return hits / k

def _recall_at_k(
        recommended: list[int], 
        relevant: set[int], 
        k: int = 10
    )-> float:
    """Fraction of all relevant movies found in the top-k."""
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)

def _dcg(
        relevances: list[int], 
        k: int = 10
    )-> float:
    """Calculate Discounted Cumulative Gain."""
    return sum(
        (2 ** relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances[:k])
    )

def _ndcg_at_k(
        recommended: list[int], 
        user_ratings: dict[int, int], 
        k: int = 10
    )-> float:
    """
    Calculate NDCG using graded relevance.
    Rating 1 -> relevance 0, ..., rating 5 -> relevance 4.
    """
    # Relevance of recommended movies, in recommendation order.
    predicted_relevances = [
        user_ratings.get(movie_id, 0) - 1
        for movie_id in recommended[:k]
    ]

    dcg = _dcg(predicted_relevances, k)

    # Ideal ranking: sort all future ratings from highest to lowest.
    ideal_relevances = sorted(
        [rating - 1 for rating in user_ratings.values()],
        reverse=True
    )

    idcg = _dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def metric_evaluation(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalogue: pd.Series,
    k: int = 10,
    relevance_threshold: int = 4,
) -> dict[str, float | int]:
    
    """
    Evaluate top-k recommendations.

    predictions:
        DataFrame with user_id, movie_id, rank.

    ground_truth:
        DataFrame with user_id, movie_id, rating.

    catalogue:
        Series containing all movie IDs.
    """

    # 1. Validate required columns.
    prediction_columns = {"user_id", "movie_id", "rank"}
    ground_truth_columns = {"user_id", "movie_id", "rating"}

    if not prediction_columns.issubset(predictions.columns):
        raise ValueError("Predictions must contain user_id, movie_id, rank.")

    if not ground_truth_columns.issubset(ground_truth.columns):
        raise ValueError("Ground truth must contain user_id, movie_id, rating.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if predictions.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user/movie recommendations found.")

    if ground_truth.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user/movie ground-truth ratings found.")

    # 2. Prepare catalogue and recommendations.
    catalogue = set(catalogue)

    predictions = predictions[
        predictions["movie_id"].isin(catalogue)
    ].copy()

    predictions = predictions.sort_values(["user_id", "rank"])

    top_k = predictions.groupby("user_id", sort=False).head(k)

    # 3. Prepare ground truth.
    # Map each user to their future movie ratings.
    user_ratings = {
        user_id: dict(zip(group["movie_id"], group["rating"]))
        for user_id, group in ground_truth.groupby("user_id")
    }

    # Only ratings >= threshold count as relevant for Precision and Recall.
    relevant = {
        user_id: {
            movie_id
            for movie_id, rating in ratings.items()
            if rating >= relevance_threshold
        }
        for user_id, ratings in user_ratings.items()
    }

    # Exclude users without relevant future ratings from accuracy averages.
    evaluation_users = {
        user_id
        for user_id, movies in relevant.items()
        if movies
    }

    # 4. Calculate metrics for each evaluation user.
    recommended_by_user = {
        user_id: group["movie_id"].tolist()
        for user_id, group in top_k.groupby("user_id")
    }

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id in evaluation_users:
        recommended = recommended_by_user.get(user_id, [])

        precision_scores.append(
            _precision_at_k(recommended, relevant[user_id], k)
        )

        recall_scores.append(
            _recall_at_k(recommended, relevant[user_id], k)
        )

        ndcg_scores.append(
            _ndcg_at_k(recommended, user_ratings[user_id], k)
        )

    # 5. Calculate catalogue coverage.
    recommended_movies = set(top_k["movie_id"])
    coverage = (
        len(recommended_movies) / len(catalogue)
        if catalogue else 0.0
    )

    # 6. Return results.
    results = {
        f"Precision@{k}": (
            sum(precision_scores) / len(precision_scores)
            if precision_scores else 0.0
        ),
        f"Recall@{k}": (
            sum(recall_scores) / len(recall_scores)
            if recall_scores else 0.0
        ),
        f"NDCG@{k}": (
            sum(ndcg_scores) / len(ndcg_scores)
            if ndcg_scores else 0.0
        ),
        "Catalogue Coverage": coverage,
        "Evaluated Users": len(evaluation_users),
        "Excluded Users": len(user_ratings) - len(evaluation_users),
    }

    print("\n========== EVALUATION RESULTS ==========")

    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return results




