from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, KNNWithMeans, Prediction, Reader, accuracy
from part_1.utils import read_dat_file


REQUIRED_DATA_FILES = ("ratings.dat", "movies.dat", "users.dat")

# Tuned hyperparameters
# k tuned on validation NDCG@10 (warm users, train-only fit) over 5-200.
NEIGHBORHOOD_SIZE_K = 20
SIMILARITY_MEASURE = "pearson"  # "cosine", "msd", "pearson", "pearson_baseline"
MIN_SUPPORT = 10

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



def item_based_knn_baseline(
        ratings_train: pd.DataFrame, 
        ratings_validation: pd.DataFrame, 
        movies: pd.DataFrame
    ) -> KNNWithMeans:
    """
    Implement an item-based kNN baseline recommender system.

    Fit on train and evaluate on validation, then refit on train plus validation
    and evaluate on test, so test ratings stay unseen and later history is used.
    """
    k = NEIGHBORHOOD_SIZE_K
    similarity = SIMILARITY_MEASURE
    min_support = MIN_SUPPORT

    print("\n========== ITEM-BASED KNN ==========")
    print(f"k: {k}")
    print(f"Similarity: {similarity}")
    print(f"Min support: {min_support}")

    rating_columns = ["user_id", "movie_id", "rating"]

    reader = Reader(rating_scale=(1, 5))
    training_set = Dataset.load_from_df(ratings_train[rating_columns], reader)
    training_set = training_set.build_full_trainset()

    # KNNWithMeans predicts from mean-centred neighbour ratings.
    # user_based=False makes the similarities item-item instead of user-user,
    # so the means are item means.
    model = KNNWithMeans(
        k=k,
        sim_options={
            "name": similarity,
            "user_based": False,
            "min_support": min_support,
        },
        verbose=False,
    )
    model.fit(training_set)

    # validation for choosing k and similarity measure
    # Neighborhood size k and similarity measure are chosen using validation result, so test stays unbiased.
    predictions_valid = item_based_knn_recommend(
        model=model,
        ratings_history=ratings_train,
        user_ids=ratings_validation["user_id"].unique(),
        catalogue=movies["movie_id"],
        top_n=10,
    )
    print("\n========== Item-based kNN (validation, all users) ==========")
    metric_evaluation(
        predictions=predictions_valid,
        ground_truth=ratings_validation,
        catalogue=movies["movie_id"],
        k=10,
        relevance_threshold=4,
    )

    # Refit on train + validation
    combined_ratings = pd.concat([ratings_train, ratings_validation], ignore_index=True)
    combined_set = Dataset.load_from_df(combined_ratings[rating_columns], reader)
    combined_set = combined_set.build_full_trainset()
    model.fit(combined_set)

    return model


def item_based_knn_recommend(
        model: KNNWithMeans,
        ratings_history: pd.DataFrame,
        user_ids,
        catalogue: pd.Series,
        top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend the top-n unseen catalogue movies per user with item-based kNN.

    Every candidate movie i is scored with the item-based KNNWithMeans
    formula over the k most similar movies j the user rated, counting only
    positive similarities:

        score(u, i) = mean(i) + sum(sim(i, j) * (r(u, j) - mean(j))) / sum(sim(i, j))

    This matches model.predict, but runs on the whole catalogue at once
    instead of one pair at a time.

    Movies the model never saw, movies without a positive neighbour, and
    users the model never saw get no score, so some users receive fewer
    than top_n recommendations. model.predict would fall back to the item
    mean for a movie without a positive neighbour; that is left out here,
    because with min_support it would rank barely rated movies by their
    noisy mean. The output has the same columns as recommend_popular:
    user_id, movie_id, rank, score.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    trainset = model.trainset
    k = model.k

    # Include every historical rating, including low ratings.
    seen_by_user = (
        ratings_history.groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    # Candidates are the catalogue movies the model knows, ordered by
    # movie_id so that a stable sort breaks score ties by movie_id.
    inner_by_movie = {
        trainset.to_raw_iid(inner_id): inner_id
        for inner_id in trainset.all_items()
    }
    candidate_movies = np.array(
        sorted(set(catalogue) & inner_by_movie.keys())
    )
    candidate_inner = np.array(
        [inner_by_movie[movie_id] for movie_id in candidate_movies],
        dtype=int,
    )
    candidate_sim = model.sim[candidate_inner]
    # With user_based=False, model.means holds the item means.
    item_means = np.asarray(model.means)
    candidate_means = item_means[candidate_inner]

    inner_by_user = {
        trainset.to_raw_uid(inner_id): inner_id
        for inner_id in trainset.all_users()
    }

    recommendations = []

    for user_id in pd.Series(user_ids).unique():
        inner_uid = inner_by_user.get(user_id)
        if inner_uid is None:
            continue

        rated_inner, rated_ratings = map(np.array, zip(*trainset.ur[inner_uid]))
        rated_inner = rated_inner.astype(int)
        similarities = candidate_sim[:, rated_inner]
        # The user's rating of each neighbour, relative to that neighbour's mean.
        deviations = np.broadcast_to(
            rated_ratings - item_means[rated_inner], similarities.shape
        )

        # Keep each candidate's k most similar rated movies. The stable sort
        # breaks similarity ties in rating order, as KNNWithMeans does.
        if similarities.shape[1] > k:
            top_k = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
            similarities = np.take_along_axis(similarities, top_k, axis=1)
            deviations = np.take_along_axis(deviations, top_k, axis=1)

        # KNNWithMeans ignores neighbours with non-positive similarity.
        similarities = np.clip(similarities, 0, None)
        sim_sums = similarities.sum(axis=1)

        scores = np.full(len(candidate_movies), -np.inf)
        scorable = sim_sums > 0
        scores[scorable] = candidate_means[scorable] + (
            (similarities * deviations).sum(axis=1)[scorable] / sim_sums[scorable]
        )

        seen_movies = seen_by_user.get(user_id, set())
        scores[np.isin(candidate_movies, list(seen_movies))] = -np.inf

        best = np.argsort(-scores, kind="stable")[:top_n]
        best = best[np.isfinite(scores[best])]

        recommendations.append(
            pd.DataFrame(
                {
                    "user_id": user_id,
                    "movie_id": candidate_movies[best],
                    "rank": np.arange(1, len(best) + 1),
                    "score": scores[best],
                }
            )
        )

    columns = ["user_id", "movie_id", "rank", "score"]

    if not recommendations:
        return pd.DataFrame(columns=columns)

    return pd.concat(recommendations, ignore_index=True)[columns]


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
        user_ratings.get(movie_id, 1) - 1
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

def metric_evaluation_warm_users(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ratings_history: pd.DataFrame,
    catalogue: pd.Series,
    k: int = 10,
    relevance_threshold: int = 4,
) -> dict[str, float | int]:
    """
    Evaluate top-k recommendations on warm users only.

    Cold-start users, who have no ratings in ratings_history, are removed
    from the predictions and the ground truth before calling
    metric_evaluation. Use it with the history the model was trained on.
    """
    warm_users = set(ratings_history["user_id"])

    is_warm = ground_truth["user_id"].isin(warm_users)
    n_cold_users = ground_truth.loc[~is_warm, "user_id"].nunique()
    print(f"Cold-start test users removed: {n_cold_users}")

    return metric_evaluation(
        predictions=predictions[predictions["user_id"].isin(warm_users)],
        ground_truth=ground_truth[is_warm],
        catalogue=catalogue,
        k=k,
        relevance_threshold=relevance_threshold,
    )

def run_final_evaluation(
    kNN_result: pd.DataFrame,
    most_rated_result: pd.DataFrame,
    most_positive_result: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalogue: pd.Series,
    ratings_history: pd.DataFrame,
    k: int = 10,
    threshold: int = 4,
) -> None:
    baselines = [
        ("Item-based kNN (all users)", kNN_result),
        ("Most-rated Item", most_rated_result),
        ("Most positive rated Item", most_positive_result)
    ]

    for model_name, predictions in baselines:
        print(f"\n========== {model_name.capitalize()} ==========")
        metric_evaluation(
            predictions=predictions,
            ground_truth=ground_truth,
            catalogue=catalogue,
            k=k,
            relevance_threshold=threshold,
        )

    # kNN cannot recommend to users without training ratings, so also
    # report its metrics on warm users only.
    print("\n========== Item-based kNN (warm users only) ==========")
    metric_evaluation_warm_users(
        predictions=kNN_result,
        ground_truth=ground_truth,
        ratings_history=ratings_history,
        catalogue=catalogue,
        k=k,
        relevance_threshold=threshold,
    )

