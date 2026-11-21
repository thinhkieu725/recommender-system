from part_1.part_1 import metric_evaluation
import numpy as np
import pandas as pd

def encoding_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    """Encoding movie genres and release years as binary features"""
    movies = movies.set_index("movie_id")

    genre_features = (
        movies["genres"].str.get_dummies(sep="|")

    )

    # Extract the final parenthesized year, e.g. "Toy Story (1995)".
    years = (
        movies["title"]
        .str.extract(r"\((\d{4})\)\s*$", expand=False)
        .astype("Int64")
    )

    # 1995 -> 1990; 1987 -> 1980.
    decades = (years // 10) * 10

    # Each movie belongs to one release decade.
    decade_features = pd.get_dummies(
        decades,
        prefix="decade",
        dummy_na=True,
        dtype=int,
    ).rename(columns={"decade_<NA>": "decade_unknown"})

    features = pd.concat(
        [genre_features, decade_features],
        axis=1,
    )

    return features.astype(float).sort_index()

def build_user_profile(
    user_id: int,
    ratings_history: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.Series:
    """Build one user's profile from mean-centered historical ratings."""
    user_ratings = ratings_history[
        ratings_history["user_id"] == user_id
        ]
    # A user without history has no content-based profile.
    if user_ratings.empty:
        return pd.Series(0.0, index=features.columns)

    mean_rating = user_ratings["rating"].mean() # avg(u)
    centered_ratings = user_ratings["rating"] - mean_rating # r(u,i) - avg(u)

    # Select the feature vector for each rated movie, in rating order.
    rated_features = features.loc[
        user_ratings["movie_id"]
    ]

    # Match the centered ratings to those rows by position.
    weighted_features = rated_features.mul(
        centered_ratings.to_numpy(),
        axis=0,
    )

    return weighted_features.sum(axis=0)

def cosine_similarity(
    user_profile: pd.Series,
    movie_vector: pd.Series,
) -> float:
    """Compute cosine similarity between user profile and the rated movie's features."""
    dot_product = np.dot(user_profile, movie_vector)

    profile_length = np.linalg.norm(user_profile)
    movie_length = np.linalg.norm(movie_vector)

    if profile_length == 0 or movie_length == 0:
        return 0.0

    return dot_product / (profile_length * movie_length)

def recommend_for_user(
        user_id: int,
        ratings_history: pd.DataFrame,
        movie_features: pd.DataFrame,
        top_n: int = 10,
) -> pd.DataFrame:
    """Recommend the highest-scoring unseen movies to one user."""
    columns = ["user_id", "movie_id", "rank", "score"]

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Build profile
    profile = build_user_profile(
        user_id,
        ratings_history,
        movie_features,
    )

    # Cosine is undefined for a zero profile.
    profile_length = np.linalg.norm(profile)
    if profile_length == 0:
        return pd.DataFrame(columns=columns)

    # Include every historical rating, even ratings below 4.
    seen_movies = ratings_history.loc[
        ratings_history["user_id"] == user_id,
        "movie_id",
    ]

    candidates = movie_features.loc[
        ~movie_features.index.isin(seen_movies)
    ]

    # Calculate the cosine similarity between user's profile and unrated movies' features
    scores = []
    for movie_id, movie_vector in candidates.iterrows():
        score = cosine_similarity(profile, movie_vector)
        scores.append(score)

    recommendations = pd.DataFrame({
        "movie_id": candidates.index,
        "score": scores,
    })

    # Match the baseline's tie-breaking rule.
    recommendations = recommendations.sort_values(
        ["score", "movie_id"],
        ascending=[False, True],
    ).head(top_n).reset_index(drop=True)

    recommendations.insert(0, "user_id", user_id)
    recommendations["rank"] = range(1, len(recommendations) + 1)

    return recommendations[columns]

def recommend_for_users(
    user_ids,
    ratings_history: pd.DataFrame,
    movie_features: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Combine the top unseen movies for each requested user."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    recommendations = []
    for user_id in pd.unique(user_ids):
        user_recommendations = recommend_for_user(
            user_id, ratings_history, movie_features, top_n
        )
        if not user_recommendations.empty:
            recommendations.append(user_recommendations)

    if not recommendations:
        return pd.DataFrame(columns=["user_id", "movie_id", "rank", "score"])

    return pd.concat(recommendations, ignore_index=True)

#--- Disaggregated results ---
def evaluate_test_slice(
    label,
    test_slice: pd.DataFrame,
    recommendations: pd.DataFrame,
    catalogue
):
    slice_users = test_slice["user_id"].unique()
    slice_recommendations = recommendations[
        recommendations["user_id"].isin(slice_users)
    ]

    print(f"\n========== {label} ==========")
    return metric_evaluation(
        predictions=slice_recommendations,
        ground_truth=test_slice,
        catalogue=catalogue,
        k=10,
        relevance_threshold=4,
    )

def evaluate_by_user_activity(
    ratings_history,
    ratings_test,
    model_recommendations,
    catalogue
):
    """Evaluate test users grouped by their historical rating count."""
    test_users = ratings_test["user_id"].unique()
    history_counts = (
        ratings_history.groupby("user_id")
        .size()
        .reindex(test_users, fill_value=0)
    )
    median_count = history_counts[history_counts > 0].median()
    print(f"User activity cutoff: {median_count:g} historical ratings")

    groups = {
        "No history": history_counts == 0,
        "Lower activity": (history_counts > 0) & (history_counts <= median_count),
        "Higher activity": history_counts > median_count,
    }

    for label, mask in groups.items():
        group_users = history_counts.index[mask]
        if len(group_users) == 0:
            continue

        test_slice = ratings_test[ratings_test["user_id"].isin(group_users)]
        evaluate_test_slice(label = label,
                            test_slice= test_slice,
                            recommendations = model_recommendations,
                            catalogue = catalogue
        )


def evaluate_by_test_period(
    ratings_test,
    model_recommendations,
    catalogue
):
    """Evaluate test ratings before and from the middle timestamp."""
    timestamps = ratings_test["timestamp"].sort_values()
    cutoff = timestamps.iloc[len(timestamps) // 2]
    is_early = ratings_test["timestamp"] < cutoff

    periods = {
        "Early test": ratings_test[is_early],
        "Late test": ratings_test[ratings_test["timestamp"] >= cutoff],
    }

    for label, test_slice in periods.items():
        if not test_slice.empty:
            evaluate_test_slice(label = label,
                                test_slice= test_slice,
                                recommendations = model_recommendations,
                                catalogue = catalogue
                                )


#--- Sparsity Ablation ---
def sparsity_test(
    test_users,
    shuffled_history,
    ground_truth,
    catalogue,
    features
):
    """ Subsample the training interaction to 50% and 20%"""
    for fraction in (0.50, 0.20):
        sample_size = int(len(shuffled_history) * fraction)
        sampled_history = shuffled_history.iloc[:sample_size]

        sparse_recommendations = recommend_for_users(
            user_ids=test_users,
            ratings_history=sampled_history,
            movie_features=features,
            top_n=10,
        )

        print(f"\n========== Content-based test, {fraction:.0%} history ==========")
        metric_evaluation(
            predictions=sparse_recommendations,
            ground_truth=ground_truth,
            catalogue=catalogue,
            k=10,
            relevance_threshold=4,
        )
