import part_1
from part_1 import utils
from scipy.stats import ttest_rel
from part_1.part_1 import _ndcg_at_k, metric_evaluation

import numpy as np
import pandas as pd

def encoding_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding movie genres and release years as binary features"""
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

#--- Variability report: statistical paired test---
def ranked_movies_by_user(recommendations: pd.DataFrame):
    """
    Reformat the DataFrame for compatibility with ndcg computation method
    """
    return (
        recommendations.sort_values(["user_id", "rank"])
        .groupby("user_id")["movie_id"]
        .agg(list)
        .to_dict()
    )

def compare_test_ndcg(content_recommendations, baseline_recommendations, ratings_test):
    """Compare content-based minus baseline NDCG@10 on paired test users."""

    content_by_user = ranked_movies_by_user(content_recommendations)
    baseline_by_user = ranked_movies_by_user(baseline_recommendations)

    content_scores = []
    baseline_scores = []

    for user_id, ratings in ratings_test.groupby("user_id"):
        # Match the evaluator's eligibility rule.
        if not (ratings["rating"] >= 4).any():
            continue

        user_ratings = dict(zip(ratings["movie_id"], ratings["rating"]))
        content_scores.append(
            _ndcg_at_k(content_by_user.get(user_id, []), user_ratings, k=10)
        )
        baseline_scores.append(
            _ndcg_at_k(baseline_by_user.get(user_id, []), user_ratings, k=10)
        )

    if len(content_scores) < 2:
        raise ValueError("At least two eligible test users are required.")

    test = ttest_rel(content_scores, baseline_scores)
    interval = test.confidence_interval(confidence_level=0.95)

    content_mean = float(np.mean(content_scores))
    baseline_mean = float(np.mean(baseline_scores))

    return {
        "evaluated_users": len(content_scores),
        "content_ndcg": content_mean,
        "baseline_ndcg": baseline_mean,
        "mean_difference": content_mean - baseline_mean,
        "ci_95": (float(interval.low), float(interval.high)),
        "p_value": float(test.pvalue),
    }

#--- Disaggregated results
def evaluate_test_slice(label,
                        test_slice: pd.DataFrame,
                        recommendations: pd.DataFrame,
                        catalogue):
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