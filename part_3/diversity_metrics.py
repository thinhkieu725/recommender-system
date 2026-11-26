
import numpy as np
import pandas as pd

from part_3.mmr import (
    GENRES,
    parse_genres,
    genres_to_vector,
    cosine_distance,
)

def _build_genre_lookup(movies: pd.DataFrame) -> dict:
    """Build a movie_id -> binary genre vector mapping."""
    required_columns = {"movie_id", "genres"}

    if not required_columns.issubset(movies.columns):
        raise ValueError(f"movies must contain columns: {required_columns}")

    genre_map = {}
    for row in movies.itertuples(index=False):
        genre_map[row.movie_id] = genres_to_vector(parse_genres(row.genres))

    return genre_map

def _genre_distribution(movie_ids: list, genre_lookup: dict) -> np.ndarray:
    """
    Calculate a normalized genre distribution.

    Each movie contributes equally. For movies with multiple genres,
    its contribution is divided equally among those genres.
    """
    distribution = np.zeros(len(GENRES), dtype=float)

    for movie_id in movie_ids:
        vector = genre_lookup.get(movie_id)

        if vector is None:
            continue

        number_of_genres = vector.sum()

        if number_of_genres > 0:
            distribution += vector / number_of_genres

    total = distribution.sum()

    if total > 0:
        distribution /= total

    return distribution

# 1. Intra-List Diversity (ILD)
def intra_list_diversity(recommended_movie_ids: list, genre_lookup: dict) -> float:
    """
    Calculate average pairwise cosine distance between
    recommended movies.

    Returns a value between 0 and 1.
    Higher means more diverse.
    """

    vectors = []
    for movie_id in recommended_movie_ids:
        vectors.append(genre_lookup[movie_id])

    if len(vectors) < 2:
        return 0.0

    distances = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distance = cosine_distance(vectors[i], vectors[j])
            distances.append(distance)

    return float(np.mean(distances))

# 2. Gini Coefficient
def gini_coefficient(recommendation_counts: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of recommendation exposure.

    Includes catalogue items with zero recommendations.
    0 = equal exposure.
    Values closer to 1 = more unequal exposure.
    """
    counts = np.asarray(recommendation_counts, dtype=float)

    if np.any(counts < 0):
        raise ValueError("Recommendation counts cannot be negative.")

    if len(counts) == 0 or counts.sum() == 0:
        return 0.0

    sorted_counts = np.sort(counts)
    n = len(sorted_counts)

    indices = np.arange(1, n + 1)

    gini = (2 * np.sum(indices * sorted_counts) / (n * np.sum(sorted_counts)) - (n + 1) / n)

    return float(gini)

# 3. Calibration Error
def calibration_error(
    recommended_movie_ids: list,
    historical_movie_ids: list,
    genre_lookup: dict,
) -> float:
    """
    Compare the user's historical genre distribution
    with the genre distribution of their recommendations.

    Uses normalized L1 distance (total variation distance).
    Lower means better calibration.
    """
    user_profile = _genre_distribution(
        historical_movie_ids,
        genre_lookup,
    )

    recommendation_profile = _genre_distribution(
        recommended_movie_ids,
        genre_lookup,
    )

    # Cannot calculate calibration without valid genre information.
    if user_profile.sum() == 0 or recommendation_profile.sum() == 0:
        return np.nan

    return float(0.5 * np.sum(np.abs(user_profile - recommendation_profile)))

# 4. Evaluate all diversity metrics
def evaluate_diversity(
    predictions: pd.DataFrame,
    ratings_history: pd.DataFrame,
    movies: pd.DataFrame,
    catalogue: pd.Series,
    k: int = 10,
) -> dict:
    """
    Calculate ILD@k, Gini and calibration error.

    Inputs:
    - predictions:      DataFrame containing user_id, movie_id and rank.

    - ratings_history:  Training history containing user_id and movie_id.

    - movies:           Movie metadata containing movie_id and genres.

    - catalogue:        Complete catalogue of movie IDs.

    - k:                Number of recommended movies to evaluate.
    """
    required_prediction_columns = {"user_id", "movie_id", "rank"}

    required_history_columns = {"user_id", "movie_id"}

    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(f"predictions must contain {required_prediction_columns}")

    if not required_history_columns.issubset(ratings_history.columns):
        raise ValueError(f"ratings_history must contain {required_history_columns}")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if predictions.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user/movie predictions found.")

    if movies["movie_id"].duplicated().any():
        raise ValueError("Duplicate movie IDs found in movies.")

    genre_lookup = _build_genre_lookup(movies)

    # Keep the top-k recommendations for each user.
    top_recommendations = (
        predictions
        .sort_values(["user_id", "rank"], kind="stable")
        .groupby("user_id", sort=False)
        .head(k)
    )

    if top_recommendations.empty:
        raise ValueError("No recommendations to evaluate.")

    # Every recommended movie must have genre metadata.
    missing_metadata = (set(top_recommendations["movie_id"]) - set(genre_lookup))

    if missing_metadata:
        raise ValueError(f"{len(missing_metadata)} recommended movies are missing genre metadata.")

    # Build user -> historical movie IDs.
    history_by_user = (
        ratings_history
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    # Build user -> recommended movie IDs.
    recommendations_by_user = (
        top_recommendations
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    # ILD and calibration per user
    ild_values = []
    calibration_values = []

    for user_id, recommended_ids in recommendations_by_user.items():
        # Intra-list diversity.
        ild = intra_list_diversity(recommended_ids, genre_lookup)
        ild_values.append(ild)

        # Calibration against the user's training history.
        historical_ids = history_by_user.get(user_id, [])

        calibration = calibration_error(
            recommended_movie_ids=recommended_ids,
            historical_movie_ids=historical_ids,
            genre_lookup=genre_lookup,
        )

        if not np.isnan(calibration):
            calibration_values.append(calibration)

    if ild_values:
        mean_ild = float(np.mean(ild_values))
    else:
        mean_ild = np.nan

    if calibration_values:
        mean_calibration = float(np.mean(calibration_values))
    else:
        mean_calibration = np.nan

    # Gini across the complete catalogue
    catalogue_ids = pd.Series(catalogue).drop_duplicates().tolist()

    recommendation_counts = top_recommendations["movie_id"].value_counts()

    counts = np.zeros(len(catalogue_ids), dtype=float)
    for i, movie_id in enumerate(catalogue_ids):
        counts[i] = recommendation_counts.get(movie_id, 0)
    
    gini = gini_coefficient(counts)

    return {
        f"ILD@{k}": mean_ild,
        "Gini": gini,
        "Calibration Error": mean_calibration,
        "ILD Evaluated Users": len(ild_values),
        "Calibration Evaluated Users": len(calibration_values),
    }
