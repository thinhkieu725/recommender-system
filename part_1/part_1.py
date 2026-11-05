from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from part_1.utils import read_dat_file

REQUIRED_DATA_FILES = ("ratings.dat", "movies.dat", "users.dat")


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