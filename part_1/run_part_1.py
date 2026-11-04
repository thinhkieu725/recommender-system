from pathlib import Path
import pandas as pd

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
    raise NotImplementedError("Implement the characterize_data function.")


def run(data_dir: Path) -> None:
    """
    Entry point for running part 1: data, protocol, and baselines.
    """
    check_data_dir(data_dir)
    users, movies, ratings = load_data(data_dir)
    characterize_data(users, movies, ratings)

    # TODO: Temporal train/valid/test split.

    # TODO: Item-based kNN baseline.

    # TODO: Most popular baseline.

    raise NotImplementedError("Implement the run function of part 1.")