import argparse
from pathlib import Path

import part_1

REQUIRED_DATA_FILES = ("ratings.dat", "movies.dat", "users.dat")

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
    part_1.check_data_dir(args.data_dir)

if __name__ == "__main__":
    main()
