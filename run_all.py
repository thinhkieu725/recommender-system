import argparse
from pathlib import Path

from part_1 import run_part_1

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
    run_part_1.run(args.data_dir)

if __name__ == "__main__":
    main()
