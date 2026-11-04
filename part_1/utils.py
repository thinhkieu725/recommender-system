from pathlib import Path
import pandas as pd

def read_dat_file(file_path: Path, column_types: dict[str, type]) -> pd.DataFrame:
    """
    Load a "::"-separated data file as a pandas DataFrame.
    """
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        header=None,
        names=list(column_types),
        dtype=column_types,
        encoding="latin-1",
    )
