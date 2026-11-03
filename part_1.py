from pathlib import Path

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

