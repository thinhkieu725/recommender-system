# Recommender System on MovieLens 1M

This project builds and evaluates movie recommenders on the MovieLens 1M dataset.
It splits the ratings temporally into train, validation and test sets, then compares an item-based KNN model (`KNNWithMeans`, tuned on validation) against two popularity baselines: most-rated and most-positively-rated.
All models produce top-10 recommendations per test user and are evaluated with ranking metrics at k=10 (relevant = rating >= 4).

## Data preparation

1. Download the MovieLens 1M dataset (`ml-1m.zip`) from https://grouplens.org/datasets/movielens/1m/.
2. Unzip it and copy these three files into a `data/` folder at the project root:

```
recommender-system/
├── data/
│   ├── movies.dat
│   ├── ratings.dat
│   └── users.dat
├── part_1/
├── requirements.txt
└── run_all.py
```

The `data/` folder is in `.gitignore`, so the dataset is not committed. The script stops with an error if any of the three files is missing.

## Running the code

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline, passing the folder that holds the `.dat` files. For example, with the files located in the `data` folder under the root folder:

```bash
python run_all.py --data-dir data
```

## Report and deliverables

The online version of the report (updated by progress) is available here: https://tuni-my.sharepoint.com/:w:/g/personal/thinh_kieu_tuni_fi/IQC_ADOi9-qQQaiSvtbzi-R2AUEQSeUp_zPaiOACLS1eeN0?e=cveboe.

The online version of the result spreadsheet (updated by progress) is available here: https://tuni-my.sharepoint.com/:x:/g/personal/thinh_kieu_tuni_fi/IQCezm202h_nS4SJrDQcPeYwAUQ3ZINhiafv4jfPM7H8tM4?e=hQtCSv