"""
data_loader.py
==============
Download and load the class Netflix-competition data.

Data source: https://www.tau.ac.il/~saharon/StatsLearn2026-7/
(Note: the plain http:// links on the course page return 404 -- https works.)

Conventions in the raw files
----------------------------
* Ratings are integers 1..5, and 0 means "missing".
* Dates are the number of days since 1997-01-01, and 0 means "missing".
* Column j of the ratings/dates matrices corresponds to row j of movie_titles.txt.
* The first 14 movies were rated by *every* user (train and test).

In the loaded DataFrames we replace the 0 placeholders with NaN, so pandas/numpy
missing-value handling works naturally.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.tau.ac.il/~saharon/StatsLearn2026-7/"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILES = [
    "train_ratings_all.txt",
    "test_ratings_all.txt",
    "train_dates_all.txt",
    "test_dates_all.txt",
    "train_y_rating.txt",
    "train_y_date.txt",
    "test_y_date.txt",
    "movie_titles.txt",
]
N_FULL = 14                     # number of movies rated by all users
DATE_ORIGIN = pd.Timestamp("1997-01-01")


def download_data(data_dir: Path = DATA_DIR, force: bool = False) -> None:
    """Download all competition files into `data_dir` (skips existing files)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = data_dir / name
        if target.exists() and not force:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, target)


@dataclass
class NetflixData:
    """Container for the full competition data set."""
    movies: pd.DataFrame          # columns: year, title (one row per movie column)
    X_train: pd.DataFrame         # ratings, 10000 x 99, NaN = missing
    X_test: pd.DataFrame          # ratings, 2931 x 99, NaN = missing
    D_train: pd.DataFrame         # rating dates (days since 1997-01-01), NaN = missing
    D_test: pd.DataFrame
    y_train: pd.Series            # "Miss Congeniality" rating (train only)
    y_date_train: pd.Series       # date of the "Miss Congeniality" rating
    y_date_test: pd.Series


def _read_matrix(path: Path, columns) -> pd.DataFrame:
    m = np.loadtxt(path, dtype=float)
    m = np.atleast_2d(m)
    m[m == 0] = np.nan            # 0 => missing
    return pd.DataFrame(m, columns=columns)


def _read_vector(path: Path, name: str) -> pd.Series:
    v = np.loadtxt(path, dtype=float).ravel()
    v[v == 0] = np.nan
    return pd.Series(v, name=name)


def load_data(data_dir: Path = DATA_DIR, download: bool = True) -> NetflixData:
    """Load all files into a `NetflixData` object (downloading them if needed)."""
    if download:
        download_data(data_dir)

    # Split on the FIRST comma only -- some titles contain commas themselves.
    rows = []
    with open(data_dir / "movie_titles.txt", encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if line:
                year, title = line.split(",", 1)
                rows.append((int(year), title.strip()))
    movies = pd.DataFrame(rows, columns=["year", "title"])
    cols = movies["title"].tolist()

    return NetflixData(
        movies=movies,
        X_train=_read_matrix(data_dir / "train_ratings_all.txt", cols),
        X_test=_read_matrix(data_dir / "test_ratings_all.txt", cols),
        D_train=_read_matrix(data_dir / "train_dates_all.txt", cols),
        D_test=_read_matrix(data_dir / "test_dates_all.txt", cols),
        y_train=_read_vector(data_dir / "train_y_rating.txt", "y"),
        y_date_train=_read_vector(data_dir / "train_y_date.txt", "y_date"),
        y_date_test=_read_vector(data_dir / "test_y_date.txt", "y_date"),
    )


if __name__ == "__main__":
    d = load_data()
    print("X_train", d.X_train.shape, "| X_test", d.X_test.shape,
          "| y_train", d.y_train.shape)
