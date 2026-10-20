from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# source for downloading the data
# uci contains multiple subsets of data, have used the cleveland data
SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

ALL_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FEATURES = [name for name in ALL_COLUMNS if name != "num"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
TARGET = "target"

DEFAULT_RAW = Path("data/raw/processed.cleveland.data")
DEFAULT_CLEAN = Path("data/processed/heart_clean.csv")

# downloads the data from the source and put the raw data at mentioned filepath
def download_dataset(destination: Path = DEFAULT_RAW, force: bool = False) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return destination
    with urlopen(SOURCE_URL, timeout=60) as response:
        content = response.read()
    if not content or b"," not in content:
        raise RuntimeError("Dataset download failed or the data is corrupt")
    destination.write_bytes(content)
    return destination

# clean raw record values, nan values are left for train fold imputation, after cross validation split
def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw records; leave missing feature values for train-fold imputation."""
    if list(frame.columns) != ALL_COLUMNS:
        raise ValueError(f"Expected columns {ALL_COLUMNS}, got {list(frame.columns)}")
    result = frame.copy()
    for name in ALL_COLUMNS:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if result["num"].isna().any():
        raise ValueError("Target has missing or invalid values")
    if not result["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Target values must be in 0..4")
    if result[FEATURES].isna().all(axis=1).any():
        raise ValueError("Found a record with no usable predictors")
    result[TARGET] = (result["num"] > 0).astype("int64")
    result = result.drop(columns="num").drop_duplicates().reset_index(drop=True)
    return result[FEATURES + [TARGET]]

# loading the data in pandas dataframe, note-> '?' depicts the null values, and are further marked as Nan
def load_raw_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=ALL_COLUMNS, na_values=["?"])

# prepare the dataset- involves downloading and storing the raw dataset, cleaning and storing the clean dataset
def prepare_dataset(raw: Path = DEFAULT_RAW, output: Path = DEFAULT_CLEAN) -> pd.DataFrame:
    raw = download_dataset(Path(raw))
    df = load_raw_file(raw)
    cleaned = clean_dataframe(df)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    print(f"Clean dataset saved: {output} | rows={len(cleaned)}")
    print(f"Target distribution: {cleaned[TARGET].value_counts().to_dict()}")
    print(f"Missing counts: {cleaned.isna().sum().to_dict()}")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLEAN)
    args = parser.parse_args()
    prepare_dataset(args.raw, args.output)


if __name__ == "__main__":
    main()
