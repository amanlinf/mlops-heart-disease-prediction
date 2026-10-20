import pandas as pd
import pytest

from src.data import FEATURES, clean_dataframe


def test_binary_target_and_columns(raw_frame):
    cleaned = clean_dataframe(raw_frame)
    assert len(cleaned) == 1
    assert cleaned["target"].iloc[0] == 1
    assert set(cleaned.columns) == set(FEATURES + ["target"])
    assert "num" not in cleaned.columns


def test_question_mark_is_missing(raw_frame):
    raw_frame["ca"] = raw_frame["ca"].astype(object)
    raw_frame.loc[0, "ca"] = "?"
    cleaned = clean_dataframe(raw_frame)
    assert pd.isna(cleaned.loc[0, "ca"])


def test_invalid_target_rejected(raw_frame):
    raw_frame.loc[0, "num"] = 8
    with pytest.raises(ValueError, match="Target values"):
        clean_dataframe(raw_frame)


def test_no_disease_target(raw_frame):
    raw_frame.loc[0, "num"] = 0
    assert clean_dataframe(raw_frame)["target"].iloc[0] == 0
