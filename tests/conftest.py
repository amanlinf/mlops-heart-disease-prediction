from __future__ import annotations

import pandas as pd
import pytest

from src.data import ALL_COLUMNS, FEATURES

# sample patient record
# @pytest.fixture - used to provide reusable objects for testing
@pytest.fixture
def patient_dict() -> dict:
    return {
        "age": 60, "sex": 1, "cp": 1, "trestbps": 145, "chol": 133,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.4, "slope": 3, "ca": 0, "thal": 6,
    }

# sample artificial dataset creation for running automated tests
@pytest.fixture
def sample_dataset(patient_dict) -> pd.DataFrame:
    rows = []
    for idx in range(40):
        record = patient_dict.copy()
        record["age"] += (idx % 17) - 7
        record["trestbps"] += (idx % 6) * 3
        record["chol"] += idx * 2
        record["thalach"] -= idx % 14
        record["sex"] = idx % 2
        record["cp"] = (idx % 4) + 1
        record["fbs"] = idx % 2
        record["exang"] = (idx // 2) % 2
        record["ca"] = idx % 4
        record["target"] = (idx // 3) % 2
        rows.append(record)
    return pd.DataFrame(rows, columns=FEATURES + ["target"])

# dataframe creation
@pytest.fixture
def raw_frame(patient_dict):
    record = {name: patient_dict[name] for name in FEATURES}
    record["num"] = 3
    return pd.DataFrame([record], columns=ALL_COLUMNS)
