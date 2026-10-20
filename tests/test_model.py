import numpy as np

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.model import candidate_models, make_preprocessor

# test to check if all the dataset features are processed 
def test_all_features_are_processed():
    transformer = make_preprocessor()
    columns = [name for _, _, names in transformer.transformers for name in names]
    assert set(columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert len(columns) == 13

# test to check if both the model config could handle missing values, fit successfully and generate a probabilty target value between 0 and 1
def test_model_fit_and_missing_values(sample_dataset):
    x = sample_dataset.drop(columns="target")
    y = sample_dataset["target"]
    x.loc[0, "ca"] = np.nan
    for name, (model, _grid) in candidate_models().items():
        model.fit(x, y)
        probability = model.predict_proba(x.iloc[:1])[0, 1]
        assert 0 <= probability <= 1, name
