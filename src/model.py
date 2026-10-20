from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# For numerical features, median is taken for imputing values, not sensitive to outliers, further scaled, which helps in regularized logistic regression
# For Categorical features, most-frequent class is used for imputation, one hot encoding is used for creating separate binary indicators per observed catgeory
# handle_unknown="ignore" makes all-zero entries for the feature group that were not observed during training phase
# pre processor config specifications and feature preparation
def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

# model defintion and hyper-parameter specification
# pipeline contains both preprocessing and classifier details, this ensures that pipeline uses same preprocessing transformations
# this prevents data leakage during cross-validation phase, as the preprocessing refitted separately for each fold
def candidate_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": (
            Pipeline([
                ("preprocess", make_preprocessor()),
                ("classifier", LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=random_state
                )),
            ]),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline([
                ("preprocess", make_preprocessor()),
                ("classifier", RandomForestClassifier(
                    random_state=random_state, class_weight="balanced", n_jobs=1
                )),
            ]),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, None],
                "classifier__min_samples_leaf": [1, 3],
            },
        ),
    }
