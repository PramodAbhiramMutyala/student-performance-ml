"""
Preprocessing module.
Creates scikit-learn ColumnTransformers, Pipelines, and handles feature/target separation.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES_EARLY_WARNING,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    NUMERICAL_FEATURES,
    RISK_LABELS,
    TARGET_COLUMN,
)


def categorize_academic_risk(score: float) -> str:
    """
    Categorizes student grade (0-20 scale) into academic risk tiers:
    - score < 10.0: High Academic Risk (failing in Portuguese secondary system)
    - 10.0 <= score < 14.0: Moderate Academic Risk (satisfactory pass)
    - score >= 14.0: Low Academic Risk (good to excellent performance)
    """
    if pd.isna(score):
        return "Unknown"
    score_val = float(score)
    if score_val < HIGH_RISK_THRESHOLD:
        return RISK_LABELS["high"]
    elif score_val < MEDIUM_RISK_THRESHOLD:
        return RISK_LABELS["medium"]
    else:
        return RISK_LABELS["low"]


def get_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that imputes and standardizes numerical
    features, and imputes and one-hot encodes categorical features.
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts early-warning feature matrix X (excluding G1, G2, G3),
    regression target y_reg (G3), and classification target y_clf (risk tier).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from dataframe.")

    # Exclude G1, G2, and G3 to guarantee no target or future-grade leakage
    cols_to_drop = [c for c in EXCLUDED_FEATURES_EARLY_WARNING if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Ensure all expected features are present in X
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = X[[c for c in expected_cols if c in X.columns]]

    y_reg = df[TARGET_COLUMN].astype(float)
    y_clf = y_reg.apply(categorize_academic_risk)

    return X, y_reg, y_clf


def split_data(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits features and both targets into reproducible train and test sets,
    stratifying on the risk tier to ensure balanced representation.
    """
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
