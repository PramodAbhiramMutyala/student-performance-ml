"""
Unit tests for Machine Learning components:
Data loader, validation, preprocessing, evaluation metrics, and inference.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURE_COLUMNS, RISK_LABELS, TARGET_COLUMN
from src.ml.evaluate import evaluate_classification_model, evaluate_regression_model
from src.ml.predict import ModelPredictor, predictor
from src.ml.preprocessing import (
    categorize_academic_risk,
    get_preprocessor,
    prepare_features_and_targets,
    split_data,
)
from src.ml.validation import validate_dataset, validate_single_record


def test_categorize_academic_risk():
    """Verifies that grade thresholds map correctly to Portuguese risk categories."""
    assert categorize_academic_risk(0.0) == RISK_LABELS["high"]
    assert categorize_academic_risk(9.99) == RISK_LABELS["high"]
    assert categorize_academic_risk(10.0) == RISK_LABELS["medium"]
    assert categorize_academic_risk(13.99) == RISK_LABELS["medium"]
    assert categorize_academic_risk(14.0) == RISK_LABELS["low"]
    assert categorize_academic_risk(20.0) == RISK_LABELS["low"]
    assert categorize_academic_risk(np.nan) == "Unknown"


def test_validate_dataset_valid(sample_synthetic_dataframe):
    """Checks that a correctly formatted synthetic DataFrame passes validation."""
    report = validate_dataset(sample_synthetic_dataframe, require_target=True)
    assert report.is_valid is True
    assert report.total_rows == 5
    assert len(report.missing_columns) == 0
    assert len(report.errors) == 0


def test_validate_dataset_empty():
    """Verifies that an empty DataFrame is flagged as invalid."""
    empty_df = pd.DataFrame()
    report = validate_dataset(empty_df, require_target=False)
    assert report.is_valid is False
    assert "empty" in report.errors[0].lower()


def test_validate_dataset_missing_columns(sample_synthetic_dataframe):
    """Verifies that missing feature columns are caught and reported."""
    bad_df = sample_synthetic_dataframe.drop(columns=["age", "studytime"])
    report = validate_dataset(bad_df, require_target=False)
    assert report.is_valid is False
    assert "age" in report.missing_columns
    assert "studytime" in report.missing_columns


def test_validate_dataset_out_of_bounds(sample_synthetic_dataframe):
    """Verifies that illegal categories and numerical values out-of-range are caught."""
    bad_df = sample_synthetic_dataframe.copy()
    bad_df.loc[0, "school"] = "INVALID_NAME"
    bad_df.loc[1, "absences"] = 999.0
    bad_df.loc[2, "age"] = 40.0

    report = validate_dataset(bad_df, require_target=False)
    assert report.is_valid is False
    assert "school" in report.category_violations
    assert "absences" in report.range_violations
    assert "age" in report.range_violations


def test_validate_single_record(sample_valid_student, sample_invalid_student):
    """Tests single dictionary validation for valid and invalid records."""
    is_valid, errors = validate_single_record(sample_valid_student)
    assert is_valid is True
    assert len(errors) == 0

    is_invalid, err_list = validate_single_record(sample_invalid_student)
    assert is_invalid is False
    assert len(err_list) >= 4  # invalid school, age, Medu, failures, absences


def test_preprocessing_feature_isolation(sample_synthetic_dataframe):
    """Guarantees that G1, G2, and G3 are never in early-warning feature matrix X."""
    X, y_reg, y_clf = prepare_features_and_targets(sample_synthetic_dataframe)
    assert "G1" not in X.columns
    assert "G2" not in X.columns
    assert "G3" not in X.columns
    assert len(X.columns) == len(ALL_FEATURE_COLUMNS)
    assert len(y_reg) == len(sample_synthetic_dataframe)
    assert len(y_clf) == len(sample_synthetic_dataframe)


def test_preprocessor_column_transformer(sample_synthetic_dataframe):
    """Tests fitting and transformation of scikit-learn ColumnTransformer."""
    X, _, _ = prepare_features_and_targets(sample_synthetic_dataframe)
    preprocessor = get_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(X.columns)  # OneHotEncoder expands dimensions


def test_predictor_single_and_batch(sample_valid_student, sample_invalid_student):
    """Verifies ModelPredictor single and batch inference."""
    if not predictor.is_ready:
        pytest.skip("Model artifacts not yet generated.")

    # 1. Single valid prediction
    res = predictor.predict_single(sample_valid_student)
    assert "predicted_score" in res
    assert 0.0 <= res["predicted_score"] <= 20.0
    assert res["risk_category"] in list(RISK_LABELS.values())
    assert len(res["recommendations"]) > 0

    # 2. Single invalid prediction raises ValueError
    with pytest.raises(ValueError):
        predictor.predict_single(sample_invalid_student)

    # 3. Batch prediction with mixed rows
    mixed_df = pd.DataFrame([sample_valid_student, sample_invalid_student])
    results, summary = predictor.predict_batch(mixed_df)
    assert summary["total_records"] == 2
    assert summary["processed_successfully"] == 1
    assert summary["failed_validation"] == 1
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "rejected"
    assert len(results[1]["errors"]) > 0


def test_missing_model_handling():
    """Checks that ModelPredictor raises informative RuntimeError if models are missing."""
    dummy_predictor = ModelPredictor(
        regressor_path=Path("non_existent_regressor.joblib"),
        classifier_path=Path("non_existent_classifier.joblib"),
    )
    assert dummy_predictor.is_ready is False
    with pytest.raises(RuntimeError) as exc_info:
        dummy_predictor.predict_single({"school": "GP"})
    assert "not loaded" in str(exc_info.value).lower()
