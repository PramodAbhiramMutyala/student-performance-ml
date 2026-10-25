"""
Model evaluation module.
Computes regression (MAE, RMSE, R²) and classification (Accuracy, Precision,
Recall, F1, Confusion Matrix) metrics on test sets.
"""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def evaluate_regression_model(
    model: Pipeline,
    X_test: Any,
    y_test: Any,
) -> Dict[str, float]:
    """
    Evaluates a regression model on test set.
    Returns MAE, RMSE, R2, and Explained Variance.
    """
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    # root_mean_squared_error is standard in recent scikit-learn, fallback to np.sqrt(mean_squared_error)
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = float(root_mean_squared_error(y_test, y_pred))
    except ImportError:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    r2 = float(r2_score(y_test, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def evaluate_classification_model(
    model: Pipeline,
    X_test: Any,
    y_test: Any,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a classification model on test set.
    Returns Accuracy, Precision, Recall, F1 score, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)

    if labels is None:
        labels = sorted(list(set(y_test) | set(y_pred)))

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def extract_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts top feature importances or model coefficients from trained pipeline.
    """
    regressor = pipeline.named_steps.get("regressor") or pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not regressor or not preprocessor:
        return []

    # Try getting feature names out of preprocessor
    try:
        encoded_names = preprocessor.get_feature_names_out()
        cleaned_names = [name.replace("num__", "").replace("cat__", "") for name in encoded_names]
    except Exception:
        cleaned_names = [f"feat_{i}" for i in range(100)]

    importances: List[Dict[str, Any]] = []

    if hasattr(regressor, "feature_importances_"):
        raw_vals = regressor.feature_importances_
        for name, val in zip(cleaned_names, raw_vals):
            importances.append({"feature": name, "importance": round(float(val), 4)})
    elif hasattr(regressor, "coef_"):
        raw_vals = np.abs(regressor.coef_)
        if raw_vals.ndim > 1:
            raw_vals = np.mean(raw_vals, axis=0)
        for name, val in zip(cleaned_names, raw_vals):
            importances.append({"feature": name, "importance": round(float(val), 4)})

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances[:15]  # Top 15 most important features
