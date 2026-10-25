"""
Model training and evaluation pipeline.
Generates EDA artifacts, trains candidate regression and classification models,
performs cross-validation, selects best models, and serializes artifacts.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURE_COLUMNS,
    CLASSIFIER_MODEL_PATH,
    EDA_DIR,
    METADATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    REGRESSOR_MODEL_PATH,
    RISK_LABELS,
    TARGET_COLUMN,
)
from src.ml.data_loader import load_dataset
from src.ml.evaluate import (
    evaluate_classification_model,
    evaluate_regression_model,
    extract_feature_importance,
)
from src.ml.preprocessing import (
    get_preprocessor,
    prepare_features_and_targets,
    split_data,
)
from src.ml.validation import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_eda_artifacts(df: pd.DataFrame, output_dir: Path) -> List[str]:
    """Generates and saves exploratory data analysis charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # 1. Final Grade (G3) Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=21, color="#0d9488", ax=ax)
    ax.set_title("Distribution of Final Mathematics Grades (G3, 0-20 Scale)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Student Count")
    ax.axvline(10, color="#ef4444", linestyle="--", label="Passing Threshold (10)")
    ax.axvline(df[TARGET_COLUMN].mean(), color="#3b82f6", linestyle="-", label=f"Mean Grade ({df[TARGET_COLUMN].mean():.2f})")
    ax.legend()
    g3_path = output_dir / "grade_distribution.png"
    fig.savefig(g3_path, dpi=200)
    plt.close(fig)
    generated_files.append(str(g3_path))

    # 2. Correlation Heatmap
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns] + [TARGET_COLUMN]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True)
    ax.set_title("Correlation Heatmap of Numerical Indicators and Final Grade", fontsize=13, fontweight="bold", pad=12)
    corr_path = output_dir / "correlation_heatmap.png"
    fig.savefig(corr_path, dpi=200)
    plt.close(fig)
    generated_files.append(str(corr_path))

    # 3. Study Time vs Final Grade
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="studytime", y=TARGET_COLUMN, data=df, palette="crest", ax=ax)
    ax.set_title("Weekly Study Time vs Final Mathematics Grade (G3)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    ax.set_ylabel("Final Grade (G3)")
    study_path = output_dir / "studytime_vs_grade.png"
    fig.savefig(study_path, dpi=200)
    plt.close(fig)
    generated_files.append(str(study_path))

    # 4. Absences vs Final Grade
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="absences", y=TARGET_COLUMN, data=df, hue="sex", alpha=0.8, s=60, palette="viridis", ax=ax)
    ax.set_title("School Absences vs Final Grade (G3)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of School Absences")
    ax.set_ylabel("Final Grade (G3)")
    abs_path = output_dir / "absences_vs_grade.png"
    fig.savefig(abs_path, dpi=200)
    plt.close(fig)
    generated_files.append(str(abs_path))

    # 5. Failures vs Final Grade
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x="failures", y=TARGET_COLUMN, data=df, palette="magma", ax=ax)
    ax.set_title("Past Academic Failures vs Final Grade (G3)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Past Class Failures (0 to 3+)")
    ax.set_ylabel("Final Grade (G3)")
    fail_path = output_dir / "failures_vs_grade.png"
    fig.savefig(fail_path, dpi=200)
    plt.close(fig)
    generated_files.append(str(fail_path))

    logger.info("Generated %d EDA visual artifacts in %s", len(generated_files), output_dir)
    return generated_files


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes end-to-end dataset validation, EDA plot generation, model training,
    cross-validation, test-set evaluation, and artifact serialization.
    """
    logger.info("Step 1: Loading raw dataset...")
    df = load_dataset()
    logger.info("Loaded dataset with shape: %s", df.shape)

    logger.info("Step 2: Validating dataset schema...")
    val_report = validate_dataset(df, require_target=True)
    if not val_report.is_valid:
        raise ValueError(f"Dataset validation failed: {val_report.errors}")
    logger.info("Dataset validation passed with 0 critical errors.")

    logger.info("Step 3: Generating EDA visual artifacts...")
    generate_eda_artifacts(df, EDA_DIR)

    logger.info("Step 4: Preparing features and target (Early-Warning experiment)...")
    # Exclusion of G1 and G2 guaranteed by prepare_features_and_targets
    X, y_reg, y_clf = prepare_features_and_targets(df)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    # ------------------ REGRESSION EXPERIMENT ------------------
    logger.info("Step 5: Training and evaluating candidate regression models...")
    regression_candidates: Dict[str, Any] = {
        "DummyRegressor (Baseline)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_comparison: Dict[str, Dict[str, Any]] = {}
    best_reg_name = ""
    best_reg_pipeline = None
    best_reg_rmse = float("inf")

    for name, model_inst in regression_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", get_preprocessor()),
                ("regressor", model_inst),
            ]
        )

        # 5-fold cross-validation on training set only (negative MAE and RMSE)
        cv_mae = -cross_val_score(pipeline, X_train, y_reg_train, cv=kf, scoring="neg_mean_absolute_error").mean()
        cv_rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_reg_train, cv=kf, scoring="neg_mean_squared_error").mean())

        # Fit on full training set
        pipeline.fit(X_train, y_reg_train)

        # Test set evaluation
        test_metrics = evaluate_regression_model(pipeline, X_test, y_reg_test)

        reg_comparison[name] = {
            "cv_mae": round(float(cv_mae), 4),
            "cv_rmse": round(float(cv_rmse), 4),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }

        logger.info(
            "[%s] CV RMSE: %.4f | Test MAE: %.4f | Test RMSE: %.4f | Test R2: %.4f",
            name, cv_rmse, test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"]
        )

        # Selection criteria: Lowest CV RMSE among candidate models
        if cv_rmse < best_reg_rmse:
            best_reg_rmse = cv_rmse
            best_reg_name = name
            best_reg_pipeline = pipeline

    logger.info("Selected best regression model: %s (CV RMSE: %.4f)", best_reg_name, best_reg_rmse)

    # ------------------ CLASSIFICATION EXPERIMENT ------------------
    logger.info("Step 6: Training and evaluating academic risk classification models...")
    risk_labels_order = [RISK_LABELS["high"], RISK_LABELS["medium"], RISK_LABELS["low"]]

    classification_candidates: Dict[str, Any] = {
        "DummyClassifier (Baseline)": DummyClassifier(strategy="most_frequent"),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42),
    }

    clf_comparison: Dict[str, Dict[str, Any]] = {}
    best_clf_name = ""
    best_clf_pipeline = None
    best_clf_f1 = -1.0

    for name, model_inst in classification_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", get_preprocessor()),
                ("classifier", model_inst),
            ]
        )

        pipeline.fit(X_train, y_clf_train)
        test_metrics = evaluate_classification_model(pipeline, X_test, y_clf_test, labels=risk_labels_order)

        clf_comparison[name] = test_metrics
        logger.info(
            "[%s] Test Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
            name, test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["f1_weighted"]
        )

        if test_metrics["f1_macro"] > best_clf_f1:
            best_clf_f1 = test_metrics["f1_macro"]
            best_clf_name = name
            best_clf_pipeline = pipeline

    logger.info("Selected best classification model: %s (Macro F1: %.4f)", best_clf_name, best_clf_f1)

    # ------------------ SERIALIZATION ------------------
    logger.info("Step 7: Serializing model artifacts to disk...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_reg_pipeline, REGRESSOR_MODEL_PATH)
    joblib.dump(best_clf_pipeline, CLASSIFIER_MODEL_PATH)

    feature_importances = extract_feature_importance(best_reg_pipeline, ALL_FEATURE_COLUMNS)

    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regression": {
            "selected_model": best_reg_name,
            "candidates": reg_comparison,
            "best_model_test_metrics": reg_comparison[best_reg_name],
            "feature_importance": feature_importances,
        },
        "classification": {
            "selected_model": best_clf_name,
            "candidates": {k: {m: v for m, v in vals.items() if m != "classification_report"} for k, vals in clf_comparison.items()},
            "best_model_test_metrics": clf_comparison[best_clf_name],
        },
    }

    metadata_payload = {
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "UCI Student Performance (Mathematics)",
        "dataset_records_count": len(df),
        "features_count": len(ALL_FEATURE_COLUMNS),
        "target_variable": TARGET_COLUMN,
        "excluded_features": ["G1", "G2", "G3"],
        "experiment_type": "Early-Warning Academic Performance Prediction",
        "selected_regressor": best_reg_name,
        "selected_classifier": best_clf_name,
        "risk_categories": list(RISK_LABELS.values()),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    logger.info("Successfully saved model artifacts to %s", MODELS_DIR)

    # Print clean readable console summary
    print("\n" + "=" * 65)
    print("      STUDENT PERFORMANCE ML - TRAINING PIPELINE SUMMARY      ")
    print("=" * 65)
    print(f"Dataset Size           : {len(df)} records, {len(ALL_FEATURE_COLUMNS)} early-warning features")
    print(f"Target Column          : {TARGET_COLUMN} (Excluding G1, G2, G3)")
    print(f"Best Regressor Model   : {best_reg_name}")
    print(f"  - Test MAE           : {reg_comparison[best_reg_name]['test_mae']:.4f}")
    print(f"  - Test RMSE          : {reg_comparison[best_reg_name]['test_rmse']:.4f}")
    print(f"  - Test R²            : {reg_comparison[best_reg_name]['test_r2']:.4f}")
    print(f"Best Risk Classifier   : {best_clf_name}")
    print(f"  - Test Accuracy      : {clf_comparison[best_clf_name]['accuracy']:.4f}")
    print(f"  - Test Macro F1      : {clf_comparison[best_clf_name]['f1_macro']:.4f}")
    print(f"Saved Regressor File   : {REGRESSOR_MODEL_PATH}")
    print(f"Saved Classifier File  : {CLASSIFIER_MODEL_PATH}")
    print(f"Metrics & Metadata     : {METRICS_PATH}, {METADATA_PATH}")
    print("=" * 65 + "\n")

    return metrics_payload


if __name__ == "__main__":
    run_training_pipeline()
