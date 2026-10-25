"""
Inference module for Student Performance & Academic Risk Prediction.
Loads serialized pipelines, validates inputs, predicts regression scores,
and computes academic risk classifications.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURE_COLUMNS,
    CLASSIFIER_MODEL_PATH,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    REGRESSOR_MODEL_PATH,
    RISK_LABELS,
    RISK_RECOMMENDATIONS,
)
from src.ml.preprocessing import categorize_academic_risk
from src.ml.validation import validate_single_record

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Manages loading and inference for the regression and risk classification pipelines.
    Gracefully handles cases where model artifacts have not yet been trained.
    """

    def __init__(
        self,
        regressor_path: Path = REGRESSOR_MODEL_PATH,
        classifier_path: Path = CLASSIFIER_MODEL_PATH,
    ):
        self.regressor_path = Path(regressor_path)
        self.classifier_path = Path(classifier_path)
        self.regressor = None
        self.classifier = None
        self.load_models()

    def load_models(self) -> None:
        """Loads trained pipelines from disk if available."""
        if self.regressor_path.exists():
            try:
                self.regressor = joblib.load(self.regressor_path)
                logger.info("Loaded regression model from %s", self.regressor_path)
            except Exception as e:
                logger.error("Failed to load regression model: %s", e)
                self.regressor = None
        else:
            logger.warning("Regressor artifact not found at %s", self.regressor_path)
            self.regressor = None

        if self.classifier_path.exists():
            try:
                self.classifier = joblib.load(self.classifier_path)
                logger.info("Loaded classifier model from %s", self.classifier_path)
            except Exception as e:
                logger.error("Failed to load classifier model: %s", e)
                self.classifier = None
        else:
            logger.warning("Classifier artifact not found at %s", self.classifier_path)
            self.classifier = None

    @property
    def is_ready(self) -> bool:
        """Returns True if the primary regression model is loaded."""
        return self.regressor is not None

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a single student record and returns the predicted grade (0-20),
        academic risk tier, and targeted interventions.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Machine learning model artifact is not loaded. "
                "Please run 'python -m src.ml.train' to train and save the model pipeline."
            )

        # 1. Validation
        is_valid, errors = validate_single_record(input_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        # 2. DataFrame formatting
        df_row = pd.DataFrame([{col: input_data[col] for col in ALL_FEATURE_COLUMNS}])

        # 3. Continuous grade prediction
        raw_pred = float(self.regressor.predict(df_row)[0])
        clamped_score = max(0.0, min(20.0, raw_pred))
        rounded_score = round(clamped_score, 2)

        # 4. Academic risk category prediction
        risk_category = None
        probabilities: Dict[str, float] = {}

        if self.classifier is not None:
            try:
                pred_label = self.classifier.predict(df_row)[0]
                risk_category = str(pred_label)

                if hasattr(self.classifier, "predict_proba"):
                    classes = self.classifier.classes_
                    proba_vals = self.classifier.predict_proba(df_row)[0]
                    probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, proba_vals)}
            except Exception as e:
                logger.warning("Classifier prediction failed; falling back to score threshold: %s", e)
                risk_category = categorize_academic_risk(rounded_score)
        else:
            risk_category = categorize_academic_risk(rounded_score)

        recommendations = RISK_RECOMMENDATIONS.get(risk_category, [
            "Monitor ongoing academic milestones.",
            "Attend regular scheduled subject review sessions."
        ])

        return {
            "predicted_score": rounded_score,
            "raw_score": round(raw_pred, 4),
            "risk_category": risk_category,
            "risk_probabilities": probabilities,
            "scale": "0 to 20 Portuguese secondary mathematics scale",
            "passing_threshold": HIGH_RISK_THRESHOLD,
            "recommendations": recommendations,
            "disclaimer": (
                "Academic predictions are statistical estimations intended solely for proactive "
                "early counseling support. They must never be used for punitive or high-stakes decisions."
            ),
        }

    def predict_batch(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Processes a batch DataFrame with row-level validation.
        Valid rows receive predictions; invalid rows are flagged with specific reasons.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Machine learning model artifact is not loaded. "
                "Please run 'python -m src.ml.train' to train and save the model pipeline."
            )

        results: List[Dict[str, Any]] = []
        valid_rows: List[Dict[str, Any]] = []
        valid_indices: List[int] = []

        total_rows = len(df)
        success_count = 0
        failure_count = 0

        # Validate rows individually
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            is_valid, errors = validate_single_record(row_dict)

            if is_valid:
                valid_rows.append({col: row_dict[col] for col in ALL_FEATURE_COLUMNS})
                valid_indices.append(idx)
            else:
                failure_count += 1
                results.append({
                    "row_index": idx + 1,
                    "status": "rejected",
                    "predicted_score": None,
                    "risk_category": None,
                    "errors": errors,
                })

        # Batch predict valid rows
        if valid_rows:
            batch_df = pd.DataFrame(valid_rows)
            preds = self.regressor.predict(batch_df)

            clf_preds = [None] * len(valid_rows)
            if self.classifier is not None:
                try:
                    clf_preds = self.classifier.predict(batch_df)
                except Exception:
                    pass

            for idx, p_score, c_label in zip(valid_indices, preds, clf_preds):
                clamped = round(max(0.0, min(20.0, float(p_score))), 2)
                risk_label = str(c_label) if c_label is not None else categorize_academic_risk(clamped)
                success_count += 1
                results.append({
                    "row_index": idx + 1,
                    "status": "success",
                    "predicted_score": clamped,
                    "risk_category": risk_label,
                    "errors": [],
                })

        # Re-sort results to preserve original input row order
        results.sort(key=lambda x: x["row_index"])

        summary = {
            "total_records": total_rows,
            "processed_successfully": success_count,
            "failed_validation": failure_count,
            "average_predicted_score": round(
                float(np.mean([r["predicted_score"] for r in results if r["predicted_score"] is not None])), 2
            ) if success_count > 0 else 0.0,
        }

        return results, summary


# Global predictor singleton
predictor = ModelPredictor()
