"""
Analytics service.
Extracts empirical statistical metrics, distributions, and model performance
indicators directly from the dataset and serialized training records.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    HIGH_RISK_THRESHOLD,
    METADATA_PATH,
    METRICS_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.ml.data_loader import load_dataset


def get_saved_model_metrics() -> Optional[Dict[str, Any]]:
    """Loads saved model evaluation metrics JSON."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_saved_model_metadata() -> Optional[Dict[str, Any]]:
    """Loads saved model metadata JSON."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_analytics_overview() -> Dict[str, Any]:
    """
    Computes global summary statistics for dashboard and overview endpoints.
    """
    try:
        df = load_dataset()
    except Exception:
        # Fallback if raw dataset not yet downloaded
        return {
            "dataset_available": False,
            "total_students": 0,
            "average_grade": 0.0,
            "passing_rate": 0.0,
            "failing_rate": 0.0,
            "average_absences": 0.0,
            "average_studytime": 0.0,
            "model_metrics": get_saved_model_metrics(),
        }

    total_count = len(df)
    grades = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else pd.Series([])

    avg_grade = float(grades.mean()) if not grades.empty else 0.0
    passing_count = int((grades >= HIGH_RISK_THRESHOLD).sum()) if not grades.empty else 0
    failing_count = total_count - passing_count
    passing_rate = round((passing_count / total_count) * 100, 2) if total_count > 0 else 0.0
    failing_rate = round((failing_count / total_count) * 100, 2) if total_count > 0 else 0.0

    avg_absences = float(df["absences"].mean()) if "absences" in df.columns else 0.0
    avg_studytime = float(df["studytime"].mean()) if "studytime" in df.columns else 0.0

    return {
        "dataset_available": True,
        "total_students": total_count,
        "average_grade": round(avg_grade, 2),
        "passing_rate": passing_rate,
        "failing_rate": failing_rate,
        "average_absences": round(avg_absences, 2),
        "average_studytime": round(avg_studytime, 2),
        "model_metrics": get_saved_model_metrics(),
        "model_metadata": get_saved_model_metadata(),
    }


def get_distributions_data() -> Dict[str, Any]:
    """
    Produces real aggregated distribution data for Chart.js dashboard charts.
    """
    try:
        df = load_dataset()
    except Exception:
        return {"error": "Dataset not available"}

    # 1. Final Grade Distribution (0 to 20 integers)
    grade_bins = list(range(21))
    grade_counts = df[TARGET_COLUMN].value_counts().reindex(grade_bins, fill_value=0).to_dict()

    # 2. Study Time vs Average Grade
    # studytime: 1 (<2h), 2 (2-5h), 3 (5-10h), 4 (>10h)
    study_map = {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}
    study_grp = df.groupby("studytime")[TARGET_COLUMN].agg(["mean", "count"]).reset_index()
    study_data = []
    for _, row in study_grp.iterrows():
        st_val = int(row["studytime"])
        study_data.append({
            "category": study_map.get(st_val, f"Level {st_val}"),
            "average_grade": round(float(row["mean"]), 2),
            "student_count": int(row["count"]),
        })

    # 3. Absence Intervals vs Average Grade
    # Interval bins: 0, 1-5, 6-10, 11-20, 21+
    def bucket_absences(val: int) -> str:
        if val == 0:
            return "0 absences"
        elif 1 <= val <= 5:
            return "1 - 5 absences"
        elif 6 <= val <= 10:
            return "6 - 10 absences"
        elif 11 <= val <= 20:
            return "11 - 20 absences"
        else:
            return "21+ absences"

    df_copy = df.copy()
    df_copy["absence_bucket"] = df_copy["absences"].apply(bucket_absences)
    bucket_order = ["0 absences", "1 - 5 absences", "6 - 10 absences", "11 - 20 absences", "21+ absences"]
    abs_grp = df_copy.groupby("absence_bucket")[TARGET_COLUMN].agg(["mean", "count"]).reindex(bucket_order).reset_index()

    absences_data = []
    for _, row in abs_grp.iterrows():
        if pd.notna(row["absence_bucket"]):
            absences_data.append({
                "bucket": row["absence_bucket"],
                "average_grade": round(float(row["mean"]), 2) if pd.notna(row["mean"]) else 0.0,
                "count": int(row["count"]) if pd.notna(row["count"]) else 0,
            })

    # 4. Academic Risk Tier breakdown from true dataset
    high_risk_count = int((df[TARGET_COLUMN] < HIGH_RISK_THRESHOLD).sum())
    mod_risk_count = int(((df[TARGET_COLUMN] >= HIGH_RISK_THRESHOLD) & (df[TARGET_COLUMN] < 14)).sum())
    low_risk_count = int((df[TARGET_COLUMN] >= 14).sum())

    risk_data = {
        "High Academic Risk (<10)": high_risk_count,
        "Moderate Academic Risk (10-13)": mod_risk_count,
        "Low Academic Risk (14-20)": low_risk_count,
    }

    return {
        "grades": {
            "labels": [str(g) for g in grade_bins],
            "values": [int(grade_counts[g]) for g in grade_bins],
        },
        "study_time": study_data,
        "absences": absences_data,
        "risk_breakdown": risk_data,
    }
