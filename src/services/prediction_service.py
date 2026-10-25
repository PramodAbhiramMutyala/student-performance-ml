"""
Prediction service.
Coordinates model inference with SQLite database logging and pagination.
"""
from datetime import datetime, timezone
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from src.database.models import PredictionLog
from src.ml.predict import predictor
from src.schemas.student import StudentFeatureInput


def process_single_prediction(
    student_data: StudentFeatureInput,
    db: Session,
) -> Dict[str, Any]:
    """
    Executes single prediction and logs non-identifying summary to SQLite.
    """
    input_dict = student_data.model_dump()
    result = predictor.predict_single(input_dict)

    # Minimal non-identifying metadata
    minimal_meta = {
        "age": input_dict.get("age"),
        "studytime": input_dict.get("studytime"),
        "failures": input_dict.get("failures"),
        "absences": input_dict.get("absences"),
    }

    log_entry = PredictionLog(
        timestamp=datetime.now(timezone.utc),
        prediction_type="single",
        model_version="1.0.0",
        predicted_score=result["predicted_score"],
        risk_category=result["risk_category"],
        record_count=1,
        metadata_json=json.dumps(minimal_meta),
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return result


def process_batch_dataframe(
    df: pd.DataFrame,
    source_type: str,
    db: Session,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes batch prediction with row-level error reporting,
    and logs batch aggregate audit entry to SQLite.
    """
    results, summary = predictor.predict_batch(df)

    log_entry = PredictionLog(
        timestamp=datetime.now(timezone.utc),
        prediction_type=source_type,
        model_version="1.0.0",
        predicted_score=summary.get("average_predicted_score"),
        risk_category=f"Batch: {summary.get('processed_successfully')}/{summary.get('total_records')} processed",
        record_count=summary.get("total_records", 0),
        metadata_json=json.dumps({
            "total": summary.get("total_records"),
            "successful": summary.get("processed_successfully"),
            "failed": summary.get("failed_validation"),
        }),
    )
    db.add(log_entry)
    db.commit()

    return results, summary


def get_prediction_history(
    db: Session,
    page: int = 1,
    page_size: int = 15,
) -> Dict[str, Any]:
    """
    Retrieves paginated prediction audit history from SQLite.
    """
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 15

    total = db.query(PredictionLog).count()
    total_pages = max(1, math.ceil(total / page_size))
    offset = (page - 1) * page_size

    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = [log.to_dict() for log in logs]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": items,
    }


def clear_prediction_history(db: Session) -> int:
    """Removes all stored prediction log entries from the database."""
    deleted_count = db.query(PredictionLog).delete()
    db.commit()
    return deleted_count
