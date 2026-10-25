"""
REST API route definitions.
Implements model inference, batch execution, file validation, analytics,
and audit history endpoints.
"""
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.dependencies import get_active_predictor, get_database_session
from src.config import settings
from src.database.session import get_db
from src.ml.predict import ModelPredictor
from src.ml.validation import validate_dataset
from src.schemas.prediction import (
    BatchPredictionJSONRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PaginatedPredictionsResponse,
    PredictionResponse,
    ValidationResponse,
)
from src.schemas.student import StudentFeatureInput
from src.services.analytics_service import (
    get_analytics_overview,
    get_distributions_data,
    get_saved_model_metadata,
    get_saved_model_metrics,
)
from src.services.prediction_service import (
    clear_prediction_history,
    get_prediction_history,
    process_batch_dataframe,
    process_single_prediction,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def check_health(
    db: Session = Depends(get_database_session),
    predictor: ModelPredictor = Depends(get_active_predictor),
) -> HealthResponse:
    """Returns application status, model readiness, and database connectivity."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error("Health check DB query failed: %s", e)

    return HealthResponse(
        status="healthy" if (predictor.is_ready and db_connected) else "degraded",
        model_loaded=predictor.is_ready,
        classifier_loaded=predictor.classifier is not None,
        database_connected=db_connected,
        app_version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/api/analytics/overview", tags=["Analytics"])
def get_overview_metrics() -> Dict[str, Any]:
    """Returns dataset size, observed grade statistics, and active model metrics."""
    return get_analytics_overview()


@router.get("/api/analytics/distributions", tags=["Analytics"])
def get_distributions() -> Dict[str, Any]:
    """Returns aggregated distribution data for grades, study time, and absences."""
    return get_distributions_data()


@router.get("/api/model/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_information(predictor: ModelPredictor = Depends(get_active_predictor)) -> ModelInfoResponse:
    """Returns active model details, version, target definition, and validation metrics."""
    metrics = get_saved_model_metrics() or {}
    meta = get_saved_model_metadata() or {}

    return ModelInfoResponse(
        model_version=meta.get("model_version", settings.APP_VERSION),
        experiment_type=meta.get("experiment_type", "Early-Warning Academic Performance Prediction"),
        target_variable=meta.get("target_variable", "G3 (Final Mathematics Grade)"),
        excluded_features=meta.get("excluded_features", ["G1", "G2", "G3"]),
        selected_regressor=meta.get("selected_regressor", "Not trained"),
        selected_classifier=meta.get("selected_classifier", "Not trained"),
        metrics=metrics,
    )


@router.get("/api/metrics", tags=["Model"])
def get_metrics() -> Dict[str, Any]:
    """Returns empirical candidate comparison and evaluation metrics."""
    metrics = get_saved_model_metrics()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model evaluation metrics not found. Please train models via 'python -m src.ml.train'."
        )
    return metrics


@router.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_student_grade(
    student: StudentFeatureInput,
    db: Session = Depends(get_database_session),
    predictor: ModelPredictor = Depends(get_active_predictor),
) -> PredictionResponse:
    """
    Accepts validated student feature values and returns predicted final grade (0-20),
    academic risk classification, and tailored interventions.
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run 'python -m src.ml.train' to train model.",
        )

    try:
        result = process_single_prediction(student, db)
        return PredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error("Single prediction failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@router.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_students_batch(
    payload: BatchPredictionJSONRequest,
    db: Session = Depends(get_database_session),
    predictor: ModelPredictor = Depends(get_active_predictor),
) -> BatchPredictionResponse:
    """
    Accepts a list of student records in JSON format and executes batch inference with row-level validation.
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run 'python -m src.ml.train'.",
        )

    if len(payload.students) > settings.MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size exceeds maximum allowed of {settings.MAX_BATCH_ROWS} records.",
        )

    # Convert Pydantic items to DataFrame
    rows_data = [item.model_dump() for item in payload.students]
    df = pd.DataFrame(rows_data)

    results, summary = process_batch_dataframe(df, source_type="batch_json", db=db)
    return BatchPredictionResponse(summary=summary, results=results)


def _safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Safely decodes and loads a CSV from memory, handling delimiter sniffing."""
    try:
        decoded_str = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded_str = file_bytes.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid text-encoded CSV.")

    buffer = io.StringIO(decoded_str)
    try:
        df = pd.read_csv(buffer, sep=None, engine="python")
    except Exception:
        buffer.seek(0)
        try:
            df = pd.read_csv(buffer, sep=";")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")

    df.columns = [c.strip() for c in df.columns]
    return df


@router.post("/api/dataset/validate", response_model=ValidationResponse, tags=["Dataset"])
async def validate_uploaded_csv(
    file: UploadFile = File(...),
) -> ValidationResponse:
    """
    Accepts an uploaded CSV file, validates columns, data types, missing values,
    and returns a structured validation report without executing predictions.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
        )

    df = _safe_read_csv(contents)

    if len(df) > settings.MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file contains {len(df)} rows, exceeding limit of {settings.MAX_BATCH_ROWS}."
        )

    report = validate_dataset(df, require_target=False)
    return ValidationResponse(**report.to_dict())


@router.post("/api/dataset/predict", tags=["Inference"])
async def predict_csv_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_database_session),
    predictor: ModelPredictor = Depends(get_active_predictor),
) -> StreamingResponse:
    """
    Accepts an uploaded CSV file, runs row-level validation and inference,
    and returns a downloadable CSV with predicted scores and risk categories appended.
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Please run 'python -m src.ml.train'."
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds maximum size of 5MB.")

    df = _safe_read_csv(contents)
    if len(df) > settings.MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum of {settings.MAX_BATCH_ROWS} rows.")

    results, summary = process_batch_dataframe(df, source_type="batch_csv", db=db)

    # Build output dataframe
    output_df = df.copy()
    output_df["prediction_status"] = [r["status"] for r in results]
    output_df["predicted_G3"] = [r["predicted_score"] for r in results]
    output_df["academic_risk_category"] = [r["risk_category"] for r in results]
    output_df["validation_errors"] = ["; ".join(r["errors"]) if r["errors"] else "" for r in results]

    stream = io.StringIO()
    output_df.to_csv(stream, index=False)
    stream.seek(0)

    download_filename = f"predictions_{Path(file.filename).stem}.csv"
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={download_filename}"},
    )
    return response


@router.get("/api/predictions", response_model=PaginatedPredictionsResponse, tags=["History"])
def list_prediction_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(15, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_database_session),
) -> PaginatedPredictionsResponse:
    """Returns paginated historical prediction logs from SQLite."""
    history_data = get_prediction_history(db, page=page, page_size=page_size)
    return PaginatedPredictionsResponse(**history_data)


@router.delete("/api/predictions", tags=["History"])
def delete_prediction_history(db: Session = Depends(get_database_session)) -> Dict[str, Any]:
    """Clears all demonstration prediction logs after user confirmation."""
    deleted_count = clear_prediction_history(db)
    return {"message": f"Successfully cleared {deleted_count} prediction audit records.", "deleted_count": deleted_count}
