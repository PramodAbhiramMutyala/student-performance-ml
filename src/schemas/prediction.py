"""
Pydantic schemas for API predictions, validation, analytics, and history.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.schemas.student import StudentFeatureInput


class PredictionResponse(BaseModel):
    """Result of a single student performance prediction."""
    predicted_score: float = Field(..., description="Estimated final grade (0-20 scale)")
    raw_score: float = Field(..., description="Unclamped model prediction output")
    risk_category: str = Field(..., description="Academic risk tier (High, Moderate, or Low)")
    risk_probabilities: Dict[str, float] = Field(default_factory=dict, description="Risk class probabilities")
    scale: str = Field(default="0 to 20 Portuguese secondary mathematics scale")
    passing_threshold: float = Field(default=10.0)
    recommendations: List[str] = Field(default_factory=list, description="Targeted educational interventions")
    disclaimer: str = Field(..., description="Academic caution disclaimer")


class BatchPredictionItem(BaseModel):
    """Validation and prediction outcome for a single row in a batch."""
    row_index: int
    status: str  # "success" or "rejected"
    predicted_score: Optional[float] = None
    risk_category: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class BatchPredictionResponse(BaseModel):
    """Summary and row-level breakdown for batch predictions."""
    summary: Dict[str, Any]
    results: List[BatchPredictionItem]


class BatchPredictionJSONRequest(BaseModel):
    """Request body for submitting multiple student records in JSON format."""
    students: List[StudentFeatureInput]


class ValidationResponse(BaseModel):
    """Outcome of dataset or CSV schema validation."""
    is_valid: bool
    total_rows: int
    total_columns: int
    missing_values_count: int
    duplicate_rows_count: int
    missing_columns: List[str]
    unexpected_columns: List[str]
    category_violations: Dict[str, List[str]]
    range_violations: Dict[str, Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    row_errors_count: int


class HealthResponse(BaseModel):
    """Application and subsystem health indicator."""
    status: str
    model_loaded: bool
    classifier_loaded: bool
    database_connected: bool
    app_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Metadata regarding currently active machine learning model."""
    model_version: str
    experiment_type: str
    target_variable: str
    excluded_features: List[str]
    selected_regressor: str
    selected_classifier: str
    metrics: Dict[str, Any]


class PredictionLogItem(BaseModel):
    """Historical prediction log record."""
    id: int
    timestamp: Optional[str]
    prediction_type: str
    model_version: str
    predicted_score: Optional[float]
    risk_category: Optional[str]
    record_count: int
    metadata_json: Optional[str]


class PaginatedPredictionsResponse(BaseModel):
    """Paginated collection of audit prediction history."""
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[PredictionLogItem]
