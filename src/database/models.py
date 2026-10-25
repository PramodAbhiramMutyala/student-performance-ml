"""
SQLAlchemy ORM models for storing prediction audit history.
Does not persist complete student private profiles.
"""
from datetime import datetime, timezone
from typing import Any, Dict
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from src.database.session import Base


class PredictionLog(Base):
    """
    Stores non-identifying historical logs of prediction requests and outcomes.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prediction_type = Column(String(50), nullable=False)  # "single", "batch_csv", "batch_json"
    model_version = Column(String(20), default="1.0.0", nullable=False)
    predicted_score = Column(Float, nullable=True)  # Clamped 0-20 score (or batch average)
    risk_category = Column(String(50), nullable=True)  # High, Moderate, Low Risk
    record_count = Column(Integer, default=1, nullable=False)
    metadata_json = Column(Text, nullable=True)  # Non-identifying key inputs summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "prediction_type": self.prediction_type,
            "model_version": self.model_version,
            "predicted_score": round(self.predicted_score, 2) if self.predicted_score is not None else None,
            "risk_category": self.risk_category,
            "record_count": self.record_count,
            "metadata_json": self.metadata_json,
        }
