"""
Unit tests for SQLite database persistence, SQLAlchemy models, and pagination.
"""
from datetime import datetime, timezone
import json
import pytest
from sqlalchemy.orm import Session

from src.database.models import PredictionLog
from src.services.prediction_service import clear_prediction_history, get_prediction_history


def test_prediction_log_crud(db_session: Session):
    """Verifies creation, querying, and serialization of PredictionLog records."""
    log = PredictionLog(
        timestamp=datetime.now(timezone.utc),
        prediction_type="single",
        model_version="1.0.0",
        predicted_score=14.50,
        risk_category="Low Academic Risk",
        record_count=1,
        metadata_json=json.dumps({"age": 17, "studytime": 3}),
    )
    db_session.add(log)
    db_session.commit()
    db_session.refresh(log)

    assert log.id is not None
    assert log.predicted_score == 14.50

    as_dict = log.to_dict()
    assert as_dict["id"] == log.id
    assert as_dict["prediction_type"] == "single"
    assert as_dict["risk_category"] == "Low Academic Risk"
    assert "studytime" in as_dict["metadata_json"]


def test_prediction_pagination(db_session: Session):
    """Tests page calculations and slice boundaries on audit history logs."""
    clear_prediction_history(db_session)

    # Insert 25 sample audit records
    for i in range(25):
        log = PredictionLog(
            timestamp=datetime.now(timezone.utc),
            prediction_type="single" if i % 2 == 0 else "batch_csv",
            model_version="1.0.0",
            predicted_score=10.0 + (i * 0.2),
            risk_category="Moderate Academic Risk",
            record_count=1,
        )
        db_session.add(log)
    db_session.commit()

    # Query page 1 (size 10)
    page1 = get_prediction_history(db_session, page=1, page_size=10)
    assert page1["total"] == 25
    assert page1["page"] == 1
    assert page1["total_pages"] == 3
    assert len(page1["items"]) == 10

    # Query page 3 (size 10) -> Should have 5 items
    page3 = get_prediction_history(db_session, page=3, page_size=10)
    assert len(page3["items"]) == 5

    # Clear history
    deleted = clear_prediction_history(db_session)
    assert deleted == 25

    # Query again -> Should be 0
    empty_page = get_prediction_history(db_session, page=1, page_size=10)
    assert empty_page["total"] == 0
    assert len(empty_page["items"]) == 0
