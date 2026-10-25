"""
FastAPI route dependencies.
Provides database sessions and verified model predictor instances.
"""
from typing import Generator
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.ml.predict import ModelPredictor, predictor


def get_database_session() -> Generator[Session, None, None]:
    """Yields a database session."""
    yield from get_db()


def get_active_predictor() -> ModelPredictor:
    """Returns the ModelPredictor instance, verifying readiness."""
    if not predictor.is_ready:
        predictor.load_models()
    return predictor
