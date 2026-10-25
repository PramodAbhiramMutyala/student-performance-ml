"""
Shared pytest fixtures and test environment setup.
Uses synthetic student fixtures and an in-memory SQLite database.
"""
from typing import Any, Dict, Generator
from fastapi.testclient import TestClient
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.dependencies import get_database_session
from src.database.session import Base
from src.main import app

# In-memory SQLite engine for automated unit tests
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initializes in-memory database tables for test session."""
    import src.database.models  # noqa: F401
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provides a transactional database session per test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with overridden database session dependency."""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_database_session] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_valid_student() -> Dict[str, Any]:
    """Synthetic valid student feature record."""
    return {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 2,
    }


@pytest.fixture
def sample_invalid_student() -> Dict[str, Any]:
    """Synthetic student record with out-of-range bounds and bad category."""
    return {
        "school": "INVALID_SCHOOL",  # invalid domain
        "sex": "F",
        "age": 35,  # age > 22
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 9,  # Medu > 4
        "Fedu": 3,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 3,
        "failures": 10,  # failures > 4
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 500,  # absences > 93
    }


@pytest.fixture
def sample_synthetic_dataframe(sample_valid_student) -> pd.DataFrame:
    """Returns a synthetic 5-row pandas DataFrame with varied features for ML tests."""
    rows = []
    schools = ["GP", "MS", "GP", "MS", "GP"]
    sexes = ["F", "M", "F", "M", "F"]
    jobs = ["services", "teacher", "health", "at_home", "other"]

    for i in range(5):
        row = dict(sample_valid_student)
        row["school"] = schools[i]
        row["sex"] = sexes[i]
        row["Mjob"] = jobs[i]
        row["age"] = 16 + (i % 4)
        row["studytime"] = (i % 4) + 1
        row["absences"] = i * 2
        row["G1"] = 10 + i
        row["G2"] = 11 + i
        row["G3"] = 12 + i
        rows.append(row)
    return pd.DataFrame(rows)
