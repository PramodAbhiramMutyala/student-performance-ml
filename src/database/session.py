"""
SQLAlchemy database session and lifecycle management for SQLite.
"""
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.config import settings

# Ensure data directory exists
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite:///"):
    sqlite_file_path = Path(db_url.replace("sqlite:///", ""))
    sqlite_file_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Creates database tables if they do not exist."""
    import src.database.models  # Ensure models are registered with Base # noqa: F401
    Base.metadata.create_all(bind=engine)


def reset_db() -> None:
    """Drops and recreates all tables for testing or resetting demonstration data."""
    import src.database.models  # noqa: F401
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session and safely closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
