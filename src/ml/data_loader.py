"""
Data loader module for Student Performance dataset.
Handles file reading, delimiter detection, and basic column hygiene.
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.config import RAW_DATA_PATH


def load_dataset(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads student performance data from a CSV file.
    Supports either semicolon or comma delimiters.
    """
    path = Path(file_path) if file_path else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. "
            "Please run 'python scripts/download_dataset.py' to acquire the UCI dataset."
        )

    # Detect separator (UCI student data defaults to semicolon)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # Fallback to semicolon if engine sniffing failed
        df = pd.read_csv(path, sep=";")

    # Strip any extraneous whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    return df


def save_processed_dataset(df: pd.DataFrame, target_path: Union[str, Path]) -> Path:
    """Saves preprocessed dataset to CSV."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
