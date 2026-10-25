"""
Demo data generator.
Creates sample valid and intentionally malformed CSV files for manual and automated testing.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student-mat.csv"


def generate_demo_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        print(f"[ERROR] Raw dataset not found at {RAW_DATA_PATH}. Run download_dataset.py first.")
        return

    # Load raw dataset
    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    # Drop target and intermediate grades to simulate early-warning batch
    features_only = df.drop(columns=["G1", "G2", "G3"])

    # 1. Valid Batch (25 sample students)
    valid_batch = features_only.sample(n=25, random_state=42).reset_index(drop=True)
    valid_file = DATA_DIR / "demo_valid_batch.csv"
    valid_batch.to_csv(valid_file, index=False)
    print(f"[SUCCESS] Created valid demo batch: {valid_file} ({len(valid_batch)} rows)")

    # 2. Batch with intentional errors for testing row-level validation
    invalid_batch = valid_batch.head(5).copy().astype(object)
    # Row 1: Invalid category for school and Medu out of range
    invalid_batch.loc[0, "school"] = "INVALID_SCHOOL"
    invalid_batch.loc[0, "Medu"] = 99
    # Row 3: Absences out of range
    invalid_batch.loc[2, "absences"] = 999
    # Row 4: Age non-numeric string
    invalid_batch.loc[3, "age"] = "not_a_number"

    invalid_file = DATA_DIR / "demo_invalid_batch.csv"
    invalid_batch.to_csv(invalid_file, index=False)
    print(f"[SUCCESS] Created invalid test batch: {invalid_file} ({len(invalid_batch)} rows)")


if __name__ == "__main__":
    generate_demo_files()
