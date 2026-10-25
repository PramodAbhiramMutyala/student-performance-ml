"""
Dataset and input validation module.
Verifies column requirements, data types, missing values, duplicates,
category domains, and numerical ranges.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUE_DOMAINS,
    NUMERICAL_FEATURES,
    NUMERICAL_VALUE_RANGES,
    TARGET_COLUMN,
)


@dataclass
class ValidationReport:
    """Holds structured outcome of a dataset validation run."""
    is_valid: bool
    total_rows: int
    total_columns: int
    missing_values_count: int
    duplicate_rows_count: int
    missing_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    category_violations: Dict[str, List[str]] = field(default_factory=dict)
    range_violations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_errors: Dict[int, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values_count": self.missing_values_count,
            "duplicate_rows_count": self.duplicate_rows_count,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "category_violations": self.category_violations,
            "range_violations": self.range_violations,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_errors_count": len(self.row_errors),
        }


def validate_dataset(df: pd.DataFrame, require_target: bool = False) -> ValidationReport:
    """
    Validates a dataset for student performance modeling or inference.
    Checks required features, nulls, duplicates, categories, and numerical bounds.
    """
    errors: List[str] = []
    warnings: List[str] = []
    row_errors: Dict[int, List[str]] = {}

    total_rows, total_cols = df.shape
    if total_rows == 0:
        return ValidationReport(
            is_valid=False,
            total_rows=0,
            total_columns=total_cols,
            missing_values_count=0,
            duplicate_rows_count=0,
            errors=["Uploaded dataset is empty (0 rows)."]
        )

    # 1. Required columns
    expected_cols: Set[str] = set(ALL_FEATURE_COLUMNS)
    if require_target:
        expected_cols.add(TARGET_COLUMN)

    present_cols = set(df.columns)
    missing_columns = sorted(list(expected_cols - present_cols))
    unexpected_columns = sorted(list(present_cols - expected_cols - {"G1", "G2"}))

    if missing_columns:
        errors.append(f"Missing required columns ({len(missing_columns)}): {', '.join(missing_columns)}")

    # 2. Missing values and duplicates
    missing_values_count = int(df.isna().sum().sum())
    duplicate_rows_count = int(df.duplicated().sum())

    if missing_values_count > 0:
        null_breakdown = df.isna().sum()[df.isna().sum() > 0].to_dict()
        warnings.append(f"Dataset contains {missing_values_count} missing values: {null_breakdown}")

    if duplicate_rows_count > 0:
        warnings.append(f"Dataset contains {duplicate_rows_count} duplicate row(s).")

    # 3. Categorical validation
    category_violations: Dict[str, List[str]] = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            allowed = set(CATEGORICAL_VALUE_DOMAINS[col])
            unique_vals = set(df[col].dropna().astype(str).unique())
            invalid = sorted(list(unique_vals - allowed))
            if invalid:
                category_violations[col] = invalid
                errors.append(
                    f"Column '{col}' contains unexpected categories: {invalid}. Allowed: {sorted(list(allowed))}"
                )

    # 4. Numerical range validation
    range_violations: Dict[str, Dict[str, Any]] = {}
    for col, (min_val, max_val) in NUMERICAL_VALUE_RANGES.items():
        if col in df.columns:
            # Check castability
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            unparseable_count = int(numeric_series.isna().sum() - df[col].isna().sum())
            if unparseable_count > 0:
                errors.append(f"Column '{col}' has {unparseable_count} non-numeric values.")

            actual_min = float(numeric_series.min()) if not numeric_series.empty else np.nan
            actual_max = float(numeric_series.max()) if not numeric_series.empty else np.nan

            out_of_bounds = (numeric_series < min_val) | (numeric_series > max_val)
            violation_count = int(out_of_bounds.sum())

            if violation_count > 0:
                range_violations[col] = {
                    "min_allowed": min_val,
                    "max_allowed": max_val,
                    "actual_min": actual_min,
                    "actual_max": actual_max,
                    "violations": violation_count
                }
                errors.append(
                    f"Column '{col}' has {violation_count} values outside expected range [{min_val}, {max_val}]. "
                    f"Observed: [{actual_min}, {actual_max}]"
                )

    # 5. Row-level error indexing for batch troubleshooting
    for idx, row in df.iterrows():
        r_errors: List[str] = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                val = str(row[col]) if pd.notna(row[col]) else None
                if val is not None and val not in CATEGORICAL_VALUE_DOMAINS[col]:
                    r_errors.append(f"'{col}' has invalid value '{val}'")
        for col, (min_val, max_val) in NUMERICAL_VALUE_RANGES.items():
            if col in df.columns:
                try:
                    num_val = float(row[col])
                    if num_val < min_val or num_val > max_val:
                        r_errors.append(f"'{col}' value {num_val} outside [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    r_errors.append(f"'{col}' is not a valid number")

        if r_errors:
            row_errors[int(idx)] = r_errors

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        total_columns=total_cols,
        missing_values_count=missing_values_count,
        duplicate_rows_count=duplicate_rows_count,
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        category_violations=category_violations,
        range_violations=range_violations,
        errors=errors,
        warnings=warnings,
        row_errors=row_errors,
    )


def validate_single_record(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates an individual student dictionary against feature schema."""
    errors: List[str] = []

    for col in ALL_FEATURE_COLUMNS:
        if col not in data:
            errors.append(f"Missing required field: '{col}'")

    for col, allowed_vals in CATEGORICAL_VALUE_DOMAINS.items():
        if col in data:
            val = str(data[col])
            if val not in allowed_vals:
                errors.append(f"Field '{col}' value '{val}' invalid. Must be one of: {allowed_vals}")

    for col, (min_val, max_val) in NUMERICAL_VALUE_RANGES.items():
        if col in data:
            try:
                num_val = float(data[col])
                if num_val < min_val or num_val > max_val:
                    errors.append(f"Field '{col}' value {num_val} outside allowable range [{min_val}, {max_val}]")
            except (ValueError, TypeError):
                errors.append(f"Field '{col}' must be a numeric value")

    return (len(errors) == 0, errors)
