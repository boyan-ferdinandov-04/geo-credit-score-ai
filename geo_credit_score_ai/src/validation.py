"""Data validation module for ensuring dataset quality and integrity.

This module provides comprehensive validation functions to check:
- Schema compliance (expected columns and types)
- Value ranges and distributions
- Missing value patterns
- Target variable properties
- Feature statistics

Usage:
    from src.validation import validate_dataset, DataValidationError

    try:
        validate_dataset(df, config)
        print("Dataset validation passed!")
    except DataValidationError as e:
        print(f"Validation failed: {e}")
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


def validate_schema(df: pd.DataFrame, expected_cols: list[str], strict: bool = True) -> None:
    """Validate that DataFrame has expected columns.

    Args:
        df: DataFrame to validate
        expected_cols: List of expected column names
        strict: If True, DataFrame must have exactly these columns.
                If False, only checks that expected columns exist.

    Raises:
        DataValidationError: If schema validation fails
    """
    df_cols = set(df.columns)
    expected_cols_set = set(expected_cols)

    # Check for missing columns
    missing_cols = expected_cols_set - df_cols
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {sorted(missing_cols)}")

    # Check for unexpected columns (if strict mode)
    if strict:
        extra_cols = df_cols - expected_cols_set
        if extra_cols:
            raise DataValidationError(f"Unexpected columns found: {sorted(extra_cols)}")

    logger.debug(f"Schema validation passed: {len(expected_cols)} columns verified")


def validate_target_column(
    df: pd.DataFrame,
    target_col: str,
    expected_values: set = {0, 1},
    min_class_samples: int = 10
) -> None:
    """Validate target column properties.

    Args:
        df: DataFrame containing target column
        target_col: Name of target column
        expected_values: Set of expected values (default: {0, 1} for binary)
        min_class_samples: Minimum samples required per class

    Raises:
        DataValidationError: If target validation fails
    """
    if target_col not in df.columns:
        raise DataValidationError(f"Target column '{target_col}' not found in DataFrame")

    # Check for NaN values
    nan_count = df[target_col].isna().sum()
    if nan_count > 0:
        raise DataValidationError(f"Target column contains {nan_count} NaN values")

    # Check unique values
    unique_values = set(df[target_col].unique())
    if not unique_values.issubset(expected_values):
        raise DataValidationError(
            f"Target column contains unexpected values. "
            f"Expected: {expected_values}, Found: {unique_values}"
        )

    # Check class balance
    value_counts = df[target_col].value_counts()
    for class_val, count in value_counts.items():
        if count < min_class_samples:
            raise DataValidationError(
                f"Class {class_val} has only {count} samples (minimum: {min_class_samples})"
            )

    logger.debug(f"Target validation passed: {value_counts.to_dict()}")


def validate_feature_ranges(
    df: pd.DataFrame,
    feature_col: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_nan: bool = True,
    max_nan_ratio: float = 0.5
) -> None:
    """Validate that feature values are within expected ranges.

    Args:
        df: DataFrame containing feature
        feature_col: Name of feature column
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        allow_nan: Whether NaN values are permitted
        max_nan_ratio: Maximum ratio of NaN values allowed (0 to 1)

    Raises:
        DataValidationError: If feature validation fails
    """
    if feature_col not in df.columns:
        raise DataValidationError(f"Feature column '{feature_col}' not found in DataFrame")

    # Check NaN values
    nan_ratio = df[feature_col].isna().sum() / len(df)
    if not allow_nan and nan_ratio > 0:
        raise DataValidationError(f"Feature '{feature_col}' contains NaN values (not allowed)")

    if nan_ratio > max_nan_ratio:
        raise DataValidationError(
            f"Feature '{feature_col}' has {nan_ratio:.1%} NaN values "
            f"(maximum allowed: {max_nan_ratio:.1%})"
        )

    # Check value ranges (excluding NaN)
    non_nan_values = df[feature_col].dropna()
    if len(non_nan_values) > 0:
        actual_min = non_nan_values.min()
        actual_max = non_nan_values.max()

        if min_val is not None and actual_min < min_val:
            raise DataValidationError(
                f"Feature '{feature_col}' has values below minimum. "
                f"Min allowed: {min_val}, Actual min: {actual_min}"
            )

        if max_val is not None and actual_max > max_val:
            raise DataValidationError(
                f"Feature '{feature_col}' has values above maximum. "
                f"Max allowed: {max_val}, Actual max: {actual_max}"
            )


def validate_no_infinite_values(df: pd.DataFrame, exclude_cols: Optional[list[str]] = None) -> None:
    """Check that DataFrame contains no infinite values.

    Args:
        df: DataFrame to validate
        exclude_cols: Columns to exclude from check

    Raises:
        DataValidationError: If infinite values are found
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    check_cols = [col for col in numeric_cols if col not in exclude_cols]

    for col in check_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            raise DataValidationError(
                f"Feature '{col}' contains {inf_count} infinite values"
            )

    logger.debug(f"Infinite value check passed for {len(check_cols)} numeric columns")


def validate_dataset_size(df: pd.DataFrame, min_rows: int = 100, max_rows: int = 1_000_000) -> None:
    """Validate that dataset size is within acceptable range.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        max_rows: Maximum number of rows allowed

    Raises:
        DataValidationError: If size validation fails
    """
    n_rows = len(df)

    if n_rows < min_rows:
        raise DataValidationError(f"Dataset too small: {n_rows} rows (minimum: {min_rows})")

    if n_rows > max_rows:
        raise DataValidationError(f"Dataset too large: {n_rows} rows (maximum: {max_rows})")

    logger.debug(f"Dataset size validation passed: {n_rows:,} rows")


def validate_feature_types(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    """Validate that specified columns are numeric.

    Args:
        df: DataFrame to validate
        numeric_cols: List of columns that should be numeric

    Raises:
        DataValidationError: If type validation fails
    """
    for col in numeric_cols:
        if col not in df.columns:
            continue  # Column might not exist yet (e.g., engineered features)

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"Feature '{col}' is not numeric (type: {df[col].dtype})"
            )

    logger.debug(f"Feature type validation passed for {len(numeric_cols)} columns")


def validate_dataset(df: pd.DataFrame, cfg: Config, stage: str = "raw") -> dict[str, Any]:
    """Comprehensive dataset validation with configurable checks.

    Validates dataset at different pipeline stages:
    - "raw": After initial dataset creation
    - "engineered": After feature engineering
    - "final": After geospatial features added

    Args:
        df: DataFrame to validate
        cfg: Configuration object
        stage: Pipeline stage ("raw", "engineered", "final")

    Returns:
        Dictionary with validation results and statistics

    Raises:
        DataValidationError: If any validation check fails

    Example:
        >>> df = create_dataset(config)
        >>> results = validate_dataset(df, config, stage="raw")
        >>> print(f"Validation passed! Dataset has {results['n_rows']} rows")
    """
    logger.info(f"Starting dataset validation (stage: {stage})...")

    results = {
        "stage": stage,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "validation_passed": False,
    }

    try:
        # 1. Dataset size validation
        validate_dataset_size(df, min_rows=100, max_rows=cfg.dataset.n_samples * 2)

        # 2. Target column validation
        validate_target_column(
            df,
            cfg.features.target_col,
            expected_values={0, 1},
            min_class_samples=10
        )

        # 3. Schema validation (stage-specific)
        if stage == "raw":
            expected_cols = cfg.features.original_cols + [cfg.features.target_col]
            validate_schema(df, expected_cols, strict=True)

        # 4. Feature type validation
        validate_feature_types(df, cfg.features.original_cols)

        # 5. No infinite values
        validate_no_infinite_values(df, exclude_cols=[cfg.features.target_col])

        # 6. Basic feature range checks for original features
        for col in cfg.features.original_cols:
            if col in df.columns:
                validate_feature_ranges(
                    df, col,
                    allow_nan=True,
                    max_nan_ratio=0.9  # Allow up to 90% missing
                )

        # 7. Calculate statistics
        results["missing_ratio"] = df[cfg.features.original_cols].isna().sum().sum() / (
            len(df) * len(cfg.features.original_cols)
        )
        results["target_distribution"] = df[cfg.features.target_col].value_counts().to_dict()
        results["validation_passed"] = True

        logger.info(
            f"✓ Validation passed: {results['n_rows']:,} rows, "
            f"{results['n_cols']} columns, "
            f"{results['missing_ratio']:.1%} missing values"
        )

        return results

    except DataValidationError as e:
        logger.error(f"✗ Validation failed: {e}")
        results["error"] = str(e)
        raise


def validate_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    min_test_ratio: float = 0.1,
    max_test_ratio: float = 0.5
) -> dict[str, Any]:
    """Validate train/test split proportions and distributions.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        min_test_ratio: Minimum test set size ratio
        max_test_ratio: Maximum test set size ratio

    Returns:
        Dictionary with split statistics

    Raises:
        DataValidationError: If split validation fails
    """
    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples

    # Validate split ratio
    if test_ratio < min_test_ratio:
        raise DataValidationError(
            f"Test set too small: {test_ratio:.1%} (minimum: {min_test_ratio:.1%})"
        )

    if test_ratio > max_test_ratio:
        raise DataValidationError(
            f"Test set too large: {test_ratio:.1%} (maximum: {max_test_ratio:.1%})"
        )

    # Validate feature columns match
    if list(X_train.columns) != list(X_test.columns):
        raise DataValidationError("Train and test feature columns do not match")

    # Validate class distribution in both sets
    train_dist = y_train.value_counts()
    test_dist = y_test.value_counts()

    results = {
        "total_samples": total_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_ratio": test_ratio,
        "train_class_distribution": train_dist.to_dict(),
        "test_class_distribution": test_dist.to_dict(),
    }

    logger.info(
        f"✓ Split validation passed: {len(X_train):,} train, {len(X_test):,} test "
        f"({test_ratio:.1%} test ratio)"
    )

    return results
