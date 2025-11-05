"""Tests for data validation module."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.validation import (
    DataValidationError,
    validate_dataset,
    validate_dataset_size,
    validate_feature_ranges,
    validate_feature_types,
    validate_no_infinite_values,
    validate_schema,
    validate_target_column,
    validate_train_test_split,
)


@pytest.fixture
def config():
    """Load test configuration."""
    return Config.from_yaml("config/model_config.yaml")


@pytest.fixture
def valid_df(config):
    """Create valid DataFrame for testing."""
    n_samples = 200
    data = {col: np.random.randn(n_samples) for col in config.features.original_cols}
    data[config.features.target_col] = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(data)


class TestValidateSchema:
    """Test schema validation."""

    def test_valid_schema_strict(self, valid_df, config):
        """Test validation passes with correct schema in strict mode."""
        expected_cols = config.features.original_cols + [config.features.target_col]
        validate_schema(valid_df, expected_cols, strict=True)  # Should not raise

    def test_valid_schema_non_strict(self, valid_df, config):
        """Test validation passes with subset of columns in non-strict mode."""
        expected_cols = config.features.original_cols[:5]
        validate_schema(valid_df, expected_cols, strict=False)  # Should not raise

    def test_missing_columns(self, valid_df, config):
        """Test validation fails when required columns are missing."""
        df_missing = valid_df.drop(columns=["age"])
        expected_cols = config.features.original_cols + [config.features.target_col]

        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_schema(df_missing, expected_cols, strict=True)

    def test_extra_columns_strict(self, valid_df, config):
        """Test validation fails with extra columns in strict mode."""
        valid_df["extra_column"] = 1
        expected_cols = config.features.original_cols + [config.features.target_col]

        with pytest.raises(DataValidationError, match="Unexpected columns found"):
            validate_schema(valid_df, expected_cols, strict=True)

    def test_extra_columns_non_strict(self, valid_df, config):
        """Test validation passes with extra columns in non-strict mode."""
        valid_df["extra_column"] = 1
        expected_cols = config.features.original_cols + [config.features.target_col]
        validate_schema(valid_df, expected_cols, strict=False)  # Should not raise


class TestValidateTargetColumn:
    """Test target column validation."""

    def test_valid_binary_target(self, valid_df, config):
        """Test validation passes with valid binary target."""
        validate_target_column(valid_df, config.features.target_col)

    def test_missing_target_column(self, valid_df):
        """Test validation fails when target column is missing."""
        with pytest.raises(DataValidationError, match="Target column .* not found"):
            validate_target_column(valid_df, "non_existent_target")

    def test_target_with_nan(self, valid_df, config):
        """Test validation fails when target contains NaN."""
        valid_df.loc[0, config.features.target_col] = np.nan

        with pytest.raises(DataValidationError, match="Target column contains .* NaN values"):
            validate_target_column(valid_df, config.features.target_col)

    def test_unexpected_target_values(self, valid_df, config):
        """Test validation fails with unexpected target values."""
        valid_df.loc[0, config.features.target_col] = 2  # Invalid value

        with pytest.raises(DataValidationError, match="Target column contains unexpected values"):
            validate_target_column(valid_df, config.features.target_col)

    def test_insufficient_class_samples(self, config):
        """Test validation fails when class has too few samples."""
        df = pd.DataFrame({
            "age": [1, 2, 3, 4, 5],
            config.features.target_col: [0, 0, 0, 0, 1],  # Only 1 sample of class 1
        })

        with pytest.raises(DataValidationError, match="Class .* has only .* samples"):
            validate_target_column(df, config.features.target_col, min_class_samples=10)

    def test_custom_expected_values(self, config):
        """Test validation with custom expected values."""
        df = pd.DataFrame({
            "age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "target": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
        })

        validate_target_column(df, "target", expected_values={0, 1, 2}, min_class_samples=3)


class TestValidateFeatureRanges:
    """Test feature range validation."""

    def test_valid_ranges(self, valid_df):
        """Test validation passes when values are within range."""
        validate_feature_ranges(valid_df, "age", min_val=-5, max_val=5, allow_nan=True)

    def test_values_below_minimum(self, valid_df):
        """Test validation fails when values are below minimum."""
        valid_df.loc[0, "age"] = -100

        with pytest.raises(DataValidationError, match="values below minimum"):
            validate_feature_ranges(valid_df, "age", min_val=-5)

    def test_values_above_maximum(self, valid_df):
        """Test validation fails when values are above maximum."""
        valid_df.loc[0, "age"] = 100

        with pytest.raises(DataValidationError, match="values above maximum"):
            validate_feature_ranges(valid_df, "age", max_val=5)

    def test_missing_feature_column(self, valid_df):
        """Test validation fails when feature column is missing."""
        with pytest.raises(DataValidationError, match="Feature column .* not found"):
            validate_feature_ranges(valid_df, "non_existent_feature")

    def test_nan_not_allowed(self, valid_df):
        """Test validation fails when NaN present but not allowed."""
        valid_df.loc[0, "age"] = np.nan

        with pytest.raises(DataValidationError, match="contains NaN values \\(not allowed\\)"):
            validate_feature_ranges(valid_df, "age", allow_nan=False)

    def test_excessive_nan_ratio(self, valid_df):
        """Test validation fails when NaN ratio exceeds threshold."""
        valid_df.loc[:150, "age"] = np.nan  # 75% NaN

        with pytest.raises(DataValidationError, match="NaN values.*maximum allowed"):
            validate_feature_ranges(valid_df, "age", allow_nan=True, max_nan_ratio=0.5)

    def test_all_nan_values(self, valid_df):
        """Test validation with all NaN values."""
        valid_df["age"] = np.nan
        validate_feature_ranges(valid_df, "age", allow_nan=True, max_nan_ratio=1.0)


class TestValidateInfiniteValues:
    """Test infinite value validation."""

    def test_no_infinite_values(self, valid_df, config):
        """Test validation passes when no infinite values present."""
        validate_no_infinite_values(valid_df, exclude_cols=[config.features.target_col])

    def test_infinite_values_detected(self, valid_df):
        """Test validation fails when infinite values are present."""
        valid_df.loc[0, "age"] = np.inf

        with pytest.raises(DataValidationError, match="contains .* infinite values"):
            validate_no_infinite_values(valid_df)

    def test_negative_infinite_values(self, valid_df):
        """Test validation fails with negative infinite values."""
        valid_df.loc[0, "income"] = -np.inf

        with pytest.raises(DataValidationError, match="contains .* infinite values"):
            validate_no_infinite_values(valid_df)

    def test_exclude_columns(self, valid_df, config):
        """Test validation excludes specified columns."""
        valid_df.loc[0, "age"] = np.inf
        validate_no_infinite_values(valid_df, exclude_cols=["age"])

    def test_mixed_infinite_values(self, valid_df):
        """Test validation detects multiple infinite values."""
        valid_df.loc[0, "age"] = np.inf
        valid_df.loc[1, "age"] = -np.inf

        with pytest.raises(DataValidationError, match="contains .* infinite values"):
            validate_no_infinite_values(valid_df)


class TestValidateDatasetSize:
    """Test dataset size validation."""

    def test_valid_size(self, valid_df):
        """Test validation passes with valid dataset size."""
        validate_dataset_size(valid_df, min_rows=100, max_rows=500)

    def test_dataset_too_small(self):
        """Test validation fails when dataset is too small."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(DataValidationError, match="Dataset too small"):
            validate_dataset_size(df, min_rows=100)

    def test_dataset_too_large(self, valid_df):
        """Test validation fails when dataset is too large."""
        with pytest.raises(DataValidationError, match="Dataset too large"):
            validate_dataset_size(valid_df, max_rows=50)

    def test_edge_cases(self):
        """Test validation at boundary values."""
        df = pd.DataFrame({"a": range(100)})
        validate_dataset_size(df, min_rows=100, max_rows=100)  # Exactly at limits


class TestValidateFeatureTypes:
    """Test feature type validation."""

    def test_all_numeric(self, valid_df, config):
        """Test validation passes when all features are numeric."""
        validate_feature_types(valid_df, config.features.original_cols)

    def test_non_numeric_feature(self, valid_df, config):
        """Test validation fails when feature is not numeric."""
        valid_df["age"] = valid_df["age"].astype(str)

        with pytest.raises(DataValidationError, match="is not numeric"):
            validate_feature_types(valid_df, config.features.original_cols)

    def test_missing_columns_ignored(self, valid_df):
        """Test validation ignores missing columns."""
        validate_feature_types(valid_df, ["age", "non_existent_column"])

    def test_mixed_types(self, valid_df, config):
        """Test validation with mixed numeric types."""
        valid_df["age"] = valid_df["age"].astype(int)
        valid_df["income"] = valid_df["income"].astype(float)
        validate_feature_types(valid_df, ["age", "income"])


class TestValidateDataset:
    """Test comprehensive dataset validation."""

    def test_valid_raw_dataset(self, valid_df, config):
        """Test validation passes for valid raw dataset."""
        results = validate_dataset(valid_df, config, stage="raw")

        assert results["validation_passed"] is True
        assert results["stage"] == "raw"
        assert results["n_rows"] == len(valid_df)
        assert "missing_ratio" in results
        assert "target_distribution" in results

    def test_dataset_too_small(self, config):
        """Test validation fails when dataset is too small."""
        df = pd.DataFrame({col: [1, 2] for col in config.features.original_cols})
        df[config.features.target_col] = [0, 1]

        with pytest.raises(DataValidationError, match="Dataset too small"):
            validate_dataset(df, config, stage="raw")

    def test_missing_target_column(self, valid_df, config):
        """Test validation fails when target column is missing."""
        df_no_target = valid_df.drop(columns=[config.features.target_col])

        with pytest.raises(DataValidationError, match="Target column .* not found"):
            validate_dataset(df_no_target, config, stage="raw")

    def test_infinite_values_in_features(self, valid_df, config):
        """Test validation fails with infinite values in features."""
        valid_df.loc[0, "age"] = np.inf

        with pytest.raises(DataValidationError, match="infinite values"):
            validate_dataset(valid_df, config, stage="raw")

    def test_validation_with_high_missing_ratio(self, valid_df, config):
        """Test validation passes with high but acceptable missing ratio."""
        # Set 80% of values to NaN (below 90% threshold)
        for col in config.features.original_cols[:3]:
            valid_df.loc[:159, col] = np.nan

        results = validate_dataset(valid_df, config, stage="raw")
        assert results["validation_passed"] is True

    def test_validation_with_excessive_missing(self, valid_df, config):
        """Test validation fails when feature has too many missing values."""
        # Set 95% of values to NaN (above 90% threshold)
        valid_df.loc[:189, "age"] = np.nan

        with pytest.raises(DataValidationError, match="NaN values"):
            validate_dataset(valid_df, config, stage="raw")


class TestValidateTrainTestSplit:
    """Test train/test split validation."""

    def test_valid_split(self, config):
        """Test validation passes with valid train/test split."""
        n_samples = 200
        X_train = pd.DataFrame({
            col: np.random.randn(140) for col in config.features.original_cols
        })
        X_test = pd.DataFrame({
            col: np.random.randn(60) for col in config.features.original_cols
        })
        y_train = pd.Series(np.random.randint(0, 2, 140))
        y_test = pd.Series(np.random.randint(0, 2, 60))

        results = validate_train_test_split(X_train, X_test, y_train, y_test)

        assert results["total_samples"] == 200
        assert results["train_samples"] == 140
        assert results["test_samples"] == 60
        assert 0.1 <= results["test_ratio"] <= 0.5

    def test_test_set_too_small(self, config):
        """Test validation fails when test set is too small."""
        X_train = pd.DataFrame({col: np.random.randn(190) for col in config.features.original_cols})
        X_test = pd.DataFrame({col: np.random.randn(10) for col in config.features.original_cols})
        y_train = pd.Series(np.random.randint(0, 2, 190))
        y_test = pd.Series(np.random.randint(0, 2, 10))

        with pytest.raises(DataValidationError, match="Test set too small"):
            validate_train_test_split(X_train, X_test, y_train, y_test)

    def test_test_set_too_large(self, config):
        """Test validation fails when test set is too large."""
        X_train = pd.DataFrame({col: np.random.randn(40) for col in config.features.original_cols})
        X_test = pd.DataFrame({col: np.random.randn(160) for col in config.features.original_cols})
        y_train = pd.Series(np.random.randint(0, 2, 40))
        y_test = pd.Series(np.random.randint(0, 2, 160))

        with pytest.raises(DataValidationError, match="Test set too large"):
            validate_train_test_split(X_train, X_test, y_train, y_test)

    def test_mismatched_columns(self, config):
        """Test validation fails when train/test columns don't match."""
        X_train = pd.DataFrame({col: np.random.randn(140) for col in config.features.original_cols})
        X_test = pd.DataFrame({col: np.random.randn(60) for col in config.features.original_cols[:5]})
        y_train = pd.Series(np.random.randint(0, 2, 140))
        y_test = pd.Series(np.random.randint(0, 2, 60))

        with pytest.raises(DataValidationError, match="feature columns do not match"):
            validate_train_test_split(X_train, X_test, y_train, y_test)

    def test_custom_ratio_limits(self, config):
        """Test validation with custom min/max test ratios."""
        X_train = pd.DataFrame({col: np.random.randn(70) for col in config.features.original_cols})
        X_test = pd.DataFrame({col: np.random.randn(30) for col in config.features.original_cols})
        y_train = pd.Series(np.random.randint(0, 2, 70))
        y_test = pd.Series(np.random.randint(0, 2, 30))

        results = validate_train_test_split(
            X_train, X_test, y_train, y_test,
            min_test_ratio=0.2, max_test_ratio=0.4
        )
        assert results["test_ratio"] >= 0.2
        assert results["test_ratio"] <= 0.4

    def test_class_distribution_included(self, config):
        """Test that results include class distributions."""
        X_train = pd.DataFrame({col: np.random.randn(140) for col in config.features.original_cols})
        X_test = pd.DataFrame({col: np.random.randn(60) for col in config.features.original_cols})
        y_train = pd.Series([0] * 100 + [1] * 40)
        y_test = pd.Series([0] * 40 + [1] * 20)

        results = validate_train_test_split(X_train, X_test, y_train, y_test)

        assert "train_class_distribution" in results
        assert "test_class_distribution" in results
        assert results["train_class_distribution"][0] == 100
        assert results["train_class_distribution"][1] == 40
        assert results["test_class_distribution"][0] == 40
        assert results["test_class_distribution"][1] == 20
