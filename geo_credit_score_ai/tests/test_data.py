"""Tests for dataset creation and feature engineering."""

import numpy as np
import pandas as pd

from src.data import create_dataset, engineer_features


class TestCreateDataset:
    """Tests for create_dataset function."""

    def test_creates_dataframe(self, config):
        """Test that create_dataset returns a DataFrame."""
        df = create_dataset(config)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_samples(self, config):
        """Test that dataset has correct number of samples."""
        df = create_dataset(config)
        assert len(df) == config.dataset.n_samples

    def test_has_target_column(self, config):
        """Test that dataset has target column."""
        df = create_dataset(config)
        assert config.features.target_col in df.columns

    def test_target_values_binary(self, config):
        """Test that target column contains only 0 and 1."""
        df = create_dataset(config)
        target_values = df[config.features.target_col].unique()
        assert set(target_values).issubset({0, 1})

    def test_has_original_columns(self, config):
        """Test that dataset has all original feature columns."""
        df = create_dataset(config)
        for col in config.features.original_cols:
            assert col in df.columns

    def test_class_imbalance(self, config):
        """Test that class distribution matches configuration."""
        df = create_dataset(config)
        value_counts = df[config.features.target_col].value_counts(normalize=True)
        # Check that minority class is roughly 10% (with some tolerance)
        minority_class_ratio = value_counts.min()
        assert 0.08 <= minority_class_ratio <= 0.12

    def test_has_missing_values(self, config):
        """Test that dataset has missing values as configured."""
        df = create_dataset(config)
        missing_count = df[config.features.original_cols].isna().sum().sum()
        assert missing_count > 0  # Should have some missing values

    def test_reproducibility(self, config):
        """Test that same config produces same dataset."""
        df1 = create_dataset(config)
        df2 = create_dataset(config)
        pd.testing.assert_frame_equal(df1, df2)

    def test_missing_percentage_approximately_correct(self, config):
        """Test that missing value percentage is approximately as configured."""
        df = create_dataset(config)
        total_values = df[config.features.original_cols].shape[0] * df[config.features.original_cols].shape[1]
        missing_values = df[config.features.original_cols].isna().sum().sum()
        missing_ratio = missing_values / total_values
        # Allow some tolerance around configured p_missing
        expected_ratio = config.missing_values.p_missing
        assert abs(missing_ratio - expected_ratio) < 0.02


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_returns_tuple(self, sample_df, config):
        """Test that engineer_features returns tuple."""
        result = engineer_features(sample_df, config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_dataframe_and_list(self, sample_df, config):
        """Test that return types are correct."""
        df, feature_cols = engineer_features(sample_df, config)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(feature_cols, list)

    def test_creates_derived_features(self, sample_df, config):
        """Test that derived features are created."""
        df, _ = engineer_features(sample_df, config)
        expected_features = [
            "income_log",
            "delinq_utilization",
            "credit_age_ratio",
            "loan_to_income",
            "balance_to_income",
            "delinq_per_creditline",
            "utilization_squared",
            "credit_utilization_interaction",
            "payment_stress",
            "longterm_loan_flag",
        ]
        for feature in expected_features:
            assert feature in df.columns

    def test_creates_missing_indicators(self, sample_df, config):
        """Test that missing indicators are created."""
        # Add some missing values
        sample_df.loc[0, "income"] = np.nan
        df, _ = engineer_features(sample_df, config)

        for col in config.features.missing_indicator_cols:
            assert f"{col}_missing" in df.columns

    def test_income_log_non_negative(self, sample_df, config):
        """Test that income_log is non-negative."""
        df, _ = engineer_features(sample_df, config)
        assert (df["income_log"] >= 0).all()

    def test_utilization_squared_calculation(self, sample_df, config):
        """Test that utilization_squared is correctly calculated."""
        df, _ = engineer_features(sample_df, config)
        expected = sample_df["utilization"] ** 2
        pd.testing.assert_series_equal(df["utilization_squared"], expected, check_names=False)

    def test_longterm_loan_flag_binary(self, sample_df, config):
        """Test that longterm_loan_flag is binary."""
        df, _ = engineer_features(sample_df, config)
        assert set(df["longterm_loan_flag"].unique()).issubset({0, 1})

    def test_feature_cols_contains_originals(self, sample_df, config):
        """Test that feature_cols includes original columns."""
        _, feature_cols = engineer_features(sample_df, config)
        for col in config.features.original_cols:
            assert col in feature_cols

    def test_feature_cols_contains_derived(self, sample_df, config):
        """Test that feature_cols includes derived features."""
        _, feature_cols = engineer_features(sample_df, config)
        assert "income_log" in feature_cols
        assert "payment_stress" in feature_cols

    def test_no_inf_values(self, sample_df, config):
        """Test that no infinite values are created."""
        df, feature_cols = engineer_features(sample_df, config)
        # Check numerical columns for inf
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        assert not np.isinf(df[numerical_cols]).any().any()

    def test_payment_stress_range(self, sample_df, config):
        """Test that payment_stress values are reasonable."""
        df, _ = engineer_features(sample_df, config)
        # payment_stress should be non-negative
        assert (df["payment_stress"] >= 0).all()

    def test_division_by_zero_handling(self, config):
        """Test that division by zero is handled gracefully."""
        # Create edge case data with zeros - using actual config column names
        data = {
            "age": [0, 30, 50],
            "income": [0, 1000, 0],
            "loan_amount": [1000, 2000, 3000],
            "loan_term": [12, 24, 36],
            "credit_history_len": [0, 5, 10],
            "num_credit_lines": [0, 3, 5],
            "delinquencies": [0, 1, 2],
            "utilization": [0.5, 0.7, 0.9],
            "employment_years": [0, 5, 10],
            "num_open_accounts": [0, 2, 4],
            "avg_monthly_balance": [0, 1000, 2000],
            "feature_extra": [0.3, 0.5, 0.7],
        }
        df = pd.DataFrame(data)
        result_df, _ = engineer_features(df, config)

        # Check no inf values were created
        numerical_cols = result_df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result_df[numerical_cols]).any().any()
