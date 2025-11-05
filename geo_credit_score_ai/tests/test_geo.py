"""Tests for geospatial features."""

import numpy as np
import pandas as pd

from main import (
    add_geo_data,
    build_monotone_constraints,
    create_bank_locations,
    inject_distance_label_signal,
)


class TestCreateBankLocations:
    """Tests for create_bank_locations function."""

    def test_returns_numpy_array(self, config):
        """Test that function returns numpy array."""
        banks = create_bank_locations(config)
        assert isinstance(banks, np.ndarray)

    def test_correct_shape(self, config):
        """Test that array has correct shape."""
        banks = create_bank_locations(config)
        assert banks.shape == (config.geo.n_banks, 2)

    def test_within_bounds(self, config):
        """Test that bank locations are within configured bounds."""
        banks = create_bank_locations(config)
        x_low, x_high = config.geo.bounds["x"]
        y_low, y_high = config.geo.bounds["y"]

        assert (banks[:, 0] >= x_low).all()
        assert (banks[:, 0] <= x_high).all()
        assert (banks[:, 1] >= y_low).all()
        assert (banks[:, 1] <= y_high).all()

    def test_reproducibility(self, config):
        """Test that same config produces same bank locations."""
        banks1 = create_bank_locations(config)
        banks2 = create_bank_locations(config)
        np.testing.assert_array_equal(banks1, banks2)


class TestAddGeoData:
    """Tests for add_geo_data function."""

    def test_adds_client_coordinates(self, sample_df, config, bank_locations):
        """Test that client coordinates are added."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert "client_x" in result.columns
        assert "client_y" in result.columns

    def test_adds_distance_features(self, sample_df, config, bank_locations):
        """Test that distance features are added."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert "min_bank_distance" in result.columns
        assert "mean_bank_distance" in result.columns
        assert "std_bank_distance" in result.columns
        assert "distance_balance_risk" in result.columns

    def test_client_coords_within_bounds(self, sample_df, config, bank_locations):
        """Test that client coordinates are within bounds."""
        result = add_geo_data(sample_df, config, bank_locations)
        x_low, x_high = config.geo.bounds["x"]
        y_low, y_high = config.geo.bounds["y"]

        assert (result["client_x"] >= x_low).all()
        assert (result["client_x"] <= x_high).all()
        assert (result["client_y"] >= y_low).all()
        assert (result["client_y"] <= y_high).all()

    def test_min_distance_non_negative(self, sample_df, config, bank_locations):
        """Test that minimum distance is non-negative."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert (result["min_bank_distance"] >= 0).all()

    def test_mean_distance_non_negative(self, sample_df, config, bank_locations):
        """Test that mean distance is non-negative."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert (result["mean_bank_distance"] >= 0).all()

    def test_std_distance_non_negative(self, sample_df, config, bank_locations):
        """Test that std distance is non-negative."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert (result["std_bank_distance"] >= 0).all()

    def test_min_less_than_mean(self, sample_df, config, bank_locations):
        """Test that minimum distance is less than or equal to mean distance."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert (result["min_bank_distance"] <= result["mean_bank_distance"]).all()

    def test_distance_balance_risk_non_negative(self, sample_df, config, bank_locations):
        """Test that distance_balance_risk is non-negative."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert (result["distance_balance_risk"] >= 0).all()

    def test_same_length_as_input(self, sample_df, config, bank_locations):
        """Test that output has same length as input."""
        result = add_geo_data(sample_df, config, bank_locations)
        assert len(result) == len(sample_df)

    def test_reproducibility(self, sample_df, config, bank_locations):
        """Test that same inputs produce same output."""
        result1 = add_geo_data(sample_df.copy(), config, bank_locations)
        result2 = add_geo_data(sample_df.copy(), config, bank_locations)
        pd.testing.assert_frame_equal(result1, result2)


class TestInjectDistanceLabelSignal:
    """Tests for inject_distance_label_signal function."""

    def test_disabled_injection_no_changes(self, sample_df, config):
        """Test that when injection is disabled, labels don't change."""
        original_target = sample_df[config.features.target_col].copy()
        config.geo.inject_label_signal = False
        inject_distance_label_signal(sample_df, config)
        pd.testing.assert_series_equal(sample_df[config.features.target_col], original_target)

    def test_enabled_injection_modifies_labels(self, config):
        """Test that when enabled, some labels are modified."""
        # Create data with distance feature
        np.random.seed(42)
        df = pd.DataFrame({
            "min_bank_distance": np.random.uniform(0, 10, 1000),
            config.features.target_col: np.zeros(1000),  # All zeros initially
        })

        config.geo.inject_label_signal = True
        inject_distance_label_signal(df, config)

        # Some labels should have been flipped to 1
        assert (df[config.features.target_col] == 1).sum() > 0

    def test_only_affects_far_distances(self, config):
        """Test that only far distance samples are affected."""
        np.random.seed(42)
        df = pd.DataFrame({
            "min_bank_distance": np.random.uniform(0, 10, 1000),
            config.features.target_col: np.zeros(1000),
        })

        config.geo.inject_label_signal = True
        qthr = df["min_bank_distance"].quantile(config.geo.signal_quantile)

        inject_distance_label_signal(df, config)

        # Check that only samples with distance >= threshold were potentially flipped
        near_samples = df[df["min_bank_distance"] < qthr]
        # Near samples should still be 0 (none flipped)
        assert (near_samples[config.features.target_col] == 0).all()

    def test_flips_within_signal_strength(self, config):
        """Test that proportion of flips is roughly as configured."""
        np.random.seed(42)
        df = pd.DataFrame({
            "min_bank_distance": np.random.uniform(0, 10, 10000),  # Large sample for statistical test
            config.features.target_col: np.zeros(10000),
        })

        config.geo.inject_label_signal = True
        qthr = df["min_bank_distance"].quantile(config.geo.signal_quantile)
        far_samples_count = (df["min_bank_distance"] >= qthr).sum()

        inject_distance_label_signal(df, config)

        flipped_count = (df[config.features.target_col] == 1).sum()
        expected_flips = far_samples_count * config.geo.label_signal_strength

        # Allow 20% tolerance due to randomness
        assert abs(flipped_count - expected_flips) / expected_flips < 0.2


class TestBuildMonotoneConstraints:
    """Tests for build_monotone_constraints function."""

    def test_disabled_returns_empty_list(self, config):
        """Test that disabled constraints returns empty list."""
        config.geo.enforce_monotonic = False
        feature_cols = ["feature1", "feature2"]
        result = build_monotone_constraints(feature_cols, config)
        assert result == []

    def test_enabled_returns_constraints(self, config):
        """Test that enabled constraints returns list of constraints."""
        config.geo.enforce_monotonic = True
        feature_cols = ["min_bank_distance", "feature2", "mean_bank_distance"]
        result = build_monotone_constraints(feature_cols, config)
        assert isinstance(result, list)
        assert len(result) == len(feature_cols)

    def test_correct_constraint_values(self, config):
        """Test that distance features get constraint value 1."""
        config.geo.enforce_monotonic = True
        feature_cols = ["min_bank_distance", "other_feature", "mean_bank_distance"]
        result = build_monotone_constraints(feature_cols, config)

        assert result[0] == 1  # min_bank_distance should be constrained
        assert result[1] == 0  # other_feature should not be constrained
        assert result[2] == 1  # mean_bank_distance should be constrained

    def test_all_distance_features_constrained(self, config):
        """Test that all distance features are constrained."""
        config.geo.enforce_monotonic = True
        feature_cols = [
            "min_bank_distance",
            "mean_bank_distance",
            "std_bank_distance",
            "distance_balance_risk",
            "other_feature",
        ]
        result = build_monotone_constraints(feature_cols, config)

        assert result[0] == 1  # min_bank_distance
        assert result[1] == 1  # mean_bank_distance
        assert result[2] == 1  # std_bank_distance
        assert result[3] == 1  # distance_balance_risk
        assert result[4] == 0  # other_feature

    def test_non_distance_features_not_constrained(self, config):
        """Test that non-distance features are not constrained."""
        config.geo.enforce_monotonic = True
        feature_cols = ["income", "age", "loan_amount"]
        result = build_monotone_constraints(feature_cols, config)

        assert all(c == 0 for c in result)

    def test_constraint_length_matches_features(self, config):
        """Test that constraint list length matches feature list."""
        config.geo.enforce_monotonic = True
        for n in [5, 10, 20]:
            feature_cols = [f"feature_{i}" for i in range(n)]
            result = build_monotone_constraints(feature_cols, config)
            assert len(result) == n
