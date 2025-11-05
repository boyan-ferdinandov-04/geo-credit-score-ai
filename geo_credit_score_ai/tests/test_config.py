"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from src.config import (
    CalibrationConfig,
    Config,
    DatasetConfig,
    FeaturesConfig,
    GeoConfig,
    MissingValuesConfig,
    ModelConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_valid_config(self):
        """Test creating valid dataset config."""
        config = DatasetConfig(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.9, 0.1],
            flip_y=0.01,
        )
        assert config.n_samples == 1000
        assert config.n_features == 10
        assert len(config.weights) == 2

    def test_invalid_n_samples(self):
        """Test that n_samples must be positive."""
        with pytest.raises(ValidationError):
            DatasetConfig(n_samples=-100)

    def test_invalid_weights_sum(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValidationError):
            DatasetConfig(weights=[0.5, 0.3])

    def test_invalid_weights_length(self):
        """Test that weights must have exactly 2 elements."""
        with pytest.raises(ValidationError):
            DatasetConfig(weights=[0.33, 0.33, 0.34])

    def test_invalid_flip_y(self):
        """Test that flip_y must be between 0 and 1."""
        with pytest.raises(ValidationError):
            DatasetConfig(flip_y=1.5)


class TestMissingValuesConfig:
    """Tests for MissingValuesConfig."""

    def test_valid_p_missing(self):
        """Test valid p_missing value."""
        config = MissingValuesConfig(p_missing=0.1)
        assert config.p_missing == 0.1

    def test_invalid_p_missing_negative(self):
        """Test that p_missing cannot be negative."""
        with pytest.raises(ValidationError):
            MissingValuesConfig(p_missing=-0.1)

    def test_invalid_p_missing_too_high(self):
        """Test that p_missing cannot exceed 1.0."""
        with pytest.raises(ValidationError):
            MissingValuesConfig(p_missing=1.5)


class TestFeaturesConfig:
    """Tests for FeaturesConfig."""

    def test_default_columns(self, config):
        """Test that feature config has expected columns."""
        assert len(config.features.original_cols) == 12
        assert "income" in config.features.original_cols
        assert config.features.target_col == "default"
        assert "income" in config.features.missing_indicator_cols


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_valid_model_config(self):
        """Test creating valid model config."""
        config = ModelConfig(
            test_size=0.2,
            cv_folds=5,
            use_smote=False,
            fp_cost=100,
            fn_cost=5000,
            lgbm_params={"max_depth": 5, "learning_rate": 0.05},
        )
        assert config.test_size == 0.2
        assert config.cv_folds == 5
        assert config.fp_cost == 100

    def test_invalid_test_size(self):
        """Test that test_size must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ModelConfig(test_size=1.5)

    def test_invalid_cv_folds(self):
        """Test that cv_folds must be at least 2."""
        with pytest.raises(ValidationError):
            ModelConfig(cv_folds=1)

    def test_invalid_costs(self):
        """Test that costs must be positive."""
        with pytest.raises(ValidationError):
            ModelConfig(fp_cost=-100)


class TestGeoConfig:
    """Tests for GeoConfig."""

    def test_valid_geo_config(self):
        """Test creating valid geo config."""
        config = GeoConfig(
            n_banks=10,
            bounds={"x": [0, 10], "y": [0, 10]},
            inject_label_signal=True,
            signal_quantile=0.75,
            label_signal_strength=0.25,
        )
        assert config.n_banks == 10
        assert config.signal_quantile == 0.75

    def test_invalid_n_banks(self):
        """Test that n_banks must be positive."""
        with pytest.raises(ValidationError):
            GeoConfig(n_banks=0)

    def test_invalid_bounds(self):
        """Test that bounds must have valid structure."""
        with pytest.raises(ValidationError):
            GeoConfig(bounds={"x": [10, 0], "y": [0, 10]})  # x_low > x_high

    def test_invalid_quantile(self):
        """Test that signal_quantile must be between 0 and 1."""
        with pytest.raises(ValidationError):
            GeoConfig(signal_quantile=1.5)


class TestCalibrationConfig:
    """Tests for CalibrationConfig."""

    def test_valid_calibration_config(self):
        """Test creating valid calibration config."""
        config = CalibrationConfig(enabled=True, method="isotonic", cv=3)
        assert config.enabled is True
        assert config.method == "isotonic"
        assert config.cv == 3

    def test_invalid_method(self):
        """Test that method must be valid."""
        with pytest.raises(ValidationError):
            CalibrationConfig(method="invalid_method")

    def test_invalid_cv(self):
        """Test that cv must be at least 2."""
        with pytest.raises(ValidationError):
            CalibrationConfig(cv=1)


class TestConfig:
    """Tests for main Config class."""

    def test_load_from_yaml(self, config):
        """Test loading config from YAML file."""
        assert config.random_state == 42
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.geo, GeoConfig)

    def test_config_has_all_sections(self, config):
        """Test that config has all required sections."""
        assert hasattr(config, "dataset")
        assert hasattr(config, "missing_values")
        assert hasattr(config, "features")
        assert hasattr(config, "model")
        assert hasattr(config, "geo")
        assert hasattr(config, "calibration")

    def test_config_values_from_yaml(self, config):
        """Test that specific values are loaded correctly from YAML."""
        assert config.dataset.n_samples == 5000
        assert config.model.test_size == 0.2
        assert config.geo.n_banks == 15
        assert config.calibration.method == "isotonic"
