"""
Configuration management using Pydantic models.
This module defines type-safe configuration models that can be loaded from YAML files
and validated at runtime.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetConfig(BaseModel):
    n_samples: int = Field(gt=0, le=100_000)
    n_features: int = Field(gt=0, le=100)
    n_informative: int = Field(gt=0)
    n_redundant: int = Field(ge=0)
    n_clusters_per_class: int = Field(gt=0, le=10)
    weights: list[float] = Field(description="Class weights [negative_class, positive_class]")
    flip_y: float = Field(ge=0.0, le=0.5, description="Proportion of labels to flip (noise)")

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("weights must have exactly 2 elements")
        if not all(0 <= w <= 1 for w in v):
            raise ValueError("All weights must be between 0 and 1")
        if not abs(sum(v) - 1.0) < 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(v)}")
        return v

    @model_validator(mode="after")
    def validate_feature_counts(self) -> "DatasetConfig":
        if self.n_informative + self.n_redundant > self.n_features:
            raise ValueError(
                f"n_informative ({self.n_informative}) + n_redundant ({self.n_redundant}) "
                f"cannot exceed n_features ({self.n_features})"
            )
        return self


class MissingValuesConfig(BaseModel):
    p_missing: float = Field(ge=0.0, le=0.5, description="Probability of missing value per cell")


class FeaturesConfig(BaseModel):
    original_cols: list[str] = Field(min_length=1, description="Original feature column names")
    target_col: str = Field(description="Target variable column name")
    missing_indicator_cols: list[str] = Field(
        default_factory=list, description="Columns to create missing indicators for"
    )


class ModelConfig(BaseModel):
    test_size: float = Field(gt=0.0, lt=1.0)
    cv_folds: int = Field(gt=1, le=10)
    lgbm_params: dict[str, Any] = Field(description="LightGBM model parameters")
    use_smote: bool = Field(default=True)
    fp_cost: float = Field(default=100.0, ge=0)
    fn_cost: float = Field(default=5000.0, ge=0)

    @field_validator("lgbm_params")
    @classmethod
    def validate_lgbm_params(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "n_estimators" in v and v["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive")
        if "max_depth" in v and v["max_depth"] <= 0:
            raise ValueError("max_depth must be positive")
        if "learning_rate" in v and not 0 < v["learning_rate"] <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        return v

    @field_validator("fp_cost", "fn_cost")
    @classmethod
    def validate_costs(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Costs must be non-negative")
        return v


class GeoConfig(BaseModel):
    n_banks: int = Field(gt=0, le=1000)
    bounds: dict[str, tuple[float, float]] = Field(description="Coordinate bounds for spatial area")
    distance_features: list[str] = Field(
        default_factory=lambda: ["min_bank_distance", "mean_bank_distance", "std_bank_distance"]
    )
    enforce_monotonic: bool = Field(default=True)
    inject_label_signal: bool = Field(default=True)
    signal_quantile: float = Field(ge=0.0, le=1.0, default=0.80)
    label_signal_strength: float = Field(ge=0.0, le=1.0, default=0.25)

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        if "x" not in v or "y" not in v:
            raise ValueError("bounds must contain 'x' and 'y'")
        for axis, (low, high) in v.items():
            if low >= high:
                raise ValueError(f"bounds[{axis}]: lower bound must be < upper bound")
        return v


class VisualizationConfig(BaseModel):
    top_n_features: int = Field(gt=0, le=50)


class CalibrationConfig(BaseModel):
    enabled: bool = Field(default=True)
    method: Literal["sigmoid", "isotonic"] = Field(default="isotonic")
    cv: int = Field(default=5, ge=2, le=10)


class Config(BaseModel):
    dataset: DatasetConfig
    missing_values: MissingValuesConfig
    features: FeaturesConfig
    model: ModelConfig
    geo: GeoConfig
    visualization: VisualizationConfig
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    random_state: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "Config":
        missing_cols = set(self.features.missing_indicator_cols)
        original_cols = set(self.features.original_cols)
        invalid_cols = missing_cols - original_cols
        if invalid_cols:
            raise ValueError(f"missing_indicator_cols contains columns not in original_cols: {invalid_cols}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
