"""Shared pytest fixtures for test suite."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def config():
    """Load test configuration from YAML."""
    return Config.from_yaml("config/model_config.yaml")


@pytest.fixture
def sample_df():
    """Create a small sample dataframe for testing."""
    np.random.seed(42)
    data = {
        "age": np.random.randint(18, 70, 100),
        "income": np.random.uniform(20000, 100000, 100),
        "loan_amount": np.random.uniform(1000, 50000, 100),
        "loan_term": np.random.randint(12, 360, 100),
        "credit_history_len": np.random.randint(0, 30, 100),
        "num_credit_lines": np.random.randint(1, 15, 100),
        "delinquencies": np.random.randint(0, 10, 100),
        "utilization": np.random.uniform(0, 1, 100),
        "employment_years": np.random.randint(0, 40, 100),
        "num_open_accounts": np.random.randint(0, 10, 100),
        "avg_monthly_balance": np.random.uniform(0, 10000, 100),
        "feature_extra": np.random.uniform(0, 1, 100),
        "default": np.random.randint(0, 2, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def bank_locations():
    """Create sample bank locations."""
    np.random.seed(42)
    return np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])


@pytest.fixture
def train_test_data(config):
    """Create train/test split data with informative features."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create informative dataset using sklearn
    X, y = make_classification(
        n_samples=500,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=42,
    )

    # Convert to DataFrame with proper column names
    X_df = pd.DataFrame(X, columns=config.features.original_cols)
    y_series = pd.Series(y, name=config.features.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )

    return X_train, X_test, y_train, y_test
