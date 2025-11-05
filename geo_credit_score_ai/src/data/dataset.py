"""Dataset creation and feature engineering functions."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.config import Config
from src.constants import (
    DELINQ_DIVISOR,
    DELINQ_WEIGHT,
    EPSILON,
    LONGTERM_LOAN_QUANTILE,
    UTILIZATION_WEIGHT,
)


def create_dataset(cfg: Config) -> pd.DataFrame:
    """Create synthetic credit default dataset with missing values.

    Args:
        cfg: Configuration object with dataset parameters

    Returns:
        DataFrame with features, target column, and intentional missing values
    """
    X, y = make_classification(
        random_state=cfg.random_state,
        n_samples=cfg.dataset.n_samples,
        n_features=cfg.dataset.n_features,
        n_informative=cfg.dataset.n_informative,
        n_redundant=cfg.dataset.n_redundant,
        n_clusters_per_class=cfg.dataset.n_clusters_per_class,
        weights=cfg.dataset.weights,
        flip_y=cfg.dataset.flip_y,
    )
    df = pd.DataFrame(X, columns=cfg.features.original_cols)
    df[cfg.features.target_col] = y

    # Inject missing values
    rng = np.random.default_rng(cfg.random_state)
    mask = rng.choice(
        [False, True],
        size=df[cfg.features.original_cols].shape,
        p=[1 - cfg.missing_values.p_missing, cfg.missing_values.p_missing],
    )
    df[cfg.features.original_cols] = df[cfg.features.original_cols].mask(mask)
    return df


def engineer_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Engineer derived features and missing value indicators.

    Creates 10 derived features including:
    - Financial ratios (loan_to_income, balance_to_income)
    - Risk indicators (payment_stress, delinq_utilization)
    - Transformations (income_log, utilization_squared)
    - Interaction terms
    - Missing value flags

    Args:
        df: Input DataFrame with raw features
        cfg: Configuration object

    Returns:
        Tuple of (DataFrame with new features, list of all feature column names)
    """
    # Log transformations
    df["income_log"] = np.log1p(np.abs(df["income"]))

    # Interaction features
    df["delinq_utilization"] = df["delinquencies"] * df["utilization"]
    df["credit_utilization_interaction"] = df["credit_history_len"] * df["utilization"]

    # Ratio features
    df["credit_age_ratio"] = df["credit_history_len"] / (df["age"] + EPSILON)
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + EPSILON)
    df["balance_to_income"] = df["avg_monthly_balance"] / (df["income"] + EPSILON)
    df["delinq_per_creditline"] = df["delinquencies"] / (df["num_credit_lines"] + EPSILON)

    # Non-linear transformations
    df["utilization_squared"] = df["utilization"] ** 2

    # Composite risk indicators
    df["payment_stress"] = (
        df["utilization"] * UTILIZATION_WEIGHT +
        (df["delinquencies"] / DELINQ_DIVISOR) * DELINQ_WEIGHT
    )

    # Binary flags
    df["longterm_loan_flag"] = (
        df["loan_term"] > df["loan_term"].quantile(LONGTERM_LOAN_QUANTILE)
    ).astype(int)

    # Missing value indicators
    for col in cfg.features.missing_indicator_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Compile feature list
    base_features = cfg.features.original_cols
    derived_features = [
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
    missing_indicators = [f"{col}_missing" for col in cfg.features.missing_indicator_cols]
    feature_cols = base_features + derived_features + missing_indicators
    return df, feature_cols
