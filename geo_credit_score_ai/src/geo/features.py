"""Geospatial feature generation and processing."""

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import Config
from src.constants import EPSILON, MIN_DISTANCE_INDEX, MONOTONIC_INCREASE, NO_CONSTRAINT

logger = logging.getLogger(__name__)


def create_bank_locations(cfg: Config) -> np.ndarray:
    """Generate random bank locations within configured bounds.

    Args:
        cfg: Configuration object with geo parameters

    Returns:
        Array of shape (n_banks, 2) with (x, y) coordinates
    """
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.geo.n_banks
    x_low, x_high = cfg.geo.bounds["x"]
    y_low, y_high = cfg.geo.bounds["y"]
    bank_x = rng.uniform(x_low, x_high, n)
    bank_y = rng.uniform(y_low, y_high, n)
    return np.column_stack([bank_x, bank_y])


def add_geo_data(df: pd.DataFrame, cfg: Config, banks_xy: np.ndarray) -> pd.DataFrame:
    """Add geospatial features based on distance to bank locations.

    Creates the following features:
    - client_x, client_y: Client coordinates
    - min_bank_distance: Distance to nearest bank
    - mean_bank_distance: Average distance to all banks
    - std_bank_distance: Standard deviation of distances
    - distance_balance_risk: Composite risk indicator

    Args:
        df: Input DataFrame
        cfg: Configuration object
        banks_xy: Array of bank coordinates (n_banks, 2)

    Returns:
        DataFrame with added geospatial features
    """
    rng = np.random.default_rng(cfg.random_state)
    x_low, x_high = cfg.geo.bounds["x"]
    y_low, y_high = cfg.geo.bounds["y"]

    # Generate client locations
    df["client_x"] = rng.uniform(x_low, x_high, size=len(df))
    df["client_y"] = rng.uniform(y_low, y_high, size=len(df))

    # Calculate distances using KNN
    client_xy = df[["client_x", "client_y"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(len(banks_xy), cfg.geo.n_banks))
    nn.fit(banks_xy)
    distances, indices = nn.kneighbors(client_xy)

    # Distance features
    df["min_bank_distance"] = distances[:, MIN_DISTANCE_INDEX]
    df["mean_bank_distance"] = distances.mean(axis=1)
    df["std_bank_distance"] = distances.std(axis=1)

    # Composite risk indicator
    df["distance_balance_risk"] = df["min_bank_distance"] * (1 / (df["avg_monthly_balance"] + EPSILON))

    return df


def inject_distance_label_signal(df: pd.DataFrame, cfg: Config) -> None:
    """Inject intentional label bias for distant clients (for testing monotonic constraints).

    Flips labels for a percentage of far-distance non-defaulters to defaults,
    creating a monotonic relationship between distance and default probability.

    Args:
        df: DataFrame with distance features and target column (modified in-place)
        cfg: Configuration object

    Returns:
        None (modifies df in-place)
    """
    if not cfg.geo.inject_label_signal:
        return

    qthr = df["min_bank_distance"].quantile(cfg.geo.signal_quantile)
    far_mask = df["min_bank_distance"] >= qthr
    rng = np.random.default_rng(cfg.random_state)

    # Flip some non-defaults to defaults in far distance group
    to_flip = (df.loc[far_mask, cfg.features.target_col] == 0) & (
        rng.random(far_mask.sum()) < cfg.geo.label_signal_strength
    )
    df.loc[far_mask, cfg.features.target_col] = np.where(to_flip, 1, df.loc[far_mask, cfg.features.target_col])

    logger.info(f"Injected label signal: flipped {to_flip.sum()} labels in far distance group")


def build_monotone_constraints(feature_cols: list[str], cfg: Config) -> list[int]:
    """Build monotonic constraints vector for LightGBM.

    Enforces that distance features monotonically increase default probability:
    - min_bank_distance: +1 (more distance = higher risk)
    - mean_bank_distance: +1
    - std_bank_distance: +1
    - distance_balance_risk: +1

    Args:
        feature_cols: List of all feature column names
        cfg: Configuration object

    Returns:
        List of constraint values (0 = no constraint, 1 = monotonic increase)
    """
    if not cfg.geo.enforce_monotonic:
        return []

    # Define features that should monotonically increase default probability
    inc_cols = {"min_bank_distance", "mean_bank_distance", "std_bank_distance", "distance_balance_risk"}

    constraints = [MONOTONIC_INCREASE if f in inc_cols else NO_CONSTRAINT for f in feature_cols]
    constrained_features = [feature_cols[i] for i, c in enumerate(constraints) if c == MONOTONIC_INCREASE]

    logger.info(f"Applied monotonic constraints to {len(constrained_features)} features: {constrained_features}")

    return constraints
