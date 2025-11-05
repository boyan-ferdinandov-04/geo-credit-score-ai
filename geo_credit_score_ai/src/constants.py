"""Constants used throughout the credit risk/default prediction system.

This module centralizes all magic numbers to improve code maintainability
and make business logic explicit.
"""

# Feature Engineering Constants
# ==============================

# Payment stress calculation weights
# payment_stress = utilization * UTILIZATION_WEIGHT + (delinquencies / DELINQ_DIVISOR) * DELINQ_WEIGHT
UTILIZATION_WEIGHT = 0.4
"""Weight for utilization component in payment stress calculation."""

DELINQ_WEIGHT = 0.6
"""Weight for delinquency component in payment stress calculation."""

DELINQ_DIVISOR = 10
"""Divisor to normalize delinquency values (scale down to 0-1 range)."""

# Quantile thresholds for feature creation
LONGTERM_LOAN_QUANTILE = 0.75
"""Quantile threshold for defining long-term loans (75th percentile)."""

# Division by zero protection
EPSILON = 1
"""Small constant added to denominators to avoid division by zero.
Set to 1 instead of typical small epsilon (e.g., 1e-8) to preserve
interpretability of ratio features."""

# Array indices
MIN_DISTANCE_INDEX = 0
"""Index for minimum distance in distance array (nearest bank)."""

# Constraint values for monotonic enforcement
NO_CONSTRAINT = 0
"""No monotonic constraint applied to feature."""

MONOTONIC_INCREASE = 1
"""Feature monotonically increases target probability."""

MONOTONIC_DECREASE = -1
"""Feature monotonically decreases target probability."""

# SMOTE Parameters
SMOTE_K_NEIGHBORS = 5
"""Number of nearest neighbors for SMOTE oversampling.
Standard SMOTE default is 5. This balances between:
- Too small (e.g., 3): Overfits to very local patterns, less diversity
- Too large: May include noise, longer computation time
Value must be less than minority class samples in training set."""

SMOTE_RANDOM_STATE = 42
"""Random state for SMOTE reproducibility."""
