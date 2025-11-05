"""Tests for constants module.

These tests ensure that constants maintain their expected values
and document the business logic behind them.
"""


from src.constants import (
    DELINQ_DIVISOR,
    DELINQ_WEIGHT,
    EPSILON,
    LONGTERM_LOAN_QUANTILE,
    MIN_DISTANCE_INDEX,
    MONOTONIC_DECREASE,
    MONOTONIC_INCREASE,
    NO_CONSTRAINT,
    UTILIZATION_WEIGHT,
)


class TestPaymentStressConstants:
    """Test payment stress calculation constants."""

    def test_utilization_weight_value(self):
        """Test utilization weight is 0.4 (40% of payment stress)."""
        assert UTILIZATION_WEIGHT == 0.4

    def test_delinq_weight_value(self):
        """Test delinquency weight is 0.6 (60% of payment stress)."""
        assert DELINQ_WEIGHT == 0.6

    def test_weights_sum_to_one(self):
        """Test that payment stress weights sum to 1.0."""
        assert UTILIZATION_WEIGHT + DELINQ_WEIGHT == 1.0

    def test_delinq_divisor_value(self):
        """Test delinquency divisor is 10 (normalizes to 0-1 range)."""
        assert DELINQ_DIVISOR == 10

    def test_delinq_weight_greater_than_utilization(self):
        """Test delinquency is weighted more heavily than utilization."""
        assert DELINQ_WEIGHT > UTILIZATION_WEIGHT


class TestQuantileConstants:
    """Test quantile threshold constants."""

    def test_longterm_loan_quantile(self):
        """Test longterm loan threshold is 75th percentile."""
        assert LONGTERM_LOAN_QUANTILE == 0.75

    def test_quantile_in_valid_range(self):
        """Test quantile is between 0 and 1."""
        assert 0 <= LONGTERM_LOAN_QUANTILE <= 1


class TestEpsilonConstant:
    """Test epsilon constant for division protection."""

    def test_epsilon_value(self):
        """Test epsilon is 1 (preserves ratio interpretability)."""
        assert EPSILON == 1

    def test_epsilon_positive(self):
        """Test epsilon is positive."""
        assert EPSILON > 0


class TestArrayIndexConstants:
    """Test array index constants."""

    def test_min_distance_index(self):
        """Test minimum distance is at index 0."""
        assert MIN_DISTANCE_INDEX == 0

    def test_min_distance_index_non_negative(self):
        """Test index is non-negative."""
        assert MIN_DISTANCE_INDEX >= 0


class TestMonotonicConstraintConstants:
    """Test monotonic constraint constants."""

    def test_no_constraint_value(self):
        """Test no constraint is represented by 0."""
        assert NO_CONSTRAINT == 0

    def test_monotonic_increase_value(self):
        """Test monotonic increase is represented by 1."""
        assert MONOTONIC_INCREASE == 1

    def test_monotonic_decrease_value(self):
        """Test monotonic decrease is represented by -1."""
        assert MONOTONIC_DECREASE == -1

    def test_constraint_values_unique(self):
        """Test all constraint values are distinct."""
        values = {NO_CONSTRAINT, MONOTONIC_INCREASE, MONOTONIC_DECREASE}
        assert len(values) == 3

    def test_constraint_values_in_lgbm_range(self):
        """Test constraint values are valid for LightGBM (-1, 0, 1)."""
        valid_values = {-1, 0, 1}
        assert NO_CONSTRAINT in valid_values
        assert MONOTONIC_INCREASE in valid_values
        assert MONOTONIC_DECREASE in valid_values


class TestSMOTEConstants:
    """Test SMOTE parameter constants."""

    def test_smote_k_neighbors_value(self):
        """Test SMOTE k_neighbors is 5 (standard default)."""
        from src.constants import SMOTE_K_NEIGHBORS
        assert SMOTE_K_NEIGHBORS == 5

    def test_smote_k_neighbors_positive(self):
        """Test k_neighbors is positive."""
        from src.constants import SMOTE_K_NEIGHBORS
        assert SMOTE_K_NEIGHBORS > 0

    def test_smote_random_state_value(self):
        """Test SMOTE random state is 42 for reproducibility."""
        from src.constants import SMOTE_RANDOM_STATE
        assert SMOTE_RANDOM_STATE == 42


class TestConstantsBusinessLogic:
    """Test business logic encoded in constants."""

    def test_payment_stress_formula_example(self):
        """Test payment stress calculation with example values."""
        # Example: utilization=0.8, delinquencies=5
        utilization = 0.8
        delinquencies = 5

        payment_stress = (
            utilization * UTILIZATION_WEIGHT +
            (delinquencies / DELINQ_DIVISOR) * DELINQ_WEIGHT
        )

        # Expected: 0.8 * 0.4 + (5/10) * 0.6 = 0.32 + 0.3 = 0.62
        expected = 0.62
        assert abs(payment_stress - expected) < 0.001

    def test_epsilon_prevents_division_by_zero(self):
        """Test epsilon allows safe division."""
        value = 100
        denominator = 0

        # Without epsilon this would raise ZeroDivisionError
        result = value / (denominator + EPSILON)

        # With epsilon=1, result should be 100
        assert result == 100

    def test_longterm_loan_flag_example(self):
        """Test longterm loan flag with example data."""
        import pandas as pd

        loan_terms = pd.Series([12, 24, 36, 48, 60])
        threshold = loan_terms.quantile(LONGTERM_LOAN_QUANTILE)

        # 75th percentile of [12, 24, 36, 48, 60] is 48.0
        assert threshold == 48.0

        # Only 60 should be flagged as longterm (> 48.0)
        longterm_flags = (loan_terms > threshold).astype(int)
        assert longterm_flags.sum() == 1
        assert longterm_flags[loan_terms == 60].values[0] == 1
