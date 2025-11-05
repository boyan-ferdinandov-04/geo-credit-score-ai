"""Tests for pipeline building and model evaluation."""

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from main import build_pipeline, evaluate_model


class TestBuildPipeline:
    """Tests for build_pipeline function."""

    def test_returns_pipeline(self):
        """Test that function returns Pipeline object."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        assert isinstance(pipeline, Pipeline)

    def test_has_imputer_step(self):
        """Test that pipeline has imputer step."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        assert "imputer" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)

    def test_has_scaler_step(self):
        """Test that pipeline has scaler step."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        assert "scaler" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)

    def test_has_classifier_step(self):
        """Test that pipeline has classifier step."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        assert "clf" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["clf"], LGBMClassifier)

    def test_without_smote(self):
        """Test that pipeline without SMOTE has 3 steps."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        assert len(pipeline.steps) == 3
        assert "smote" not in pipeline.named_steps

    def test_with_smote(self):
        """Test that pipeline with SMOTE has 4 steps."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=True)
        assert len(pipeline.steps) == 4
        assert "smote" in pipeline.named_steps

    def test_model_params_passed(self):
        """Test that model parameters are passed to classifier."""
        model_params = {"max_depth": 5, "n_estimators": 20, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        clf = pipeline.named_steps["clf"]
        assert clf.max_depth == 5
        assert clf.n_estimators == 20
        assert clf.learning_rate == 0.1

    def test_step_order_without_smote(self):
        """Test that steps are in correct order without SMOTE."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["imputer", "scaler", "clf"]

    def test_step_order_with_smote(self):
        """Test that steps are in correct order with SMOTE."""
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=True)
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["imputer", "scaler", "smote", "clf"]

    def test_pipeline_can_fit(self, train_test_data):
        """Test that pipeline can fit on data."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)

        # Should not raise any errors
        pipeline.fit(X_train, y_train)
        assert hasattr(pipeline, "classes_")

    def test_pipeline_can_predict(self, train_test_data):
        """Test that pipeline can make predictions."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 3, "n_estimators": 10, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @pytest.fixture
    def fitted_pipeline(self, train_test_data):
        """Create and fit a pipeline for testing."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)
        pipeline.fit(X_train, y_train)
        return pipeline, X_test, y_test

    def test_returns_dict(self, fitted_pipeline):
        """Test that evaluate_model returns a dictionary."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert isinstance(result, dict)

    def test_has_required_keys(self, fitted_pipeline):
        """Test that result has required keys."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert "y_proba" in result
        assert "y_pred" in result
        assert "auc" in result
        assert "brier" in result

    def test_y_proba_correct_length(self, fitted_pipeline):
        """Test that y_proba has correct length."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert len(result["y_proba"]) == len(X_test)

    def test_y_pred_correct_length(self, fitted_pipeline):
        """Test that y_pred has correct length."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert len(result["y_pred"]) == len(X_test)

    def test_y_proba_range(self, fitted_pipeline):
        """Test that y_proba values are between 0 and 1."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert (result["y_proba"] >= 0).all()
        assert (result["y_proba"] <= 1).all()

    def test_y_pred_binary(self, fitted_pipeline):
        """Test that y_pred contains only 0 and 1."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert set(result["y_pred"]).issubset({0, 1})

    def test_auc_range(self, fitted_pipeline):
        """Test that AUC is between 0 and 1."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert 0 <= result["auc"] <= 1

    def test_brier_range(self, fitted_pipeline):
        """Test that Brier score is between 0 and 1."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        assert 0 <= result["brier"] <= 1

    def test_auc_reasonable_value(self, fitted_pipeline):
        """Test that AUC meets quality threshold."""
        pipeline, X_test, y_test = fitted_pipeline
        result = evaluate_model(pipeline, X_test, y_test)
        # Model should achieve AUC > 0.7
        assert result["auc"] > 0.7

    def test_accepts_custom_costs(self, fitted_pipeline):
        """Test that custom costs can be passed."""
        pipeline, X_test, y_test = fitted_pipeline
        # Should not raise any errors
        result = evaluate_model(pipeline, X_test, y_test, fp_cost=200, fn_cost=10000)
        assert isinstance(result, dict)


class TestPipelineIntegration:
    """Integration tests for full pipeline workflow."""

    def test_full_workflow_without_smote(self, train_test_data):
        """Test complete workflow without SMOTE."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbose": -1}

        # Build pipeline
        pipeline = build_pipeline(model_params, use_smote=False)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate
        results = evaluate_model(pipeline, X_test, y_test)

        assert results["auc"] > 0.7
        assert 0 <= results["brier"] <= 1

    def test_full_workflow_with_smote(self, train_test_data):
        """Test complete workflow with SMOTE."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbose": -1}

        # Build pipeline
        pipeline = build_pipeline(model_params, use_smote=True)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate
        results = evaluate_model(pipeline, X_test, y_test)

        assert results["auc"] > 0.7
        assert 0 <= results["brier"] <= 1

    def test_pipeline_with_missing_values(self, config):
        """Test that pipeline handles missing values correctly."""
        # Create data with missing values using proper feature names
        from sklearn.datasets import make_classification

        np.random.seed(42)
        X_np, y = make_classification(
            n_samples=200,
            n_features=12,
            n_informative=8,
            random_state=42
        )

        # Use actual feature names from config
        X = pd.DataFrame(X_np, columns=config.features.original_cols)
        y = pd.Series(y)

        # Inject missing values
        X.loc[0:20, config.features.original_cols[0]] = np.nan
        X.loc[10:30, config.features.original_cols[5]] = np.nan

        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]

        model_params = {"max_depth": 5, "n_estimators": 50, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
        pipeline = build_pipeline(model_params, use_smote=False)

        # Should handle missing values
        pipeline.fit(X_train, y_train)
        results = evaluate_model(pipeline, X_test, y_test)

        assert results["auc"] >= 0
        assert not np.isnan(results["auc"])

    def test_calibrated_pipeline(self, train_test_data):
        """Test pipeline wrapped with calibration."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbose": -1}

        # Build base pipeline
        pipeline = build_pipeline(model_params, use_smote=False)

        # Wrap with calibration
        calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=3)

        # Fit
        calibrated_pipeline.fit(X_train, y_train)

        # Evaluate
        results = evaluate_model(calibrated_pipeline, X_test, y_test)

        assert results["auc"] > 0.7
        assert 0 <= results["brier"] <= 1

    def test_reproducibility(self, train_test_data):
        """Test that same random_state produces same results."""
        X_train, X_test, y_train, y_test = train_test_data
        model_params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1, "random_state": 42, "verbose": -1}

        # Run 1
        pipeline1 = build_pipeline(model_params, use_smote=False)
        pipeline1.fit(X_train, y_train)
        results1 = evaluate_model(pipeline1, X_test, y_test)

        # Run 2
        pipeline2 = build_pipeline(model_params, use_smote=False)
        pipeline2.fit(X_train, y_train)
        results2 = evaluate_model(pipeline2, X_test, y_test)

        # Results should be identical
        assert results1["auc"] == results2["auc"]
        assert results1["brier"] == results2["brier"]
        np.testing.assert_array_equal(results1["y_pred"], results2["y_pred"])
        np.testing.assert_array_almost_equal(results1["y_proba"], results2["y_proba"])
