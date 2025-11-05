"""Machine learning pipeline building and model evaluation."""

import logging
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.constants import SMOTE_K_NEIGHBORS, SMOTE_RANDOM_STATE

logger = logging.getLogger(__name__)


class EvaluationResults(TypedDict):
    """Type definition for model evaluation results.

    Attributes:
        y_proba: Predicted probabilities for positive class
        y_pred: Binary predictions (0 or 1)
        auc: ROC AUC score (0 to 1)
        brier: Brier score (0 to 1, lower is better)
    """
    y_proba: npt.NDArray[np.float64]
    y_pred: npt.NDArray[np.int64]
    auc: float
    brier: float


def build_pipeline(
    model_params: dict[str, Any],
    use_smote: bool = True,
    smote_k_neighbors: int = SMOTE_K_NEIGHBORS,
    smote_random_state: int = SMOTE_RANDOM_STATE,
) -> Pipeline:
    """Build sklearn pipeline with preprocessing and modeling steps.

    Pipeline steps:
    1. SimpleImputer - Fill missing values with median
    2. StandardScaler - Normalize features to zero mean, unit variance
    3. SMOTE (optional) - Oversample minority class
    4. LGBMClassifier - Gradient boosting model

    Args:
        model_params: Dictionary of LightGBM parameters
        use_smote: Whether to include SMOTE oversampling step
        smote_k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
        smote_random_state: Random state for SMOTE reproducibility (default: 42)

    Returns:
        Configured sklearn Pipeline object

    Note:
        smote_k_neighbors must be less than the number of minority class samples
        in the training set. For safety, it should be at least 2-3 samples less.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if use_smote:
        steps.append((
            "smote",
            SMOTE(random_state=smote_random_state, k_neighbors=smote_k_neighbors)
        ))
        logger.debug(f"SMOTE configured with k_neighbors={smote_k_neighbors}")

    steps.append(("clf", LGBMClassifier(**model_params)))

    return Pipeline(steps)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float = 100.0,
    fn_cost: float = 5000.0,
) -> EvaluationResults:
    """Evaluate trained model and log performance metrics.

    Calculates and logs:
    - ROC AUC Score
    - Brier Score (calibration metric)
    - Gini Coefficient
    - Classification Report (precision, recall, f1-score)

    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features DataFrame
        y_test: Test labels Series
        fp_cost: Cost of false positive in currency units (default: 100.0)
        fn_cost: Cost of false negative in currency units (default: 5000.0)

    Returns:
        EvaluationResults dictionary containing:
            - y_proba: Predicted probabilities for positive class (numpy array)
            - y_pred: Binary predictions with 0.5 threshold (numpy array)
            - auc: ROC AUC score (float between 0 and 1)
            - brier: Brier score (float between 0 and 1, lower is better)

    Note:
        fp_cost and fn_cost are currently not used in the returned metrics
        but are reserved for future business cost calculations.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    logger.info(f"ROC AUC: {auc:.4f}, Brier: {brier:.4f}, Gini: {2 * auc - 1:.4f}")
    logger.info("\nClassification report (0.5 threshold):\n" + classification_report(y_test, y_pred))

    return {
        "y_proba": y_proba,
        "y_pred": y_pred,
        "auc": float(auc),
        "brier": float(brier),
    }
