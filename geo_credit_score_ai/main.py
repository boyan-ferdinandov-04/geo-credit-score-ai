"""
Credit risk model training pipeline with geospatial features.

This script trains a LightGBM classifier to predict credit default risk,
incorporating distance-based features to bank locations and optional calibration.

Usage:
    python main.py
"""

import logging
import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import Config
from src.data import create_dataset, engineer_features
from src.geo import (
    add_geo_data,
    build_monotone_constraints,
    create_bank_locations,
    inject_distance_label_signal,
)
from src.models import build_pipeline, evaluate_model
from src.validation import validate_dataset, validate_train_test_split

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logger.info("Loading config")
    config = Config.from_yaml("config/model_config.yaml")

    df = create_dataset(config)
    logger.info(
        f"Dataset: {len(df)} samples, target distribution: {df[config.features.target_col].value_counts().to_dict()}"
    )

    # Validate raw dataset
    validate_dataset(df, config, stage="raw")

    banks_xy = create_bank_locations(config)
    df = add_geo_data(df, config, banks_xy)
    inject_distance_label_signal(df, config)

    df, feature_cols = engineer_features(df, config)
    geo_cols = ["client_x", "client_y"] + config.geo.distance_features + ["distance_balance_risk"]
    feature_cols = list(dict.fromkeys(feature_cols + geo_cols))

    constraints = build_monotone_constraints(feature_cols, config)

    model_params = config.model.lgbm_params.copy()
    pos_weight = (df[config.features.target_col] == 0).sum() / (df[config.features.target_col] == 1).sum()
    model_params["scale_pos_weight"] = pos_weight
    if constraints:
        model_params["monotone_constraints"] = constraints
        model_params["monotone_constraints_method"] = "advanced"
        model_params["monotone_penalty"] = 2.0

    X = df[feature_cols]
    y = df[config.features.target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size, stratify=y, random_state=config.random_state
    )

    # Validate train/test split
    validate_train_test_split(X_train, X_test, y_train, y_test)

    pipeline = build_pipeline(model_params, use_smote=config.model.use_smote)

    logger.info("Cross-validation starting")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config.model.cv_folds, scoring="roc_auc")
    logger.info(f"Mean CV AUC: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # ---------------- Calibration ----------------
    # Apply calibration by wrapping the entire pipeline BEFORE fitting
    if getattr(config, "calibration", None) and config.calibration.enabled:
        logger.info(f"Applying {config.calibration.method} calibration with CV={config.calibration.cv}")
        pipeline = CalibratedClassifierCV(
            estimator=pipeline,
            method=config.calibration.method,
            cv=config.calibration.cv,
        )

    pipeline.fit(X_train, y_train)

    eval_results = evaluate_model(pipeline, X_test, y_test, fp_cost=config.model.fp_cost, fn_cost=config.model.fn_cost)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
