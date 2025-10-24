"""
Credit risk model training pipeline with geospatial features.

This script trains a LightGBM classifier to predict credit default risk,
incorporating distance-based features to bank locations and optional calibration.

Usage:
    python main.py
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =============================================================================
# Dataset Creation
# =============================================================================
def create_dataset(cfg: Config) -> pd.DataFrame:
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

    rng = np.random.default_rng(cfg.random_state)
    mask = rng.choice(
        [False, True],
        size=df[cfg.features.original_cols].shape,
        p=[1 - cfg.missing_values.p_missing, cfg.missing_values.p_missing],
    )
    df[cfg.features.original_cols] = df[cfg.features.original_cols].mask(mask)
    return df


# =============================================================================
# Feature Engineering
# =============================================================================
def engineer_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    df["income_log"] = np.log1p(np.abs(df["income"]))
    df["delinq_utilization"] = df["delinquencies"] * df["utilization"]
    df["credit_age_ratio"] = df["credit_history_len"] / (df["age"] + 1)
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["balance_to_income"] = df["avg_monthly_balance"] / (df["income"] + 1)
    df["delinq_per_creditline"] = df["delinquencies"] / (df["num_credit_lines"] + 1)
    df["utilization_squared"] = df["utilization"] ** 2
    df["credit_utilization_interaction"] = df["credit_history_len"] * df["utilization"]
    df["payment_stress"] = df["utilization"] * 0.4 + (df["delinquencies"] / 10) * 0.6
    df["longterm_loan_flag"] = (df["loan_term"] > df["loan_term"].quantile(0.75)).astype(int)

    for col in cfg.features.missing_indicator_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

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


# =============================================================================
# Geospatial Features
# =============================================================================
def create_bank_locations(cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.geo.n_banks
    x_low, x_high = cfg.geo.bounds["x"]
    y_low, y_high = cfg.geo.bounds["y"]
    bank_x = rng.uniform(x_low, x_high, n)
    bank_y = rng.uniform(y_low, y_high, n)
    return np.column_stack([bank_x, bank_y])


def add_geo_data(df: pd.DataFrame, cfg: Config, banks_xy: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    x_low, x_high = cfg.geo.bounds["x"]
    y_low, y_high = cfg.geo.bounds["y"]
    df["client_x"] = rng.uniform(x_low, x_high, size=len(df))
    df["client_y"] = rng.uniform(y_low, y_high, size=len(df))
    client_xy = df[["client_x", "client_y"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(len(banks_xy), cfg.geo.n_banks))
    nn.fit(banks_xy)
    distances, indices = nn.kneighbors(client_xy)
    df["min_bank_distance"] = distances[:, 0]
    df["mean_bank_distance"] = distances.mean(axis=1)
    df["std_bank_distance"] = distances.std(axis=1)
    df["distance_balance_risk"] = df["min_bank_distance"] * (1 / (df["avg_monthly_balance"] + 1))
    return df


def inject_distance_label_signal(df: pd.DataFrame, cfg: Config) -> None:
    if not cfg.geo.inject_label_signal:
        return
    qthr = df["min_bank_distance"].quantile(cfg.geo.signal_quantile)
    far_mask = df["min_bank_distance"] >= qthr
    rng = np.random.default_rng(cfg.random_state)
    to_flip = (df.loc[far_mask, cfg.features.target_col] == 0) & (
        rng.random(far_mask.sum()) < cfg.geo.label_signal_strength
    )
    df.loc[far_mask, cfg.features.target_col] = np.where(to_flip, 1, df.loc[far_mask, cfg.features.target_col])
    logger.info(f"Injected label signal: flipped {to_flip.sum()} labels in far distance group")


def build_monotone_constraints(feature_cols: list[str], cfg: Config) -> list[int]:
    if not cfg.geo.enforce_monotonic:
        return []
    inc_cols = {"min_bank_distance", "mean_bank_distance", "std_bank_distance", "distance_balance_risk"}
    constraints = [1 if f in inc_cols else 0 for f in feature_cols]
    constrained_features = [feature_cols[i] for i, c in enumerate(constraints) if c == 1]
    logger.info(f"Applied monotonic constraints to {len(constrained_features)} features: {constrained_features}")
    return constraints


# =============================================================================
# Pipeline & Evaluation
# =============================================================================
def build_pipeline(model_params: dict[str, Any], use_smote: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
    steps.append(("clf", LGBMClassifier(**model_params)))
    return Pipeline(steps)


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, fp_cost=100, fn_cost=5000):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    logger.info(f"ROC AUC: {auc:.4f}, Brier: {brier:.4f}, Gini: {2 * auc - 1:.4f}")
    logger.info("\nClassification report (0.5 threshold):\n" + classification_report(y_test, y_pred))
    return {"y_proba": y_proba, "y_pred": y_pred, "auc": auc, "brier": brier}


# =============================================================================
# Main Execution
# =============================================================================
def main():
    logger.info("Loading config")
    config = Config.from_yaml("config/model_config.yaml")

    df = create_dataset(config)
    logger.info(
        f"Dataset: {len(df)} samples, target distribution: {df[config.features.target_col].value_counts().to_dict()}"
    )

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

    pipeline = build_pipeline(model_params, use_smote=config.model.use_smote)

    logger.info("Cross-validation starting")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config.model.cv_folds, scoring="roc_auc")
    logger.info(f"Mean CV AUC: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    pipeline.fit(X_train, y_train)

    # ---------------- Calibration ----------------
    if getattr(config, "calibration", None) and config.calibration.enabled:
        clf = pipeline.named_steps["clf"]
        calibrated = CalibratedClassifierCV(
            estimator=clf,
            method=config.calibration.method,
            cv=config.calibration.cv,
        )
        calibrated.fit(X_train, y_train)
        pipeline.named_steps["clf"] = calibrated
        logger.info(f"Applied {config.calibration.method} calibration with CV={config.calibration.cv}")

    eval_results = evaluate_model(pipeline, X_test, y_test, fp_cost=config.model.fp_cost, fn_cost=config.model.fn_cost)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
