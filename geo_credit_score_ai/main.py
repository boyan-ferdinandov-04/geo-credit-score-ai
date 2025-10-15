"""Credit risk model training pipeline with geospatial features.

This script trains a LightGBM classifier to predict credit default risk,
incorporating distance-based features to bank locations.

Usage:
    python main.py
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# =============================================================================
# Dataset Creation
# =============================================================================


def create_dataset(cfg: Config) -> pd.DataFrame:
    """Creates a synthetic dataset that loosely resembles credit data."""
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
    """Engineers new features and creates missing value indicators."""
    df["income_log"] = np.log1p(np.abs(df["income"]))
    df["delinq_utilization"] = df["delinquencies"] * df["utilization"]
    df["credit_age_ratio"] = df["credit_history_len"] / (df["age"] + 1)

    for col in cfg.features.missing_indicator_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    feature_cols = (
        cfg.features.original_cols
        + ["income_log", "delinq_utilization", "credit_age_ratio"]
        + [f"{col}_missing" for col in cfg.features.missing_indicator_cols]
    )
    return df, feature_cols


# =============================================================================
# Geospatial Features
# =============================================================================


def create_bank_locations(cfg: Config) -> np.ndarray:
    """Populates the bounded area with banks at random points."""
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.geo.n_banks
    x_low, x_high = cfg.geo.bounds["x"]
    y_low, y_high = cfg.geo.bounds["y"]
    bank_x = rng.uniform(x_low, x_high, n)
    bank_y = rng.uniform(y_low, y_high, n)
    return np.column_stack([bank_x, bank_y])


def add_geo_data(df: pd.DataFrame, cfg: Config, banks_xy: np.ndarray) -> pd.DataFrame:
    """Adds geospatial features to dataframe."""
    rng = np.random.default_rng(cfg.random_state)
    x_low, x_high = cfg.geo.bounds["x"]
    y_low, y_high = cfg.geo.bounds["y"]

    df["client_x"] = rng.uniform(x_low, x_high, size=len(df))
    df["client_y"] = rng.uniform(y_low, y_high, size=len(df))

    client_xy = df[["client_x", "client_y"]].to_numpy()
    diff_x = client_xy[:, [0]] - banks_xy[:, 0]
    diff_y = client_xy[:, [1]] - banks_xy[:, 1]
    dists = np.sqrt(diff_x**2 + diff_y**2)

    df["min_bank_distance"] = dists.min(axis=1)
    df["mean_bank_distance"] = dists.mean(axis=1)
    df["std_bank_distance"] = dists.std(axis=1)

    return df


def inject_distance_label_signal(df: pd.DataFrame, cfg: Config) -> None:
    """Bias towards clients that are further away."""
    if not cfg.geo.inject_label_signal:
        return

    qthr = df["min_bank_distance"].quantile(cfg.geo.signal_quantile)
    far_mask = df["min_bank_distance"] >= qthr

    rng = np.random.default_rng(cfg.random_state)
    to_flip = (df.loc[far_mask, cfg.features.target_col] == 0) & (
        rng.random(far_mask.sum()) < cfg.geo.label_signal_strength
    )
    df.loc[far_mask, cfg.features.target_col] = np.where(to_flip, 1, df.loc[far_mask, cfg.features.target_col])


def build_monotone_constraints(feature_cols: list[str], cfg: Config) -> list[int]:
    """Builds monotonic constraints for distance features."""
    if not cfg.geo.enforce_monotonic:
        return []

    inc_cols = {"min_bank_distance", "mean_bank_distance"}
    constraints = []
    for col in feature_cols:
        if col in inc_cols:
            constraints.append(1)
        else:
            constraints.append(0)
    return constraints


# =============================================================================
# Model Training
# =============================================================================


def build_pipeline(model_params: dict) -> Pipeline:
    """Builds a scikit-learn pipeline for preprocessing and modeling."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(**model_params)),
        ]
    )


# =============================================================================
# Model Evaluation
# =============================================================================


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Calculates and prints model evaluation metrics."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print("\n--- Model Performance ---")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Gini Coefficient: {2 * auc - 1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    thresholds = np.arange(0.05, 0.5, 0.01)
    f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    print(f"Optimal Threshold (by F1-score): {thresholds[optimal_idx]:.3f} with F1-score: {f1_scores[optimal_idx]:.4f}")

    return {"y_proba": y_proba, "y_pred": y_pred, "auc": auc}


def plot_diagnostics(
    y_test: pd.Series,
    y_proba: np.ndarray,
    feature_importance: pd.DataFrame,
    auc: float,
    top_n_features: int,
):
    """Generates a 2x2 plot of model diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Model Performance & Diagnostics", fontsize=16)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", c="gray", linewidth=0.7)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = np.trapezoid(precision, recall)
    axes[0, 1].plot(recall, precision, label=f"AP = {ap:.3f}")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()

    # Calibration Plot
    calib_df = pd.DataFrame({"y_true": y_test, "y_prob": y_proba})
    calib_df["bin"] = pd.cut(calib_df["y_prob"], bins=np.arange(0, 1.1, 0.1), right=False)
    calib_grouped = (
        calib_df.groupby("bin", observed=False)
        .agg(observed_rate=("y_true", "mean"), predicted_prob=("y_prob", "mean"), count=("y_true", "count"))
        .reset_index()
    )
    axes[1, 0].plot(calib_grouped["predicted_prob"], calib_grouped["observed_rate"], marker="o", label="Model")
    axes[1, 0].plot([0, 1], [0, 1], "--", c="gray", linewidth=0.7, label="Perfectly Calibrated")
    axes[1, 0].set_xlabel("Mean Predicted Probability")
    axes[1, 0].set_ylabel("Observed Default Rate")
    axes[1, 0].set_title("Calibration Plot")
    axes[1, 0].legend()

    # Feature Importance
    top_features = feature_importance.head(top_n_features)
    axes[1, 1].barh(top_features["feature"], top_features["importance"])
    axes[1, 1].set_xlabel("Feature Importance")
    axes[1, 1].set_title(f"Top {top_n_features} Features")
    axes[1, 1].invert_yaxis()

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run the full training pipeline."""
    # Load configuration
    config = Config.from_yaml("config/model_config.yaml")
    print("✅ Loaded config from: config/model_config.yaml")

    # Create dataset
    df = create_dataset(config)

    # Add geospatial features
    banks_xy = create_bank_locations(config)
    df = add_geo_data(df, config, banks_xy)

    # Add geo columns to feature list
    geo_cols = ["client_x", "client_y"] + config.geo.distance_features
    config.features.original_cols.extend(geo_cols)

    # Inject distance signal
    inject_distance_label_signal(df, config)

    # Engineer features
    df, feature_cols = engineer_features(df, config)

    # Build monotonic constraints
    constraints = build_monotone_constraints(feature_cols, config)
    model_params = config.model.lgbm_params.copy()
    if constraints:
        model_params["monotone_constraints"] = constraints
        model_params["monotone_constraints_method"] = "advanced"

    # Prepare data
    X = df[feature_cols]
    y = df[config.features.target_col]

    print(f"Total features being used: {len(feature_cols)}")
    print(f"Missing values in features: {X.isnull().sum().sum()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size, stratify=y, random_state=config.random_state
    )

    # Build pipeline
    pipeline = build_pipeline(model_params)

    # Cross-validation
    print("\n--- Cross-Validation ---")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config.model.cv_folds, scoring="roc_auc")
    print(f"Mean Cross-validation AUC: {cv_scores.mean():.6f} (± {cv_scores.std():.6f})")

    # Train final model
    pipeline.fit(X_train, y_train)

    # Evaluate
    eval_results = evaluate_model(pipeline, X_test, y_test)

    # Feature importance
    feature_importance = (
        pd.DataFrame({"feature": feature_cols, "importance": pipeline.named_steps["clf"].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Top 5 Most Important Features ---")
    print(feature_importance.head())

    # Plot diagnostics
    plot_diagnostics(
        y_test=y_test,
        y_proba=eval_results["y_proba"],
        feature_importance=feature_importance,
        auc=eval_results["auc"],
        top_n_features=config.visualization.top_n_features,
    )

    # Distance-based probability analysis
    assert "min_bank_distance" in X_test.columns, "min_bank_distance missing from X_test"

    proba_test = pd.Series(eval_results["y_proba"], index=X_test.index, name="proba")
    dist_test = X_test["min_bank_distance"]

    with np.errstate(invalid="ignore"):
        deciles_test = pd.qcut(dist_test, 10, duplicates="drop")

    monotab_test = (
        pd.DataFrame({"decile": deciles_test, "proba": proba_test})
        .groupby("decile", observed=False)["proba"]
        .mean()
        .reset_index()
    )

    print("\nMean predicted probability to default by distance decile:")
    print(monotab_test)


if __name__ == "__main__":
    main()
