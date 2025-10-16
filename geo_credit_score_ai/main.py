"""Credit risk model training pipeline with geospatial features.

This script trains a LightGBM classifier to predict credit default risk,
incorporating distance-based features to bank locations.

Usage:
    python main.py
"""

import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(name)s | %(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


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
    # Basic transformations
    df["income_log"] = np.log1p(np.abs(df["income"]))
    df["delinq_utilization"] = df["delinquencies"] * df["utilization"]
    df["credit_age_ratio"] = df["credit_history_len"] / (df["age"] + 1)

    # NEW: Enhanced feature interactions
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["balance_to_income"] = df["avg_monthly_balance"] / (df["income"] + 1)
    df["delinq_per_creditline"] = df["delinquencies"] / (df["num_credit_lines"] + 1)
    df["utilization_squared"] = df["utilization"] ** 2
    df["credit_utilization_interaction"] = df["credit_history_len"] * df["utilization"]

    # Risk composite features
    df["payment_stress"] = df["utilization"] * 0.4 + (df["delinquencies"] / 10) * 0.6

    # Loan term flags
    df["longterm_loan_flag"] = (df["loan_term"] > df["loan_term"].quantile(0.75)).astype(int)

    # Missing indicators
    for col in cfg.features.missing_indicator_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Compile all features
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

    df["distance_balance_risk"] = df["min_bank_distance"] * (1 / (df["avg_monthly_balance"] + 1))

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

    # NEW: Log label signal injection
    logger.info(f"Injected label signal: flipped {to_flip.sum()} labels in far distance group")


def build_monotone_constraints(feature_cols: list[str], cfg: Config) -> list[int]:
    """Builds monotonic constraints for distance features."""
    if not cfg.geo.enforce_monotonic:
        return []

    # NEW: All distance features should increase default probability
    inc_cols = {
        "min_bank_distance",
        "mean_bank_distance",
        "std_bank_distance",
        "distance_balance_risk",  # NEW
    }
    constraints = []
    for col in feature_cols:
        if col in inc_cols:
            constraints.append(1)
        else:
            constraints.append(0)

    # NEW: Log which features have constraints
    constrained_features = [feature_cols[i] for i, c in enumerate(constraints) if c == 1]
    logger.info(f"Applied monotonic constraints to {len(constrained_features)} features: {constrained_features}")

    return constraints


# =============================================================================
# Model Training
# =============================================================================


def build_pipeline(model_params: dict[str, Any], use_smote: bool = True) -> Pipeline:
    """Builds a scikit-learn pipeline for preprocessing and modeling."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    # NEW: Add SMOTE for class imbalance handling
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))

    steps.append(("clf", LGBMClassifier(**model_params)))

    return Pipeline(steps)


# =============================================================================
# Model Evaluation
# =============================================================================


def optimize_threshold_business(
    y_true: pd.Series,
    y_proba: np.ndarray,
    fp_cost: float = 100,
    fn_cost: float = 5000,
) -> tuple[float, float, pd.DataFrame]:
    """Find optimal threshold based on business costs.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        fp_cost: Cost of false positive (rejecting good customer)
        fn_cost: Cost of false negative (missing a default)

    Returns:
        Tuple of (optimal_threshold, optimal_cost, cost_curve_df)
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = (fp * fp_cost) + (fn * fn_cost)
        results.append(
            {
                "threshold": thresh,
                "cost": total_cost,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn": tn,
            }
        )

    cost_df = pd.DataFrame(results)
    optimal_row = cost_df.loc[cost_df["cost"].idxmin()]

    return optimal_row["threshold"], optimal_row["cost"], cost_df


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fp_cost: float = 100,
    fn_cost: float = 5000,
) -> dict[str, Any]:
    """Calculates and prints model evaluation metrics."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"ROC AUC: {auc:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")
    logger.info(f"Gini Coefficient: {2 * auc - 1:.4f}")

    logger.info("\n--- Default Threshold (0.5) Classification Report ---")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    # F1-optimized threshold
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_f1_idx]

    logger.info(f"\nOptimal F1 Threshold: {optimal_f1_threshold:.3f} (F1={f1_scores[optimal_f1_idx]:.4f})")

    # NEW: Business-optimized threshold
    logger.info("\n" + "=" * 60)
    logger.info("BUSINESS-DRIVEN THRESHOLD OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Cost assumptions: FP=${fp_cost:,.0f}, FN=${fn_cost:,.0f}")

    optimal_threshold, optimal_cost, cost_curve = optimize_threshold_business(y_test, y_proba, fp_cost, fn_cost)

    logger.info(f"Optimal Business Threshold: {optimal_threshold:.3f}")
    logger.info(f"Expected Cost: ${optimal_cost:,.0f}")

    y_pred_business = (y_proba >= optimal_threshold).astype(int)
    logger.info("\n--- Business-Optimized Classification Report ---")
    logger.info("\n" + classification_report(y_test, y_pred_business, target_names=["No Default", "Default"]))

    # NEW: Compare thresholds
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_business).ravel()
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_pred).ravel()

    logger.info("\n--- Threshold Comparison ---")
    logger.info(f"Default (0.5):  TP={tp_d:4d}, FP={fp_d:4d}, FN={fn_d:4d}, TN={tn_d:4d}")
    logger.info(f"Business ({optimal_threshold:.2f}): TP={tp_b:4d}, FP={fp_b:4d}, FN={fn_b:4d}, TN={tn_b:4d}")
    logger.info(f"Default cost:  ${(fp_d * fp_cost + fn_d * fn_cost):,.0f}")
    logger.info(
        f"Business cost: ${optimal_cost:,.0f} (savings: ${(fp_d * fp_cost + fn_d * fn_cost) - optimal_cost:,.0f})"
    )

    return {
        "y_proba": y_proba,
        "y_pred": y_pred,
        "auc": auc,
        "brier": brier,
        "optimal_threshold": optimal_threshold,
        "optimal_f1_threshold": optimal_f1_threshold,
        "cost_curve": cost_curve,
    }


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
    axes[0, 0].plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], "--", c="gray", linewidth=0.7)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = np.trapezoid(precision, recall)
    axes[0, 1].plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=2)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Calibration Plot
    calib_df = pd.DataFrame({"y_true": y_test, "y_prob": y_proba})
    calib_df["bin"] = pd.cut(calib_df["y_prob"], bins=np.arange(0, 1.1, 0.1), right=False)
    calib_grouped = (
        calib_df.groupby("bin", observed=False)
        .agg(observed_rate=("y_true", "mean"), predicted_prob=("y_prob", "mean"), count=("y_true", "count"))
        .reset_index()
    )
    axes[1, 0].plot(
        calib_grouped["predicted_prob"],
        calib_grouped["observed_rate"],
        marker="o",
        label="Model",
        linewidth=2,
        markersize=8,
    )
    axes[1, 0].plot([0, 1], [0, 1], "--", c="gray", linewidth=0.7, label="Perfectly Calibrated")
    axes[1, 0].set_xlabel("Mean Predicted Probability")
    axes[1, 0].set_ylabel("Observed Default Rate")
    axes[1, 0].set_title("Calibration Plot")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Feature Importance
    top_features = feature_importance.head(top_n_features)
    axes[1, 1].barh(top_features["feature"], top_features["importance"], color="steelblue")
    axes[1, 1].set_xlabel("Feature Importance")
    axes[1, 1].set_title(f"Top {top_n_features} Features")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    config = Config.from_yaml("config/model_config.yaml")
    logger.info("=" * 60)
    logger.info("CREDIT RISK MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info("Config loaded from: config/model_config.yaml")

    # 1) Data
    logger.info("\n--- Data Generation ---")
    df = create_dataset(config)
    logger.info(f"Created dataset: {len(df)} samples, {len(df.columns)} columns")
    logger.info(f"Target distribution: {df[config.features.target_col].value_counts().to_dict()}")

    # 2) Geo features
    banks_xy = create_bank_locations(config)
    df = add_geo_data(df, config, banks_xy)
    logger.info(f"Added {config.geo.n_banks} bank locations")

    # 3) Optional label signal
    inject_distance_label_signal(df, config)

    # 4) Engineered features
    logger.info("\n--- Feature Engineering ---")
    df, feature_cols = engineer_features(df, config)

    # 5) Include geo columns in the model feature set
    geo_cols = ["client_x", "client_y"] + config.geo.distance_features + ["distance_balance_risk"]
    feature_cols = list(dict.fromkeys(feature_cols + geo_cols))  # keep order, de-dupe

    # 6) Monotone constraints
    constraints = build_monotone_constraints(feature_cols, config)
    model_params = config.model.lgbm_params.copy()

    # NEW: Add scale_pos_weight for better imbalance handling
    pos_weight = (df[config.features.target_col] == 0).sum() / (df[config.features.target_col] == 1).sum()
    model_params["scale_pos_weight"] = pos_weight
    logger.info(f"Scale pos weight: {pos_weight:.2f}")

    if constraints:
        model_params["monotone_constraints"] = constraints
        model_params["monotone_constraints_method"] = "advanced"
        model_params["monotone_penalty"] = 2.0  # NEW: Stronger enforcement

    # 7) Train/test split
    X = df[feature_cols]
    y = df[config.features.target_col]
    logger.info(f"\nTotal features: {len(feature_cols)}")
    logger.info(
        f"Missing values: {X.isnull().sum().sum()} ({X.isnull().sum().sum() / (len(X) * len(X.columns)) * 100:.2f}%)"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size, stratify=y, random_state=config.random_state
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 8) Build pipeline with SMOTE
    use_smote = config.model.use_smote
    pipeline = build_pipeline(model_params, use_smote=use_smote)
    logger.info(f"Pipeline built (SMOTE: {use_smote})")

    # 9) Cross-validation
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION")
    logger.info("=" * 60)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config.model.cv_folds, scoring="roc_auc")
    logger.info(f"Mean CV AUC: {cv_scores.mean():.6f} (± {cv_scores.std():.6f})")
    logger.info(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

    # 10) Train final model
    logger.info("\n--- Training Final Model ---")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # 11) Evaluate with business metrics
    eval_results = evaluate_model(
        pipeline,
        X_test,
        y_test,
        fp_cost=config.model.fp_cost,
        fn_cost=config.model.fn_cost,
    )

    # 12) Feature importance
    feature_importance = (
        pd.DataFrame({"feature": feature_cols, "importance": pipeline.named_steps["clf"].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)
    logger.info("\n" + feature_importance.head(10).to_string(index=False))

    # 13) Plot diagnostics
    plot_diagnostics(
        y_test=y_test,
        y_proba=eval_results["y_proba"],
        feature_importance=feature_importance,
        auc=eval_results["auc"],
        top_n_features=config.visualization.top_n_features,
    )

    # 14) Distance-based probability analysis
    logger.info("\n" + "=" * 60)
    logger.info("DISTANCE-BASED ANALYSIS")
    logger.info("=" * 60)

    assert "min_bank_distance" in X_test.columns, "min_bank_distance missing from X_test"

    proba_test = pd.Series(eval_results["y_proba"], index=X_test.index, name="proba")
    dist_test = X_test["min_bank_distance"]

    with np.errstate(invalid="ignore"):
        deciles_test = pd.qcut(dist_test, 10, duplicates="drop")

    monotab_test = (
        pd.DataFrame({"decile": deciles_test, "proba": proba_test})
        .groupby("decile", observed=False)["proba"]
        .agg(["mean", "count"])
        .reset_index()
    )

    logger.info("Mean predicted probability by distance decile:")
    logger.info("\n" + monotab_test.to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
