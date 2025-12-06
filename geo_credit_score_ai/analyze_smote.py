"""
SMOTE Analysis Script - Comparing Model Performance with and without SMOTE.

This script trains two models (with and without SMOTE oversampling) and provides
a comprehensive comparison of their performance across multiple metrics.

Usage:
    uv run python analyze_smote.py
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import Config
from src.data import create_dataset, engineer_features
from src.geo import (
    add_geo_data,
    build_monotone_constraints,
    create_bank_locations,
    inject_distance_label_signal,
)
from src.models import build_pipeline
from src.visualization import (
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_analysis,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_class_distribution(y: pd.Series, label: str = "") -> dict[str, Any]:
    """Get class distribution statistics.

    Args:
        y: Target labels
        label: Label for logging

    Returns:
        Dictionary with class distribution metrics
    """
    counts = y.value_counts().to_dict()
    total = len(y)
    minority_pct = (counts.get(1, 0) / total) * 100
    majority_pct = (counts.get(0, 0) / total) * 100

    return {
        "label": label,
        "total": total,
        "class_0": counts.get(0, 0),
        "class_1": counts.get(1, 0),
        "class_0_pct": majority_pct,
        "class_1_pct": minority_pct,
        "imbalance_ratio": counts.get(0, 0) / counts.get(1, 1),
    }


def calculate_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, fp_cost: float = 100.0, fn_cost: float = 2500.0
) -> tuple[float, float]:
    """Calculate optimal classification threshold based on business costs.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative

    Returns:
        Tuple of (optimal_threshold, minimum_cost)
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    min_cost = float("inf")
    optimal_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fp * fp_cost) + (fn * fn_cost)

        if total_cost < min_cost:
            min_cost = total_cost
            optimal_thresh = thresh

    return optimal_thresh, min_cost


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_params: dict[str, Any],
    config: Config,
    use_smote: bool,
) -> dict[str, Any]:
    """Train and evaluate a single model configuration.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_params: LightGBM parameters
        config: Configuration object
        use_smote: Whether to use SMOTE

    Returns:
        Dictionary with evaluation metrics and predictions
    """
    model_name = "With SMOTE" if use_smote else "Without SMOTE"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training Model: {model_name}")
    logger.info(f"{'=' * 60}")

    train_dist_before = get_class_distribution(y_train, f"{model_name} - Training (before)")
    logger.info(
        f"Training set: {train_dist_before['class_0']} non-defaults, "
        f"{train_dist_before['class_1']} defaults "
        f"({train_dist_before['class_1_pct']:.1f}% defaults)"
    )

    pipeline = build_pipeline(model_params, use_smote=use_smote)

    logger.info("Running cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=config.model.cv_folds, scoring="roc_auc")
    logger.info(f"Mean CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if config.calibration.enabled:
        logger.info(f"Applying {config.calibration.method} calibration...")
        pipeline = CalibratedClassifierCV(
            estimator=pipeline,
            method=config.calibration.method,
            cv=config.calibration.cv,
        )

    logger.info("Training on full training set...")
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = pipeline.predict(X_test)  # Using default 0.5 threshold

    optimal_thresh, min_cost = calculate_optimal_threshold(
        y_test.values, y_proba, config.model.fp_cost, config.model.fn_cost
    )
    y_pred_optimal = (y_proba >= optimal_thresh).astype(int)

    # Metrics with default threshold (0.5)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)

    # Metrics with optimal threshold
    precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)

    # Confusion matrices
    cm_default = confusion_matrix(y_test, y_pred_default)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)

    tn_default, fp_default, fn_default, tp_default = cm_default.ravel()
    cost_default = (fp_default * config.model.fp_cost) + (fn_default * config.model.fn_cost)

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  ROC AUC: {auc:.4f}")
    logger.info(f"  Brier Score: {brier:.4f}")
    logger.info(f"  Gini Coefficient: {2 * auc - 1:.4f}")
    logger.info(
        f"\n  Threshold 0.5 - Precision: {precision_default:.4f}, Recall: {recall_default:.4f}, F1: {f1_default:.4f}"
    )
    logger.info(f"  Threshold 0.5 - Business Cost: ${cost_default:,.0f}")
    logger.info(f"\n  Optimal Threshold: {optimal_thresh:.3f}")
    logger.info(f"  Optimal - Precision: {precision_optimal:.4f}, Recall: {recall_optimal:.4f}, F1: {f1_optimal:.4f}")
    logger.info(f"  Optimal - Business Cost: ${min_cost:,.0f} (saved ${cost_default - min_cost:,.0f})")

    return {
        "model_name": model_name,
        "use_smote": use_smote,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "auc": auc,
        "brier": brier,
        "gini": 2 * auc - 1,
        "precision_default": precision_default,
        "recall_default": recall_default,
        "f1_default": f1_default,
        "cost_default": cost_default,
        "optimal_threshold": optimal_thresh,
        "precision_optimal": precision_optimal,
        "recall_optimal": recall_optimal,
        "f1_optimal": f1_optimal,
        "cost_optimal": min_cost,
        "cost_savings": cost_default - min_cost,
        "y_proba": y_proba,
        "y_pred_default": y_pred_default,
        "y_pred_optimal": y_pred_optimal,
        "cm_default": cm_default,
        "cm_optimal": cm_optimal,
        "train_dist": train_dist_before,
    }


def create_comparison_table(results_no_smote: dict[str, Any], results_with_smote: dict[str, Any]) -> pd.DataFrame:
    """Create comparison table of metrics.

    Args:
        results_no_smote: Results from model without SMOTE
        results_with_smote: Results from model with SMOTE

    Returns:
        DataFrame with comparison metrics
    """
    metrics = {
        "Metric": [
            "Cross-Val AUC (mean ± std)",
            "Test ROC AUC",
            "Brier Score",
            "Gini Coefficient",
            "",
            "Precision (thresh=0.5)",
            "Recall (thresh=0.5)",
            "F1 Score (thresh=0.5)",
            "Business Cost (thresh=0.5)",
            "",
            "Optimal Threshold",
            "Precision (optimal)",
            "Recall (optimal)",
            "F1 Score (optimal)",
            "Business Cost (optimal)",
            "Cost Savings vs 0.5",
        ],
        "Without SMOTE": [
            f"{results_no_smote['cv_mean']:.4f} ± {results_no_smote['cv_std']:.4f}",
            f"{results_no_smote['auc']:.4f}",
            f"{results_no_smote['brier']:.4f}",
            f"{results_no_smote['gini']:.4f}",
            "",
            f"{results_no_smote['precision_default']:.4f}",
            f"{results_no_smote['recall_default']:.4f}",
            f"{results_no_smote['f1_default']:.4f}",
            f"${results_no_smote['cost_default']:,.0f}",
            "",
            f"{results_no_smote['optimal_threshold']:.3f}",
            f"{results_no_smote['precision_optimal']:.4f}",
            f"{results_no_smote['recall_optimal']:.4f}",
            f"{results_no_smote['f1_optimal']:.4f}",
            f"${results_no_smote['cost_optimal']:,.0f}",
            f"${results_no_smote['cost_savings']:,.0f}",
        ],
        "With SMOTE": [
            f"{results_with_smote['cv_mean']:.4f} ± {results_with_smote['cv_std']:.4f}",
            f"{results_with_smote['auc']:.4f}",
            f"{results_with_smote['brier']:.4f}",
            f"{results_with_smote['gini']:.4f}",
            "",
            f"{results_with_smote['precision_default']:.4f}",
            f"{results_with_smote['recall_default']:.4f}",
            f"{results_with_smote['f1_default']:.4f}",
            f"${results_with_smote['cost_default']:,.0f}",
            "",
            f"{results_with_smote['optimal_threshold']:.3f}",
            f"{results_with_smote['precision_optimal']:.4f}",
            f"{results_with_smote['recall_optimal']:.4f}",
            f"{results_with_smote['f1_optimal']:.4f}",
            f"${results_with_smote['cost_optimal']:,.0f}",
            f"${results_with_smote['cost_savings']:,.0f}",
        ],
    }

    diff_values = []
    for i, metric in enumerate(metrics["Metric"]):
        if metric == "" or "Threshold" in metric:
            diff_values.append("")
        elif "Cost" in metric and "$" in metrics["Without SMOTE"][i]:
            # Parse costs
            no_smote_val = float(metrics["Without SMOTE"][i].replace("$", "").replace(",", ""))
            smote_val = float(metrics["With SMOTE"][i].replace("$", "").replace(",", ""))
            diff = smote_val - no_smote_val
            symbol = "↓" if diff < 0 else "↑"
            diff_values.append(f"${abs(diff):,.0f} {symbol}")
        elif "±" not in metrics["Without SMOTE"][i]:
            try:
                no_smote_val = float(metrics["Without SMOTE"][i])
                smote_val = float(metrics["With SMOTE"][i])
                diff = smote_val - no_smote_val
                symbol = "↑" if diff > 0 else "↓"
                diff_values.append(f"{abs(diff):.4f} {symbol}")
            except ValueError:
                diff_values.append("")
        else:
            diff_values.append("")

    metrics["Difference"] = diff_values

    return pd.DataFrame(metrics)


def plot_smote_comparison(
    results_no_smote: dict[str, Any],
    results_with_smote: dict[str, Any],
    y_test: pd.Series,
    config: Config,
    save_path: str = "smote_comparison.png",
):
    """Create comparison visualizations.

    Args:
        results_no_smote: Results from model without SMOTE
        results_with_smote: Results from model with SMOTE
        y_test: Test labels
        config: Configuration object
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("SMOTE vs No SMOTE - Performance Comparison", fontsize=18, fontweight="bold", y=0.98)

    # Row 1: Without SMOTE
    plot_roc_curve(
        y_test.values,
        results_no_smote["y_proba"],
        results_no_smote["auc"],
        ax=axes[0, 0],
        title="ROC Curve - Without SMOTE",
    )

    plot_precision_recall_curve(
        y_test.values, results_no_smote["y_proba"], ax=axes[0, 1], title="Precision-Recall - Without SMOTE"
    )

    plot_threshold_analysis(
        y_test.values,
        results_no_smote["y_proba"],
        fp_cost=config.model.fp_cost,
        fn_cost=config.model.fn_cost,
        ax=axes[0, 2],
        title="Business Cost - Without SMOTE",
    )

    # Row 2: With SMOTE
    plot_roc_curve(
        y_test.values,
        results_with_smote["y_proba"],
        results_with_smote["auc"],
        ax=axes[1, 0],
        title="ROC Curve - With SMOTE",
    )

    plot_precision_recall_curve(
        y_test.values, results_with_smote["y_proba"], ax=axes[1, 1], title="Precision-Recall - With SMOTE"
    )

    plot_threshold_analysis(
        y_test.values,
        results_with_smote["y_proba"],
        fp_cost=config.model.fp_cost,
        fn_cost=config.model.fn_cost,
        ax=axes[1, 2],
        title="Business Cost - With SMOTE",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison plot saved to: {save_path}")


def main():
    logger.info("=" * 80)
    logger.info("SMOTE Analysis - Comparing Model Performance")
    logger.info("=" * 80)

    logger.info("\nLoading configuration...")
    config = Config.from_yaml("config/model_config.yaml")

    logger.info("Creating dataset...")
    df = create_dataset(config)
    logger.info(
        f"Dataset: {len(df)} samples, target distribution: {df[config.features.target_col].value_counts().to_dict()}"
    )

    logger.info("Adding geospatial features...")
    banks_xy = create_bank_locations(config)
    df = add_geo_data(df, config, banks_xy)
    inject_distance_label_signal(df, config)

    logger.info("Engineering features...")
    df, feature_cols = engineer_features(df, config)
    geo_cols = ["client_x", "client_y"] + config.geo.distance_features + ["distance_balance_risk"]
    feature_cols = list(dict.fromkeys(feature_cols + geo_cols))
    logger.info(f"Total features: {len(feature_cols)}")

    constraints = build_monotone_constraints(feature_cols, config)

    model_params = config.model.lgbm_params.copy()
    pos_weight = (df[config.features.target_col] == 0).sum() / (df[config.features.target_col] == 1).sum()
    model_params["scale_pos_weight"] = pos_weight

    if constraints:
        model_params["monotone_constraints"] = constraints
        model_params["monotone_constraints_method"] = "advanced"
        model_params["monotone_penalty"] = 2.0

    # 7. Train/test split
    logger.info("Splitting data...")
    X = df[feature_cols]
    y = df[config.features.target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size, stratify=y, random_state=config.random_state
    )

    test_dist = get_class_distribution(y_test, "Test set")
    logger.info(
        f"Test set: {test_dist['class_0']} non-defaults, "
        f"{test_dist['class_1']} defaults ({test_dist['class_1_pct']:.1f}% defaults)"
    )

    results_no_smote = train_and_evaluate(X_train, X_test, y_train, y_test, model_params, config, use_smote=False)

    results_with_smote = train_and_evaluate(X_train, X_test, y_train, y_test, model_params, config, use_smote=True)

    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON TABLE")
    logger.info("=" * 80)
    comparison_df = create_comparison_table(results_no_smote, results_with_smote)
    print("\n" + comparison_df.to_string(index=False))

    logger.info("\n" + "=" * 80)
    logger.info("INTERPRETATION")
    logger.info("=" * 80)

    auc_diff = results_with_smote["auc"] - results_no_smote["auc"]
    recall_diff = results_with_smote["recall_optimal"] - results_no_smote["recall_optimal"]
    precision_diff = results_with_smote["precision_optimal"] - results_no_smote["precision_optimal"]
    cost_diff = results_with_smote["cost_optimal"] - results_no_smote["cost_optimal"]

    logger.info("\nKey Findings:")
    logger.info(f"1. AUC Impact: SMOTE {'improved' if auc_diff > 0 else 'reduced'} AUC by {abs(auc_diff):.4f}")
    logger.info(
        f"2. Recall Impact: SMOTE {'improved' if recall_diff > 0 else 'reduced'} recall by {abs(recall_diff):.4f}"
    )
    logger.info(
        f"3. Precision Impact: SMOTE {'improved' if precision_diff > 0 else 'reduced'} precision by {abs(precision_diff):.4f}"
    )
    logger.info(
        f"4. Business Cost: SMOTE {'reduced' if cost_diff < 0 else 'increased'} costs by ${abs(cost_diff):,.0f}"
    )

    logger.info("\nRecommendation:")
    if auc_diff > 0.01 and cost_diff < 0:
        logger.info("✓ SMOTE significantly improves both AUC and business costs. RECOMMENDED for production.")
    elif auc_diff > 0 and cost_diff < 0:
        logger.info("✓ SMOTE improves both AUC and business costs. Consider using SMOTE.")
    elif cost_diff < 0:
        logger.info("→ SMOTE reduces business costs despite lower AUC. Consider SMOTE if cost is primary objective.")
    elif auc_diff > 0:
        logger.info("→ SMOTE improves AUC but increases costs. Consider SMOTE if AUC is primary objective.")
    else:
        logger.info("✗ SMOTE does not provide clear benefits. NOT RECOMMENDED for this dataset.")

    # 12. Create visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Generating comparison visualizations...")
    plot_smote_comparison(results_no_smote, results_with_smote, y_test, config)

    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
