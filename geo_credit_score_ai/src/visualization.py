"""
Visualization module for model diagnostics and performance analysis.

This module provides comprehensive visualization tools for evaluating
fraud detection model performance, including:
- ROC curves and Precision-Recall curves
- Calibration plots
- Feature importance charts
- Confusion matrices
- Business cost analysis
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_roc_curve(
    y_true: npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    auc: float,
    ax: Optional[plt.Axes] = None,
    title: str = "ROC Curve",
) -> plt.Axes:
    """Plot ROC (Receiver Operating Characteristic) curve.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        auc: ROC AUC score to display
        ax: Matplotlib axes object (optional, creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})", linewidth=2.5, color="steelblue")
    ax.plot([0, 1], [0, 1], "--", c="gray", linewidth=1, label="Random Classifier", alpha=0.7)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return ax


def plot_precision_recall_curve(
    y_true: npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    ax: Optional[plt.Axes] = None,
    title: str = "Precision-Recall Curve",
) -> plt.Axes:
    """Plot Precision-Recall curve.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        ax: Matplotlib axes object (optional, creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = np.trapezoid(precision, recall)  # Average Precision

    ax.plot(recall, precision, label=f"Model (AP = {ap:.3f})", linewidth=2.5, color="forestgreen")

    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"Random (AP = {baseline:.3f})", alpha=0.7)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return ax


def plot_calibration_curve(
    y_true: npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    title: str = "Calibration Plot",
) -> plt.Axes:
    """Plot calibration curve showing predicted probabilities vs actual outcomes.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins for grouping predictions
        ax: Matplotlib axes object (optional, creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create bins and calculate observed vs predicted rates
    calib_df = pd.DataFrame({"y_true": y_true, "y_prob": y_proba})
    calib_df["bin"] = pd.cut(calib_df["y_prob"], bins=np.arange(0, 1.1, 1.0 / n_bins), right=False)

    calib_grouped = (
        calib_df.groupby("bin", observed=False)
        .agg(
            observed_rate=("y_true", "mean"),
            predicted_prob=("y_prob", "mean"),
            count=("y_true", "count"),
        )
        .reset_index()
    )

    # Plot observed vs predicted
    ax.plot(
        calib_grouped["predicted_prob"],
        calib_grouped["observed_rate"],
        marker="o",
        label="Model Calibration",
        linewidth=2.5,
        markersize=10,
        color="darkorange",
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", c="gray", linewidth=1, label="Perfectly Calibrated", alpha=0.7)

    # Add bin sizes as text annotations
    for _, row in calib_grouped.iterrows():
        if not pd.isna(row["predicted_prob"]):
            ax.annotate(
                f'n={int(row["count"])}',
                (row["predicted_prob"], row["observed_rate"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Default Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return ax


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Importance",
) -> plt.Axes:
    """Plot horizontal bar chart of feature importance.

    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        ax: Matplotlib axes object (optional, creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    top_features = feature_importance.head(top_n).copy()

    # Create color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features)))

    ax.barh(top_features["feature"], top_features["importance"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (feat, imp) in enumerate(zip(top_features["feature"], top_features["importance"])):
        ax.text(imp, i, f" {imp:.3f}", va="center", fontsize=9)

    return ax


def plot_confusion_matrix(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
    labels: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Confusion Matrix",
) -> plt.Axes:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        labels: Class labels for display (default: ["Non-Default", "Default"])
        ax: Matplotlib axes object (optional, creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if labels is None:
        labels = ["Non-Default", "Default"]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            total = cm.sum()
            percentage = cm[i, j] / total * 100
            ax.text(
                j,
                i + 0.15,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )

    return ax


def plot_threshold_analysis(
    y_true: npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    fp_cost: float = 100.0,
    fn_cost: float = 5000.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Business Cost vs Threshold",
) -> plt.Axes:
    """Plot business cost analysis across different classification thresholds.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative
        ax: Matplotlib axes object (optional, creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = np.arange(0.01, 0.99, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = (fp * fp_cost) + (fn * fn_cost)
        results.append({"threshold": thresh, "cost": total_cost, "fp": fp, "fn": fn})

    cost_df = pd.DataFrame(results)

    # Plot total cost
    ax.plot(cost_df["threshold"], cost_df["cost"], linewidth=2.5, color="darkred", label="Total Cost")

    # Mark optimal threshold
    optimal_idx = cost_df["cost"].idxmin()
    optimal_thresh = cost_df.loc[optimal_idx, "threshold"]
    optimal_cost = cost_df.loc[optimal_idx, "cost"]

    ax.axvline(optimal_thresh, linestyle="--", color="green", linewidth=2, label=f"Optimal: {optimal_thresh:.3f}")
    ax.scatter([optimal_thresh], [optimal_cost], color="green", s=200, zorder=5, marker="*")

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Total Business Cost ($)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)

    # Add annotation for optimal point
    ax.annotate(
        f"Min Cost: ${optimal_cost:,.0f}",
        xy=(optimal_thresh, optimal_cost),
        xytext=(optimal_thresh + 0.15, optimal_cost * 1.1),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
    )

    return ax


def plot_comprehensive_diagnostics(
    y_true: npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.int64],
    feature_importance: pd.DataFrame,
    auc: float,
    fp_cost: float = 100.0,
    fn_cost: float = 5000.0,
    top_n_features: int = 15,
    figsize: tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive 2x3 diagnostic dashboard.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        y_pred: Predicted binary labels (at 0.5 threshold)
        feature_importance: DataFrame with 'feature' and 'importance' columns
        auc: ROC AUC score
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative
        top_n_features: Number of top features to show
        figsize: Figure size (width, height)
        save_path: Optional path to save figure (e.g., 'diagnostics.png')

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_comprehensive_diagnostics(
        ...     y_test, y_proba, y_pred, importance_df, auc=0.85,
        ...     save_path='model_diagnostics.png'
        ... )
        >>> plt.show()
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Model Performance Dashboard - Credit Risk Prediction", fontsize=18, fontweight="bold", y=0.98)

    # 1. ROC Curve
    plot_roc_curve(y_true, y_proba, auc, ax=axes[0, 0])

    # 2. Precision-Recall Curve
    plot_precision_recall_curve(y_true, y_proba, ax=axes[0, 1])

    # 3. Calibration Plot
    plot_calibration_curve(y_true, y_proba, n_bins=10, ax=axes[0, 2])

    # 4. Feature Importance
    plot_feature_importance(feature_importance, top_n=top_n_features, ax=axes[1, 0])

    # 5. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, ax=axes[1, 1])

    # 6. Threshold Analysis
    plot_threshold_analysis(y_true, y_proba, fp_cost, fn_cost, ax=axes[1, 2])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Dashboard saved to: {save_path}")

    return fig
