"""
Example script demonstrating how to use the visualization module.

This script trains the fraud detection model and generates comprehensive
diagnostic visualizations.

Usage:
    uv run python visualize_model.py
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

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
from src.visualization import plot_comprehensive_diagnostics

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    """Train model and generate visualizations."""
    logger.info("="* 60)
    logger.info("Fraud Detection Model - Visualization Dashboard")
    logger.info("=" * 60)

    # 1. Load configuration
    logger.info("Loading configuration...")
    config = Config.from_yaml("config/model_config.yaml")

    # 2. Create dataset
    logger.info("Creating dataset...")
    df = create_dataset(config)
    logger.info(
        f"Dataset: {len(df)} samples, target distribution: {df[config.features.target_col].value_counts().to_dict()}"
    )

    # Validate raw dataset
    validate_dataset(df, config, stage="raw")

    # 3. Add geospatial features
    logger.info("Adding geospatial features...")
    banks_xy = create_bank_locations(config)
    df = add_geo_data(df, config, banks_xy)
    inject_distance_label_signal(df, config)

    # 4. Engineer features
    logger.info("Engineering features...")
    df, feature_cols = engineer_features(df, config)
    geo_cols = ["client_x", "client_y"] + config.geo.distance_features + ["distance_balance_risk"]
    feature_cols = list(dict.fromkeys(feature_cols + geo_cols))
    logger.info(f"Total features: {len(feature_cols)}")

    # 5. Build monotonic constraints
    constraints = build_monotone_constraints(feature_cols, config)

    # 6. Prepare model
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

    # Validate train/test split
    validate_train_test_split(X_train, X_test, y_train, y_test)

    # 8. Train model
    logger.info("Training model...")
    pipeline = build_pipeline(model_params, use_smote=config.model.use_smote)
    pipeline.fit(X_train, y_train)

    # 9. Evaluate model
    logger.info("Evaluating model...")
    eval_results = evaluate_model(pipeline, X_test, y_test, fp_cost=config.model.fp_cost, fn_cost=config.model.fn_cost)

    # 10. Get feature importance
    logger.info("Extracting feature importance...")
    lgbm_model = pipeline.named_steps["clf"]
    feature_importance_values = lgbm_model.feature_importances_

    # Map back to original feature names
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": feature_importance_values}).sort_values(
        "importance", ascending=False
    )

    logger.info(f"\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.3f}")

    # 11. Generate comprehensive visualizations
    logger.info("\nGenerating visualization dashboard...")
    fig = plot_comprehensive_diagnostics(
        y_true=y_test.values,
        y_proba=eval_results["y_proba"],
        y_pred=eval_results["y_pred"],
        feature_importance=importance_df,
        auc=eval_results["auc"],
        fp_cost=config.model.fp_cost,
        fn_cost=config.model.fn_cost,
        top_n_features=config.visualization.top_n_features,
        save_path="model_diagnostics.png",
    )

    logger.info("="*60)
    logger.info("Dashboard generated successfully!")
    logger.info("Saved to: model_diagnostics.png")
    logger.info("="*60)

    # Show plots
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
