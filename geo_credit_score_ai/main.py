import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1) Create a synthetic dataset that loosely resembles credit data
X, y = make_classification(
    n_samples=5000,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    weights=[0.9, 0.1],  # imbalanced: 10% defaults
    flip_y=0.01,
    random_state=42,
)

# Create a DataFrame with plausible column names
cols: list[str] = [
    "age",
    "income",
    "loan_amount",
    "loan_term",
    "credit_history_len",
    "num_credit_lines",
    "delinquencies",
    "utilization",
    "employment_years",
    "num_open_accounts",
    "avg_monthly_balance",
    "feature_extra",
]

df = pd.DataFrame(X, columns=pd.Index(cols))
df["default"] = y

# Introduce missing values BEFORE feature engineering
rng = np.random.default_rng(42)
mask = rng.choice([False, True], size=(len(df), len(cols)), p=[0.95, 0.05])
df[cols] = df[cols].mask(mask)

# Feature engineering AFTER introducing missing values (more realistic)
df["income_log"] = np.log1p(np.abs(df["income"]))
df["delinq_utilization"] = df["delinquencies"] * df["utilization"]
df["credit_age_ratio"] = df["credit_history_len"] / (df["age"] + 1)

for col in ["income", "employment_years", "avg_monthly_balance"]:
    df[f"{col}_missing"] = df[col].isna().astype(int)

# Define feature columns INCLUDING engineered features
feature_cols = cols + [
    "income_log",
    "delinq_utilization",
    "credit_age_ratio",
    "income_missing",
    "employment_years_missing",
    "avg_monthly_balance_missing",
]

# 2) Train / test split
X = df[feature_cols]
y = df["default"]

print(f"Total features: {len(feature_cols)}")
print(f"Missing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3) Simple preprocessing + model pipeline
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                force_col_wise=True,
                feature_name="auto",
            ),
        ),
    ]
)

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print(f"\nCross-validation AUC: {cv_scores.mean():.6f} (± {cv_scores.std():.6f})")

pipeline.fit(X_train, y_train)

# 4) Predictions & evaluation
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Calculate metrics
auc = roc_auc_score(y_test, y_proba)
brier = brier_score_loss(y_test, y_proba)

# Get ROC curve data HERE, before using them
fpr, tpr, _ = roc_curve(y_test, y_proba)

print("\nModel Performance:")
print(f"ROC AUC: {auc:.4f}")
print(f"Brier score: {brier:.4f}")
print(f"Gini coefficient: {2 * auc - 1:.4f}")

# Feature importance for tree-based models
feature_importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": pipeline.named_steps["clf"].feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Calculate additional metrics
from sklearn.metrics import classification_report, precision_recall_curve

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# 5) Enhanced visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ROC Curve - now fpr and tpr are defined
axes[0, 0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[0, 0].plot([0, 1], [0, 1], "--", linewidth=0.6)
axes[0, 0].set_xlabel("False Positive Rate")
axes[0, 0].set_ylabel("True Positive Rate")
axes[0, 0].set_title("ROC Curve")
axes[0, 0].legend()

# Calibration plot
bins = pd.cut(y_proba, bins=10, labels=False)
if isinstance(y_test, pd.Series):
    y_test_values = y_test.values
else:
    y_test_values = np.array(y_test)

calib = pd.DataFrame({"y": y_test_values, "p": y_proba, "bin": bins})
calib_grouped = calib.groupby("bin").agg(obs=("y", "mean"), pred=("p", "mean"), count=("y", "count")).reset_index()

calib_filtered = calib_grouped[calib_grouped["count"] > 10]
axes[0, 1].plot(calib_filtered["pred"], calib_filtered["obs"], marker="o")
axes[0, 1].plot([0, 1], [0, 1], "--", linewidth=0.6)
axes[0, 1].set_xlabel("Predicted probability (bin mean)")
axes[0, 1].set_ylabel("Observed default rate")
axes[0, 1].set_title("Calibration Plot")

# Precision-Recall curve
axes[1, 0].plot(recall, precision, label=f"AP = {np.trapezoid(precision, recall):.3f}")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].set_title("Precision-Recall Curve")
axes[1, 0].legend()

# Feature importance plot
top_features = feature_importance.head(10)
axes[1, 1].barh(top_features["feature"], top_features["importance"])
axes[1, 1].set_xlabel("Feature Importance")
axes[1, 1].set_title("Top 10 Features")
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# Business metrics
print("\nBusiness Metrics:")
print(f"Default rate in test: {y_test.mean():.2%}")
print(f"Predicted default rate: {y_proba.mean():.2%}")

# Find optimal threshold for business rules
from sklearn.metrics import f1_score

thresholds = np.arange(0.05, 0.5, 0.01)
f1_scores = [f1_score(y_test, y_proba >= t) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold (by F1): {optimal_threshold:.3f}")
