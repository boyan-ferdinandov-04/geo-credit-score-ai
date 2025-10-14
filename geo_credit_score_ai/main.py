import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1) Create a synthetic dataset that loosely resembles credit data
# - binary target: default (1) / non-default (0)
# - mix of informative (financial) and noisy features
x, y = make_classification(
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

df = pd.DataFrame(x, columns=pd.Index(cols))
df["default"] = y

# Introduce some missing values (realistic) - ONLY in feature columns, NOT target
rng = np.random.default_rng(42)
# Create mask only for feature columns
mask = rng.choice([False, True], size=(len(df), len(cols)), p=[0.95, 0.05])
df[cols] = df[cols].mask(mask)  # Apply mask only to feature columns

# 2) Train / test split
x = df[cols]
y = df["default"]

print(f"Missing values in features: {x.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# 3) Simple preprocessing + model pipeline
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
    ]
)

pipeline.fit(x_train, y_train)

# 4) Predictions & evaluation
y_proba = pipeline.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
brier = brier_score_loss(y_test, y_proba)

print("\nModel Performance:")
print(f"ROC AUC: {auc:.4f}")
print(f"Brier score: {brier:.4f}")
print(f"Gini coefficient: {2 * auc - 1:.4f}")

# 5) ROC curve + simple calibration scatter
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", linewidth=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
# calibration: bin probabilities and plot observed vs predicted
bins = pd.cut(y_proba, bins=10, labels=False)

if isinstance(y_test, pd.Series):
    y_test_values = y_test.values
else:
    y_test_values = np.array(y_test)

calib = pd.DataFrame({"y": y_test_values, "p": y_proba, "bin": bins})
calib_grouped = calib.groupby("bin").agg(obs=("y", "mean"), pred=("p", "mean"), count=("y", "count")).reset_index()

# Only plot bins with sufficient samples
calib_filtered = calib_grouped[calib_grouped["count"] > 10]
plt.plot(calib_filtered["pred"], calib_filtered["obs"], marker="o")
plt.plot([0, 1], [0, 1], "--", linewidth=0.6)
plt.xlabel("Predicted probability (bin mean)")
plt.ylabel("Observed default rate")
plt.title("Calibration (binned)")

plt.tight_layout()
plt.show()
