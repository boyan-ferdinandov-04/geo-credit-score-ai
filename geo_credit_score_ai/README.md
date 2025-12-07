# Geo Credit Score AI - Fraud Detection System

A machine learning system for credit risk prediction using geospatial features and advanced calibration techniques. Built as part of the Applied Artificial Intelligence course at TU-Sofia.

## Overview

This project implements a sophisticated credit default prediction model that combines traditional financial features with geospatial data (proximity to bank locations) to improve risk assessment accuracy. The system uses LightGBM with monotonic constraints and isotonic calibration to ensure reliable probability estimates.

## Key Features

- **Geospatial Feature Engineering**: Incorporates distance-based features from client locations to bank branches
- **Monotonic Constraints**: Enforces business logic that distance from banks increases default risk
- **Advanced Calibration**: Isotonic regression for well-calibrated probability predictions
- **Business Cost Optimization**: Threshold analysis to minimize total business costs (FP/FN cost asymmetry)
- **Class Imbalance Handling**: SMOTE oversampling and class weighting strategies
- **Type-Safe Configuration**: Pydantic-based configuration management with YAML support
- **Comprehensive Testing**: 95% code coverage with 94 tests ensuring model quality (AUC > 0.7)
- **Reproducible Pipeline**: End-to-end sklearn pipeline with preprocessing, resampling, and modeling

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Generation                          │
│  • Synthetic dataset (5000 samples, 12 features, 90/10 split)   │
│  • Intentional missing values (5%)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   Feature Engineering                           │
│  • Financial ratios (loan_to_income, balance_to_income)         │
│  • Risk indicators (payment_stress, delinq_utilization)         │
│  • Transformations (income_log, utilization_squared)            │
│  • Missing value indicators                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Geospatial Features                            │
│  • Generate 15 random bank locations                            │
│  • Calculate min/mean/std distances to nearest banks            │
│  • Create distance_balance_risk composite feature               │
│  • Optional: Inject label signal for monotonic constraint test  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   ML Pipeline                                   │
│  1. SimpleImputer (median strategy)                             │
│  2. StandardScaler                                              │
│  3. SMOTE (optional, k_neighbors=3)                             │
│  4. LightGBM (max_depth=4, n_estimators=150)                    │
│     - Monotonic constraints on distance features                │
│     - Class weight balancing                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Calibration                                  │
│  • CalibratedClassifierCV (isotonic method, cv=3)               │
│  • Wraps entire pipeline for proper feature transformation      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     Evaluation                                  │
│  • ROC AUC Score                                                │
│  • Brier Score (calibration metric)                             │
│  • Gini Coefficient                                             │
│  • Classification Report                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Business Cost Analysis                             │
│  • Threshold sweep (0.01 to 0.99)                               │
│  • Cost calculation: FP × $100 + FN × $2,500                    │
│  • Optimal threshold identification                             │
│  • Comprehensive 6-panel diagnostic dashboard                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd geo_credit_score_ai
```

2. **Install dependencies**

```bash
uv sync
```

This will install all required packages:

- `lightgbm` - Gradient boosting framework
- `scikit-learn` - ML utilities and pipelines
- `imbalanced-learn` - SMOTE for class imbalance
- `pandas` & `numpy` - Data manipulation
- `pydantic` - Configuration validation
- `matplotlib` - Visualization (for future dashboards)

3. **Install development dependencies** (optional)

```bash
uv sync --dev
```

Includes:

- `pytest` & `pytest-cov` - Testing framework
- `ruff` - Linting and formatting

## Usage

### Basic Training

Run the complete training pipeline:

```bash
uv run python main.py
```

**Expected Output:**

```
__main__ | 2025-11-05 19:59:04 | INFO     | Loading config
__main__ | 2025-11-05 19:59:04 | INFO     | Dataset: 5000 samples, target distribution: {0: 4479, 1: 521}
__main__ | 2025-11-05 19:59:04 | INFO     | Injected label signal: flipped 230 labels in far distance group
__main__ | 2025-11-05 19:59:04 | INFO     | Applied monotonic constraints to 4 features: ['min_bank_distance', 'mean_bank_distance', 'std_bank_distance', 'distance_balance_risk']
__main__ | 2025-11-05 19:59:04 | INFO     | Cross-validation starting
__main__ | 2025-11-05 19:59:05 | INFO     | Mean CV AUC: 0.857165 ± 0.013643
__main__ | 2025-11-05 19:59:05 | INFO     | Applying isotonic calibration with CV=3
__main__ | 2025-11-05 19:59:06 | INFO     | ROC AUC: 0.8436, Brier: 0.0919, Gini: 0.6872
__main__ | 2025-11-05 19:59:06 | INFO     | Pipeline complete
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=. --cov-report=html

# Open coverage report in browser
open htmlcov/index.html
```

**Current Test Coverage:** 95%

### Generating Visualizations

Generate comprehensive diagnostic dashboards:

```bash
uv run python visualize_model.py
```

This creates a 2x3 dashboard with:

- **ROC Curve** - Model discrimination ability
- **Precision-Recall Curve** - Performance on imbalanced data
- **Calibration Plot** - Reliability of probability estimates
- **Feature Importance** - Top contributing features
- **Confusion Matrix** - Classification performance breakdown
- **Business Cost Analysis** - Optimal threshold for cost minimization

**Output:** `model_diagnostics.png` (saved to project directory)

### SMOTE Analysis

Compare model performance with and without SMOTE oversampling:

```bash
uv run python analyze_smote.py
```

This comprehensive analysis trains two models and compares them across multiple dimensions:

**What it Does:**
- Trains one model **without SMOTE** (using only class weights)
- Trains one model **with SMOTE** oversampling to balance minority class
- Compares both models using:
  - Cross-validation AUC scores
  - Test set metrics (AUC, Brier score, Gini coefficient)
  - Precision, Recall, and F1 scores (at both 0.5 and optimal thresholds)
  - Business cost analysis with configurable FP/FN costs
  - Visual comparisons (ROC curves, PR curves, cost curves)

**Output:**
- Detailed comparison table showing metric differences
- Interpretation of results with actionable recommendations
- `smote_comparison.png` - 2×3 visualization dashboard comparing both approaches

**Example Output:**

```
PERFORMANCE COMPARISON TABLE
================================================================================
                    Metric   Without SMOTE      With SMOTE Difference
Cross-Val AUC (mean ± std) 0.8572 ± 0.0136 0.8384 ± 0.0162
              Test ROC AUC          0.8436          0.8273   0.0163 ↓
               Brier Score          0.0919          0.0988   0.0069 ↑
          Gini Coefficient          0.6872          0.6547   0.0325 ↓
    Precision (thresh=0.5)          0.8444          0.6452   0.1992 ↓
       Recall (thresh=0.5)          0.2533          0.2667   0.0134 ↑
     F1 Score (thresh=0.5)          0.3897          0.3774   0.0123 ↓
Business Cost (thresh=0.5)        $280,700        $277,200   $3,500 ↓
         Optimal Threshold           0.050           0.040
   Business Cost (optimal)         $62,400         $64,600   $2,200 ↑

Recommendation:
✗ SMOTE does not provide clear benefits. NOT RECOMMENDED for this dataset.
```

**When to Use SMOTE:**

The script provides data-driven recommendations:
- ✓ **Use SMOTE** if it improves both AUC and business costs
- → **Consider SMOTE** if it significantly reduces costs despite lower AUC
- → **Consider SMOTE** if it improves AUC and cost is not the primary concern
- ✗ **Don't use SMOTE** if it worsens both metrics or increases costs without AUC improvement

**Note:** For this synthetic dataset with 90/10 class imbalance, SMOTE actually reduces performance because the class weighting and monotonic constraints already handle the imbalance effectively.

### Business Cost Analysis

The system includes a sophisticated business cost analysis that helps determine the optimal classification threshold by minimizing total business costs. This analysis recognizes that different types of prediction errors have different financial impacts.

**Cost Model:**

- **False Positive (FP) Cost**: $100 per occurrence (configurable in `model_config.yaml`)
  - Represents the cost of rejecting a good customer who would have repaid
  - Includes lost revenue, customer acquisition costs, and opportunity costs

- **False Negative (FN) Cost**: $2,500 per occurrence (configurable in `model_config.yaml`)
  - Represents the cost of approving a customer who will default
  - Includes loan loss, collection costs, and administrative overhead

**How It Works:**

1. **Threshold Sweep**: The analysis evaluates classification thresholds from 0.01 to 0.99 in 0.01 increments

2. **Cost Calculation**: For each threshold, calculates:

   ```python
   total_cost = (false_positives × fp_cost) + (false_negatives × fn_cost)
   ```

3. **Optimal Threshold**: Identifies the threshold that minimizes total business cost
   - This is typically NOT 0.5, as asymmetric costs favor different decision boundaries
   - The optimal threshold balances the trade-off between FP and FN costs

4. **Visualization**: The "Business Cost vs Threshold" plot shows:
   - Total cost curve across all thresholds
   - Optimal threshold marked with a vertical line and star marker
   - Minimum cost annotation with exact values

**Why This Matters:**

Traditional ML metrics (accuracy, F1-score) assume equal cost for all errors. In credit risk:

- Missing a default (FN) costs **25× more** than rejecting a good customer (FP)
- The optimal business decision threshold reflects these real-world economics
- Using the cost-optimized threshold can save thousands of dollars compared to the default 0.5 threshold

**Configuring Costs:**

Modify the costs in `config/model_config.yaml` to match your business scenario:

```yaml
model:
  fp_cost: 100.0    # Cost of false positive (rejecting good customer)
  fn_cost: 2500.0   # Cost of false negative (approving defaulter)
```

**Example Dashboard:**

The visualization module (`src/visualization.py`) provides individual plotting functions and a comprehensive dashboard generator. You can also use individual functions:

```python
from src.visualization import plot_roc_curve, plot_calibration_curve

# Plot individual diagrams
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_roc_curve(y_test, y_proba, auc=0.84, ax=ax[0])
plot_calibration_curve(y_test, y_proba, ax=ax[1])
plt.show()
```

### Configuration

Modify `config/model_config.yaml` to customize the pipeline:

```yaml
random_state: 42

dataset:
  n_samples: 5000
  n_features: 12
  weights: [0.9, 0.1]  # Class imbalance ratio

model:
  test_size: 0.2
  cv_folds: 5
  use_smote: false
  lgbm_params:
    n_estimators: 150
    max_depth: 4
    learning_rate: 0.05

geo:
  n_banks: 15
  enforce_monotonic: true
  inject_label_signal: true  # For testing monotonic constraints

calibration:
  enabled: true
  method: isotonic  # or 'sigmoid'
  cv: 3
```

## Project Structure

```
geo_credit_score_ai/
├── main.py                     # Main training pipeline
├── visualize_model.py          # Generate diagnostic dashboards
├── analyze_smote.py            # SMOTE vs No SMOTE comparison analysis
├── config/
│   └── model_config.yaml       # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic config models
│   └── visualization.py        # Plotting and diagnostic visualizations
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_config.py          # Config validation tests
│   ├── test_data.py            # Dataset & feature engineering tests
│   ├── test_geo.py             # Geospatial features tests
│   └── test_pipeline.py        # Pipeline & evaluation tests
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependencies
├── model_diagnostics.png       # Generated visualization dashboard
├── smote_comparison.png        # SMOTE comparison visualization
└── README.md                   # This file
```

## Technical Details

### Dataset

- **Source**: Synthetic data using `sklearn.make_classification`
- **Size**: 5,000 samples
- **Features**: 12 original + 10 derived + 4 geospatial + 3 missing indicators
- **Target**: Binary classification (default=1, non-default=0)
- **Class Distribution**: 90% / 10% (imbalanced)

### Feature Engineering

**Financial Features:**

- `income_log`: Log-transformed income
- `loan_to_income`: Loan amount / income ratio
- `balance_to_income`: Monthly balance / income ratio
- `payment_stress`: Weighted combination of utilization and delinquencies
- `delinq_utilization`: Interaction between delinquencies and credit utilization
- `credit_age_ratio`: Credit history / age ratio
- `delinq_per_creditline`: Delinquencies per credit line
- `utilization_squared`: Non-linear utilization effect
- `credit_utilization_interaction`: Credit history × utilization
- `longterm_loan_flag`: Binary flag for loans > 75th percentile

**Geospatial Features:**

- `client_x`, `client_y`: Client coordinates
- `min_bank_distance`: Distance to nearest bank
- `mean_bank_distance`: Average distance to 15 nearest banks
- `std_bank_distance`: Standard deviation of distances
- `distance_balance_risk`: min_distance / (balance + 1)

**Missing Value Indicators:**

- `{feature}_missing`: Binary flags for originally missing values

### Model Configuration

**LightGBM Parameters:**

- `n_estimators`: 150 trees
- `max_depth`: 4 levels
- `learning_rate`: 0.05
- `num_leaves`: 31
- `subsample`: 0.8
- `reg_alpha` / `reg_lambda`: 0.1 (L1/L2 regularization)
- `monotone_constraints`: Enforced on distance features (+1 direction)

**Monotonic Constraints:**
The model enforces that increased distance from banks monotonically increases default probability for:

- `min_bank_distance`
- `mean_bank_distance`
- `std_bank_distance`
- `distance_balance_risk`

### Model Performance

**Target Metrics:**

- ROC AUC: > 0.84
- Brier Score: < 0.10
- Test Coverage: > 75% (currently 95%)
- All test models: AUC > 0.7

### Business Cost Analysis Implementation

The business cost analysis is implemented in `src/visualization.py:plot_threshold_analysis()` (lines 267-331). Key implementation details:

**Algorithm:**

```python
# For each threshold from 0.01 to 0.99 (step 0.01):
for thresh in np.arange(0.01, 0.99, 0.01):
    y_pred = (y_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * fp_cost) + (fn * fn_cost)
```

**Key Features:**

- **Comprehensive Sweep**: Evaluates 98 different thresholds to find the global minimum
- **Confusion Matrix Analysis**: Extracts all four metrics (TP, TN, FP, FN) for cost calculation
- **Visual Feedback**: Plots total cost curve with optimal point highlighted
- **Flexible Costs**: Accepts custom `fp_cost` and `fn_cost` parameters from configuration

**Integration Points:**

- Called by `visualize_model.py` during dashboard generation
- Uses costs from `config.model.fp_cost` and `config.model.fn_cost`
- Part of the 6-panel comprehensive diagnostics dashboard
- Results stored in `model_diagnostics.png`

**Real-World Application:**
The optimal threshold found by this analysis should be used in production for classification decisions, replacing the default 0.5 threshold. This can be implemented as:

```python
# Instead of:
y_pred = (y_proba >= 0.5).astype(int)

# Use the cost-optimized threshold:
optimal_threshold = 0.32  # Example from cost analysis
y_pred = (y_proba >= optimal_threshold).astype(int)
```

## Development

### Code Quality

**Linting & Formatting:**

```bash
uv run ruff check .
uv run ruff format .
```

**Type Checking:**
The project uses type hints throughout. Configuration uses Pydantic for runtime type validation.

### Adding Tests

Tests are organized by component:

1. `test_config.py` - Configuration validation
2. `test_data.py` - Data generation and feature engineering
3. `test_geo.py` - Geospatial feature logic
4. `test_pipeline.py` - Model pipeline and evaluation

All new features should include corresponding tests to maintain 95% coverage.

### Contributing

This is an academic project. For the course assignment:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass: `uv run pytest tests/`
5. Submit pull request

## Future Enhancements

- [x] Add visualization dashboards (ROC curves, calibration plots, feature importance) ✅
- [ ] Implement experiment tracking (MLflow or Weights & Biases)
- [ ] Create CLI interface for easy model usage
- [ ] Add real-world data ingestion pipeline
- [ ] Implement model monitoring and drift detection
- [ ] Deploy as REST API with FastAPI

## Academic Context

**Course**: Applied Artificial Intelligence
**Institution**: Technical University of Sofia (TU-Sofia)
**Focus Areas**:

- Machine Learning for Credit Risk
- Geospatial Feature Engineering
- Model Calibration Techniques
- Production ML Best Practices

## License

Academic use only - TU-Sofia Applied AI Course

## Contact

For questions related to this project, please contact through the course portal.

---

**Note**: This project uses synthetic data for educational purposes. For production credit scoring systems, ensure compliance with fair lending regulations and data privacy laws.
