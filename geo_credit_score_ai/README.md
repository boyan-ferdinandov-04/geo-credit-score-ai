# Geo Credit Score AI - Fraud Detection System

A machine learning system for credit risk prediction using geospatial features and advanced calibration techniques. Built as part of the Applied Artificial Intelligence course at TU-Sofia.

## Overview

This project implements a sophisticated credit default prediction model that combines traditional financial features with geospatial data (proximity to bank locations) to improve risk assessment accuracy. The system uses LightGBM with monotonic constraints and isotonic calibration to ensure reliable probability estimates.

## Key Features

- **Geospatial Feature Engineering**: Incorporates distance-based features from client locations to bank branches
- **Monotonic Constraints**: Enforces business logic that distance from banks increases default risk
- **Advanced Calibration**: Isotonic regression for well-calibrated probability predictions
- **Class Imbalance Handling**: SMOTE oversampling and class weighting strategies
- **Type-Safe Configuration**: Pydantic-based configuration management with YAML support
- **Comprehensive Testing**: 95% code coverage with 94 tests ensuring model quality (AUC > 0.7)
- **Reproducible Pipeline**: End-to-end sklearn pipeline with preprocessing, resampling, and modeling

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Generation                           │
│  • Synthetic dataset (5000 samples, 12 features, 90/10 split)   │
│  • Intentional missing values (5%)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   Feature Engineering                            │
│  • Financial ratios (loan_to_income, balance_to_income)         │
│  • Risk indicators (payment_stress, delinq_utilization)         │
│  • Transformations (income_log, utilization_squared)            │
│  • Missing value indicators                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Geospatial Features                             │
│  • Generate 15 random bank locations                            │
│  • Calculate min/mean/std distances to nearest banks            │
│  • Create distance_balance_risk composite feature              │
│  • Optional: Inject label signal for monotonic constraint test │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   ML Pipeline                                    │
│  1. SimpleImputer (median strategy)                             │
│  2. StandardScaler                                              │
│  3. SMOTE (optional, k_neighbors=3)                             │
│  4. LightGBM (max_depth=4, n_estimators=150)                    │
│     - Monotonic constraints on distance features                │
│     - Class weight balancing                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Calibration                                   │
│  • CalibratedClassifierCV (isotonic method, cv=3)               │
│  • Wraps entire pipeline for proper feature transformation      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     Evaluation                                   │
│  • ROC AUC Score                                                │
│  • Brier Score (calibration metric)                            │
│  • Gini Coefficient                                             │
│  • Classification Report                                        │
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
