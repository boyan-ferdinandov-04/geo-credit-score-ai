"""Analyze SMOTE k_neighbors parameter suitability."""

from src.config import Config
from src.data import create_dataset

# Load config and create dataset
cfg = Config.from_yaml('config/model_config.yaml')
df = create_dataset(cfg)

# Analyze class distribution
total_samples = len(df)
class_dist = df[cfg.features.target_col].value_counts()
minority_count = class_dist.min()
majority_count = class_dist.max()

# Estimate train split
train_size = int(total_samples * cfg.model.test_size)
minority_in_train = int(minority_count * (1 - cfg.model.test_size))

print("=" * 60)
print("SMOTE k_neighbors Parameter Analysis")
print("=" * 60)
print(f"\nDataset Statistics:")
print(f"  Total samples: {total_samples:,}")
print(f"  Class distribution: {class_dist.to_dict()}")
print(f"  Minority class: {minority_count:,} samples ({minority_count/total_samples*100:.1f}%)")
print(f"  Majority class: {majority_count:,} samples ({majority_count/total_samples*100:.1f}%)")
print(f"  Imbalance ratio: 1:{majority_count/minority_count:.1f}")

print(f"\nTrain/Test Split (test_size={cfg.model.test_size}):")
print(f"  Train samples: ~{int(total_samples * (1 - cfg.model.test_size)):,}")
print(f"  Minority in train: ~{minority_in_train:,}")

print(f"\nSMOTE k_neighbors Analysis:")
current_k = 3
print(f"  Current k_neighbors: {current_k}")
print(f"  Minimum required: {1}")
print(f"  Maximum safe: {minority_in_train - 1}")

# Recommended values
recommended_k = min(5, minority_in_train - 1)
print(f"\n  Recommendation: k_neighbors = {recommended_k}")
print(f"    - Standard SMOTE default is 5")
print(f"    - With {minority_in_train} minority samples in train, k={recommended_k} is safe")
print(f"    - k=3 is {'TOO CONSERVATIVE' if minority_in_train > 10 else 'APPROPRIATE'}")

if current_k < recommended_k:
    print(f"\n  ⚠️  Current k={current_k} may be overfitting to very local patterns")
    print(f"  ✓  Increasing to k={recommended_k} will create more diverse synthetic samples")
else:
    print(f"\n  ✓  Current k={current_k} is appropriate for the dataset size")

print("\n" + "=" * 60)
