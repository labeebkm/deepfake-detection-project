# EDA Usage Examples

Complete examples for using all EDA features.

## Example 1: Complete EDA Workflow

```python
import numpy as np
import cv2
from pathlib import Path
from eda.report_generator import generate_eda_report
from eda.data_analyzer import DataQualityChecker
from eda.visualization import EDAVisualizer
from eda.statistical_tests import compare_distributions
from eda.difficulty_estimator import DataDifficultyEstimator, FeatureImportanceAnalyzer

# 1. Generate automated report
report_path = generate_eda_report(
    dataset_path="./data/raw",
    output_format='html',
    output_dir='./reports',
    dataset_name='faceforensics'
)
print(f"Report generated: {report_path}")

# 2. Quality assessment
checker = DataQualityChecker()
quality = checker.assess_dataset("./data/raw")
print(f"Quality Score: {quality['overall_score']:.2f}")
print(f"Quality Level: {quality['quality_level']}")

# 3. Load sample images
real_images = []
fake_images = []
# ... load your images ...

# 4. Statistical comparison
p_value = compare_distributions(
    real_images[:50],
    fake_images[:50],
    test_type='all'
)
print(f"Statistical test results: {p_value}")

# 5. Visualizations
visualizer = EDAVisualizer()
visualizer.plot_frequency_comparison(
    real_images[0],
    fake_images[0],
    save_path='./reports/frequency_comparison.png'
)

# 6. Difficulty estimation
estimator = DataDifficultyEstimator()
difficulty = estimator.estimate_difficulty(real_images, fake_images)
print(f"Difficulty: {difficulty['difficulty_level']} ({difficulty['difficulty_score']:.2f})")

# 7. Feature importance
importance_analyzer = FeatureImportanceAnalyzer()
importance = importance_analyzer.analyze_feature_importance(real_images, fake_images)
print(f"Most important feature: {importance['most_important']}")
```

## Example 2: Quick Quality Check

```python
from eda.data_analyzer import DataQualityChecker

checker = DataQualityChecker()
quality = checker.assess_dataset("./data/raw")

if quality['quality_level'] == 'Poor':
    print("⚠️ Dataset quality is poor. Recommendations:")
    for rec in quality['recommendations']:
        print(f"  - {rec}")
```

## Example 3: Statistical Testing

```python
from eda.statistical_tests import compare_distributions

# Quick p-value check
p_value = compare_distributions(real_images, fake_images, test_type='ks')
if p_value < 0.05:
    print("✅ Distributions are significantly different")
else:
    print("❌ No significant difference detected")

# Detailed analysis
all_results = compare_distributions(real_images, fake_images, test_type='all')
for test_name, result in all_results.items():
    print(f"{test_name}: p={result['p_value']:.4f}, significant={result['significant']}")
```

## Example 4: Feature Importance for Model Design

```python
from eda.difficulty_estimator import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
importance = analyzer.analyze_feature_importance(real_images, fake_images)

print("Feature Importance Ranking:")
for feature, score in importance['sorted_features']:
    print(f"  {feature}: {score:.3f}")

# Use recommendations for model architecture
for rec in importance['recommendations']:
    print(f"  → {rec}")
```

## Example 5: Difficulty-Based Training Strategy

```python
from eda.difficulty_estimator import DataDifficultyEstimator

estimator = DataDifficultyEstimator()
difficulty = estimator.estimate_difficulty(real_images, fake_images)

if difficulty['difficulty_level'] == 'Easy':
    print("Use standard EfficientNet architecture")
elif difficulty['difficulty_level'] == 'Medium':
    print("Use dual-stream architecture with augmentation")
else:
    print("Use advanced architecture with frequency branch and attention fusion")
    print("Consider self-supervised learning")
```

## Example 6: Command Line Usage

```bash
# Generate full EDA report
python eda_report.py \
    --dataset faceforensics \
    --data_dir ./data/raw \
    --output ./reports \
    --visualize \
    --save

# Generate report for different dataset
python eda_report.py \
    --dataset celebdf \
    --data_dir ./data/celebdf \
    --output ./reports/celebdf
```

## Example 7: Integration with Training Pipeline

```python
# In train.py or training script
from eda.data_analyzer import DataQualityChecker
from eda.difficulty_estimator import DataDifficultyEstimator
import yaml

# Load config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Assess dataset before training
checker = DataQualityChecker()
quality = checker.assess_dataset(config['dataset']['root_dir'])

# Adjust training based on quality
if quality['overall_score'] < 0.5:
    config['training']['epochs'] = 150  # Train longer for poor quality data
    config['augmentation']['enabled'] = True  # Enable augmentation

# Estimate difficulty
# ... load images ...
estimator = DataDifficultyEstimator()
difficulty = estimator.estimate_difficulty(real_images, fake_images)

# Adjust model architecture based on difficulty
if difficulty['difficulty_level'] == 'Hard':
    config['model']['frequency_branch']['enabled'] = True
    config['model']['attention_fusion']['enabled'] = True
```

## Example 8: Batch Analysis

```python
from eda.report_generator import ReportGenerator

# Analyze multiple datasets
datasets = ['faceforensics', 'celebdf', 'dfdc']
results = []

for dataset in datasets:
    report_path = generate_eda_report(
        dataset_path=f"./data/{dataset}",
        dataset_name=dataset,
        output_dir=f"./reports/{dataset}"
    )
    results.append(report_path)

# Generate comparison report
report_gen = ReportGenerator()
comparison_report = report_gen.generate_comparison_report(
    datasets=[...],  # Analysis results
    dataset_names=datasets
)
```








