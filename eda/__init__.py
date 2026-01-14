"""
EDA (Exploratory Data Analysis) module for deepfake detection.
"""

from .data_analyzer import DataAnalyzer, DataQualityChecker
from .visualization import EDAVisualizer
from .statistical_tests import StatisticalTester, compare_distributions
from .artifact_detector import ArtifactDetector
from .report_generator import ReportGenerator, generate_eda_report
from .difficulty_estimator import DataDifficultyEstimator, FeatureImportanceAnalyzer

__all__ = [
    "DataAnalyzer",
    "DataQualityChecker",
    "EDAVisualizer",
    "StatisticalTester",
    "compare_distributions",
    "ArtifactDetector",
    "ReportGenerator",
    "generate_eda_report",
    "DataDifficultyEstimator",
    "FeatureImportanceAnalyzer",
]

