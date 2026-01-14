"""
Custom metrics for deepfake detection.
"""

import tensorflow as tf
from tensorflow import keras
from typing import List


def get_metrics(metric_names: List[str]) -> List:
    """
    Get list of metrics based on names.
    
    Args:
        metric_names: List of metric names
        
    Returns:
        List of metric objects
    """
    metrics = []
    
    metric_map = {
        'accuracy': 'accuracy',
        'precision': keras.metrics.Precision(name='precision'),
        'recall': keras.metrics.Recall(name='recall'),
        'f1_score': F1Score(),
        'auc': keras.metrics.AUC(name='auc')
    }
    
    for name in metric_names:
        if name in metric_map:
            metrics.append(metric_map[name])
    
    return metrics


class F1Score(keras.metrics.Metric):
    """F1 Score metric."""
    
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-8))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()







