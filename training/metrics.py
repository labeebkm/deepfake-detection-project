"""
Custom metrics for deepfake detection.
"""

from __future__ import annotations

from typing import List

import tensorflow as tf
from tensorflow import keras


def _to_binary_labels(y_true: tf.Tensor, *, pos_class_id: int) -> tf.Tensor:
    """Convert sparse (or one-hot) labels to binary labels for a positive class."""
    y_true = tf.convert_to_tensor(y_true)
    if y_true.shape.rank is not None and y_true.shape.rank > 1:
        y_true = tf.argmax(y_true, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    return tf.cast(tf.equal(y_true, int(pos_class_id)), tf.float32)


def _select_pos_probability(y_pred: tf.Tensor, *, pos_class_id: int) -> tf.Tensor:
    """Select the positive-class probability from model outputs.

    Supports:
    - binary probability: [batch] or [batch, 1]
    - softmax probabilities: [batch, num_classes]
    """
    y_pred = tf.convert_to_tensor(y_pred)
    if y_pred.shape.rank is not None and y_pred.shape.rank > 1:
        y_pred = y_pred[..., int(pos_class_id)]
    return tf.cast(y_pred, tf.float32)


class SparseBinaryPrecision(keras.metrics.Metric):
    """Precision for sparse labels + softmax outputs (positive class=1 by default)."""

    def __init__(
        self,
        *,
        pos_class_id: int = 1,
        threshold: float = 0.5,
        name: str = "precision",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.pos_class_id = int(pos_class_id)
        self.threshold = float(threshold)
        self._precision = keras.metrics.Precision(
            name=f"{name}_internal",
            thresholds=self.threshold,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = _to_binary_labels(y_true, pos_class_id=self.pos_class_id)
        y_pred_pos = _select_pos_probability(y_pred, pos_class_id=self.pos_class_id)
        self._precision.update_state(y_true_bin, y_pred_pos, sample_weight=sample_weight)

    def result(self):
        return self._precision.result()

    def reset_state(self):
        self._precision.reset_state()


class SparseBinaryRecall(keras.metrics.Metric):
    """Recall for sparse labels + softmax outputs (positive class=1 by default)."""

    def __init__(
        self,
        *,
        pos_class_id: int = 1,
        threshold: float = 0.5,
        name: str = "recall",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.pos_class_id = int(pos_class_id)
        self.threshold = float(threshold)
        self._recall = keras.metrics.Recall(
            name=f"{name}_internal",
            thresholds=self.threshold,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = _to_binary_labels(y_true, pos_class_id=self.pos_class_id)
        y_pred_pos = _select_pos_probability(y_pred, pos_class_id=self.pos_class_id)
        self._recall.update_state(y_true_bin, y_pred_pos, sample_weight=sample_weight)

    def result(self):
        return self._recall.result()

    def reset_state(self):
        self._recall.reset_state()


class SparseBinaryF1Score(keras.metrics.Metric):
    """F1 score for sparse labels + softmax outputs (positive class=1 by default)."""

    def __init__(
        self,
        *,
        pos_class_id: int = 1,
        threshold: float = 0.5,
        name: str = "f1_score",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.pos_class_id = int(pos_class_id)
        self.threshold = float(threshold)
        self._precision = keras.metrics.Precision(
            name=f"{name}_precision_internal",
            thresholds=self.threshold,
        )
        self._recall = keras.metrics.Recall(
            name=f"{name}_recall_internal",
            thresholds=self.threshold,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = _to_binary_labels(y_true, pos_class_id=self.pos_class_id)
        y_pred_pos = _select_pos_probability(y_pred, pos_class_id=self.pos_class_id)
        self._precision.update_state(
            y_true_bin, y_pred_pos, sample_weight=sample_weight
        )
        self._recall.update_state(y_true_bin, y_pred_pos, sample_weight=sample_weight)

    def result(self):
        p = self._precision.result()
        r = self._recall.result()
        return 2.0 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self._precision.reset_state()
        self._recall.reset_state()


class SparseBinaryAUC(keras.metrics.Metric):
    """ROC AUC for sparse labels + softmax outputs (positive class=1 by default)."""

    def __init__(
        self,
        *,
        pos_class_id: int = 1,
        name: str = "auc",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.pos_class_id = int(pos_class_id)
        self._auc = keras.metrics.AUC(name=f"{name}_internal")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = _to_binary_labels(y_true, pos_class_id=self.pos_class_id)
        y_pred_pos = _select_pos_probability(y_pred, pos_class_id=self.pos_class_id)
        self._auc.update_state(y_true_bin, y_pred_pos, sample_weight=sample_weight)

    def result(self):
        return self._auc.result()

    def reset_state(self):
        self._auc.reset_state()


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
        "accuracy": "accuracy",
        # These metrics must support sparse integer labels with softmax outputs [batch, 2].
        # We compute them for the "fake" class (class_id=1).
        "precision": SparseBinaryPrecision(pos_class_id=1, threshold=0.5, name="precision"),
        "recall": SparseBinaryRecall(pos_class_id=1, threshold=0.5, name="recall"),
        "f1_score": SparseBinaryF1Score(pos_class_id=1, threshold=0.5, name="f1_score"),
        "auc": SparseBinaryAUC(pos_class_id=1, name="auc"),
    }

    for name in metric_names:
        if name in metric_map:
            metrics.append(metric_map[name])

    return metrics

