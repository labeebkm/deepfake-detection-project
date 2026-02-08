"""
Custom loss functions for deepfake detection.
"""

import tensorflow as tf
from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    """
    Focal Loss with optional label smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        name: str = "focal_loss"
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.

        Args:
            y_true: Integer class labels, shape [batch]
            y_pred: Softmax probabilities, shape [batch, num_classes]

        Returns:
            Scalar focal loss
        """

        # Ensure correct dtypes
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Number of classes
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)

        # One-hot encode labels
        y_true_one_hot = tf.one_hot(
            y_true,
            depth=tf.cast(num_classes, tf.int32)
        )

        # --------------------------------------------------
        # Label smoothing (FIXED & SAFE)
        # --------------------------------------------------
        if self.label_smoothing > 0.0:
            smooth_positives = 1.0 - self.label_smoothing
            smooth_negatives = self.label_smoothing / num_classes
            y_true_one_hot = (
                y_true_one_hot * smooth_positives + smooth_negatives
            )

        # Numerical stability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Cross-entropy
        ce = -y_true_one_hot * tf.math.log(y_pred)

        # p_t (probability of the true class)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

        # Focal weighting
        focal_weight = self.alpha * tf.pow(1.0 - p_t, self.gamma)

        # Final focal loss
        loss = focal_weight * tf.reduce_sum(ce, axis=-1)

        return tf.reduce_mean(loss)

