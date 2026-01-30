"""
Custom loss functions for deepfake detection.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0, **kwargs):
        """
        Initialize FocalLoss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            label_smoothing: Label smoothing factor
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.shape(y_pred)[-1]
            smooth_positives = 1.0 - self.label_smoothing
            smooth_negatives = self.label_smoothing / num_classes
            y_true = y_true * smooth_positives + smooth_negatives
        
        # Compute cross-entropy
        ce = -y_true * tf.math.log(y_pred + 1e-8)
        
        # Compute p_t
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Compute focal weight
        focal_weight = self.alpha * tf.pow(1.0 - p_t, self.gamma)
        
        # Compute focal loss
        focal_loss = focal_weight * tf.reduce_sum(ce, axis=-1)
        
        return tf.reduce_mean(focal_loss)








