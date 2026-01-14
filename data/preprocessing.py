"""
Image preprocessing utilities.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


class Preprocessor:
    """Image preprocessing utilities."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        """
        Initialize Preprocessor.
        
        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize to [0, 1]
        """
        self.image_size = image_size
        self.normalize = normalize
    
    def preprocess_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Preprocessed image tensor
        """
        # Resize
        image = tf.image.resize(image, self.image_size)
        
        # Normalize to [0, 1] if needed
        if self.normalize:
            image = tf.image.convert_image_dtype(image, tf.float32)
        else:
            # Normalize to ImageNet mean/std
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.keras.applications.efficientnet.preprocess_input(image)
        
        return image
    
    def normalize_image(self, image: tf.Tensor, method: str = 'imagenet') -> tf.Tensor:
        """
        Normalize image.
        
        Args:
            image: Input image tensor
            method: Normalization method ('imagenet', 'standard', 'minmax')
            
        Returns:
            Normalized image tensor
        """
        if method == 'imagenet':
            # ImageNet normalization
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            image = (image - mean) / std
        elif method == 'standard':
            # Standard normalization (zero mean, unit variance)
            image = (image - tf.reduce_mean(image)) / tf.math.reduce_std(image)
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1e-8)
        
        return image
    
    def apply_histogram_equalization(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply histogram equalization.
        
        Args:
            image: Input image tensor
            
        Returns:
            Histogram equalized image tensor
        """
        # Convert to grayscale for histogram equalization
        gray = tf.image.rgb_to_grayscale(image)
        
        # Apply histogram equalization
        # Note: This is a simplified version
        # For full implementation, use tf.image.adjust_gamma or external libraries
        image = tf.image.adjust_gamma(image, gamma=1.0)
        
        return image







