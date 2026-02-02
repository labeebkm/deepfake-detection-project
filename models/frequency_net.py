# """
# Frequency domain analysis network using DCT.
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from typing import Optional
# import numpy as np


# class FrequencyNet(keras.Model):
#     """
#     Frequency domain analysis network using DCT.
#     """
    
#     def __init__(self, 
#                  dct_size: int = 8,
#                  num_filters: int = 64,
#                  **kwargs):
#         """
#         Initialize FrequencyNet.
        
#         Args:
#             dct_size: DCT block size
#             num_filters: Number of filters in convolutional layers
#         """
#         super().__init__(**kwargs)
        
#         self.dct_size = dct_size
#         self.num_filters = num_filters
        
#         # DCT branch
#         self.dct_conv = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
#         self.dct_bn = layers.BatchNormalization()
        
#         # Additional processing layers
#         self.process_conv = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
#         self.process_bn = layers.BatchNormalization()
    
#     def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
#         """
#         Forward pass.
        
#         Args:
#             inputs: Input image tensor
#             training: Whether in training mode
            
#         Returns:
#             Frequency features
#         """
#         # Convert to grayscale if needed
#         if inputs.shape[-1] == 3:
#             gray = tf.image.rgb_to_grayscale(inputs)
#         else:
#             gray = inputs
        
#         # DCT branch
#         dct_features = self._apply_dct(gray)
#         dct_features = self.dct_conv(dct_features)
#         dct_features = self.dct_bn(dct_features, training=training)
        
#         # Additional processing
#         processed = self.process_conv(dct_features)
#         processed = self.process_bn(processed, training=training)
        
#         return processed
    
#     def _apply_dct(self, image: tf.Tensor) -> tf.Tensor:
#         """Apply DCT transformation."""
#         # Apply 2D DCT using TensorFlow's DCT
#         # DCT is applied along the last two dimensions
#         batch_size = tf.shape(image)[0]
#         h, w = tf.shape(image)[1], tf.shape(image)[2]
        
#         # Reshape for DCT (remove channel dimension temporarily)
#         image_2d = tf.squeeze(image, axis=-1)  # [batch, h, w]
        
#         # Apply 2D DCT: first along rows, then along columns
#         # Note: tf.signal.dct may require axis=-1, so we transpose as needed
        
#         # 1. DCT along rows (axis 1)
#         # Transpose to [batch, w, h] so rows become the last axis
#         image_perm = tf.transpose(image_2d, perm=[0, 2, 1])
#         dct_rows_perm = tf.signal.dct(image_perm, type=2, norm='ortho')
#         # Transpose back to [batch, h, w]
#         dct_rows = tf.transpose(dct_rows_perm, perm=[0, 2, 1])
        
#         # 2. DCT along cols (axis 2)
#         # Cols are already the last axis in [batch, h, w]
#         dct_2d = tf.signal.dct(dct_rows, type=2, norm='ortho')
        
#         # Get magnitude (DCT is real, but we take absolute for consistency)
#         dct_magnitude = tf.abs(dct_2d)
        
#         # Add channel dimension back
#         dct_real = tf.expand_dims(dct_magnitude, axis=-1)  # [batch, h, w, 1]
        
#         return dct_real



"""
Frequency domain analysis network using DCT.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional


class FrequencyNet(keras.Model):
    """
    Frequency domain analysis network using DCT.
    """

    def __init__(
        self,
        dct_size: int = 8,
        num_filters: int = 64,
        **kwargs
    ):
        """
        Initialize FrequencyNet.

        Args:
            dct_size: DCT block size
            num_filters: Number of filters in convolutional layers
        """
        super().__init__(**kwargs)

        self.dct_size = dct_size
        self.num_filters = num_filters

        # DCT branch
        self.dct_conv = layers.Conv2D(
            num_filters, (3, 3), padding="same", activation="relu"
        )
        self.dct_bn = layers.BatchNormalization()

        # Additional processing layers
        self.process_conv = layers.Conv2D(
            num_filters, (3, 3), padding="same", activation="relu"
        )
        self.process_bn = layers.BatchNormalization()

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs: Input image tensor
            training: Whether in training mode

        Returns:
            Frequency features
        """

        # ------------------------------------------------------------
        # IMPORTANT FIX:
        # DCT does NOT support float16 → force float32 here
        # ------------------------------------------------------------
        inputs = tf.cast(inputs, tf.float32)

        # Convert to grayscale if needed
        if inputs.shape[-1] == 3:
            gray = tf.image.rgb_to_grayscale(inputs)
        else:
            gray = inputs

        # ------------------------------------------------------------
        # DCT branch
        # ------------------------------------------------------------
        dct_features = self._apply_dct(gray)
        dct_features = self.dct_conv(dct_features)
        dct_features = self.dct_bn(dct_features, training=training)

        # ------------------------------------------------------------
        # Additional processing
        # ------------------------------------------------------------
        processed = self.process_conv(dct_features)
        processed = self.process_bn(processed, training=training)

        # ------------------------------------------------------------
        # OPTIONAL: cast back to mixed precision for fusion
        # ------------------------------------------------------------
        if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
            processed = tf.cast(processed, tf.float16)

        return processed

    def _apply_dct(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply 2D DCT transformation.

        Args:
            image: Grayscale image tensor [batch, H, W, 1]

        Returns:
            DCT magnitude tensor [batch, H, W, 1]
        """

        # Remove channel dimension: [batch, H, W]
        image_2d = tf.squeeze(image, axis=-1)

        # ------------------------------------------------------------
        # 1. DCT along rows
        # ------------------------------------------------------------
        # Transpose so rows become last axis
        image_perm = tf.transpose(image_2d, perm=[0, 2, 1])
        dct_rows_perm = tf.signal.dct(
            image_perm, type=2, norm="ortho"
        )
        dct_rows = tf.transpose(dct_rows_perm, perm=[0, 2, 1])

        # ------------------------------------------------------------
        # 2. DCT along columns
        # ------------------------------------------------------------
        dct_2d = tf.signal.dct(
            dct_rows, type=2, norm="ortho"
        )

        # Magnitude
        dct_magnitude = tf.abs(dct_2d)

        # Restore channel dimension: [batch, H, W, 1]
        dct_real = tf.expand_dims(dct_magnitude, axis=-1)

        return dct_real
