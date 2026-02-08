"""
Frequency domain analysis network using DCT.
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FrequencyNet(keras.Model):
    """
    Frequency domain analysis network using 2D DCT magnitude.

    Notes:
    - `dct_size` is reserved for a future block-based DCT. The current implementation
      applies a full-image 2D DCT (row DCT then column DCT).
    """

    def __init__(
        self,
        dct_size: int = 8,
        num_filters: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dct_size = dct_size
        self.num_filters = num_filters

        self.dct_conv = layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )
        self.dct_bn = layers.BatchNormalization()

        self.process_conv = layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            activation="relu",
        )
        self.process_bn = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Args:
            inputs: Image tensor [batch, H, W, C]
            training: Training mode flag

        Returns:
            Frequency feature map [batch, H, W, num_filters]
        """
        # tf.signal.dct can be problematic with float16 on some TF versions.
        inputs = tf.cast(inputs, tf.float32)

        gray = tf.image.rgb_to_grayscale(inputs) if inputs.shape[-1] == 3 else inputs

        dct_features = self._apply_dct(gray)
        dct_features = self.dct_conv(dct_features)
        dct_features = self.dct_bn(dct_features, training=training)

        processed = self.process_conv(dct_features)
        processed = self.process_bn(processed, training=training)

        # If the rest of the model runs in mixed precision, cast back.
        if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
            processed = tf.cast(processed, tf.float16)

        return processed

    def _apply_dct(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply 2D DCT to a grayscale image tensor.

        Args:
            image: Grayscale tensor [batch, H, W, 1]

        Returns:
            DCT magnitude tensor [batch, H, W, 1]
        """
        image_2d = tf.squeeze(image, axis=-1)  # [batch, H, W]

        # DCT along rows: make rows the last axis for tf.signal.dct, then transpose back.
        image_perm = tf.transpose(image_2d, perm=[0, 2, 1])  # [batch, W, H]
        dct_rows_perm = tf.signal.dct(image_perm, type=2, norm="ortho")
        dct_rows = tf.transpose(dct_rows_perm, perm=[0, 2, 1])  # [batch, H, W]

        # DCT along columns (already last axis).
        dct_2d = tf.signal.dct(dct_rows, type=2, norm="ortho")

        magnitude = tf.abs(dct_2d)
        return tf.expand_dims(magnitude, axis=-1)

