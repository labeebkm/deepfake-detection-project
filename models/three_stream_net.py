"""
Three-stream EfficientNet architecture: RGB + Frequency + Hand-crafted Features.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple

from .frequency_net import FrequencyNet


def _get_efficientnet_backbone(backbone_name: str):
    """
    Resolve an EfficientNet backbone constructor from a string like 'efficientnet-b4'.
    """
    name = (backbone_name or "efficientnet-b4").lower().replace("_", "-")

    mapping = {
        "efficientnet-b0": keras.applications.EfficientNetB0,
        "efficientnet-b1": keras.applications.EfficientNetB1,
        "efficientnet-b2": keras.applications.EfficientNetB2,
        "efficientnet-b3": keras.applications.EfficientNetB3,
        "efficientnet-b4": keras.applications.EfficientNetB4,
        "efficientnet-b5": keras.applications.EfficientNetB5,
        "efficientnet-b6": keras.applications.EfficientNetB6,
        "efficientnet-b7": keras.applications.EfficientNetB7,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported backbone_name='{backbone_name}'. "
            f"Supported: {', '.join(sorted(mapping.keys()))}"
        )
    return mapping[name]


class ThreeStreamEfficientNet(keras.Model):
    """
    Three-stream architecture:
    1. RGB Stream (EfficientNet)
    2. Frequency Stream (DCT using FrequencyNet)
    3. Feature Stream (Dense MLP for hand-crafted features)
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet-b4",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        feature_dim: int = 128,  # Dim of the explicit hand-crafted feature vector
        dct_size: int = 8,
        num_filters: int = 64,
        **kwargs,
    ):
        """
        Initialize ThreeStreamEfficientNet.
        
        Args:
            backbone_name: EfficientNet backbone name
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            feature_dim: Size of the hand-crafted feature vector
        """
        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.dct_size = dct_size
        self.num_filters = num_filters
        
        # 1. RGB Stream
        backbone_cls = _get_efficientnet_backbone(backbone_name)
        weights = "imagenet" if pretrained else None
        self.rgb_backbone = backbone_cls(
            include_top=False,
            weights=weights,
            input_shape=(224, 224, 3)
        )
        self.rgb_backbone.trainable = True
        self.rgb_pool = layers.GlobalAveragePooling2D()
        
        # 2. Frequency Stream (DCT)
        self.frequency_net = FrequencyNet(
            dct_size=dct_size,
            num_filters=num_filters,
        )
        self.freq_pool = layers.GlobalAveragePooling2D()
        
        # 3. Feature Stream (MLP)
        # Input shape: (batch_size, feature_dim)
        self.feature_net = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization()
        ], name='feature_stream')
        
        # Fusion
        # Fusing 3 vectors: RGB embedding, Freq embedding, Feature embedding
        # We project them to same dimension before attention
        self.fusion_dim = 256
        
        self.rgb_proj = layers.Dense(self.fusion_dim, activation='relu')
        self.freq_proj = layers.Dense(self.fusion_dim, activation='relu')
        self.feat_proj = layers.Dense(self.fusion_dim, activation='relu')
        
        self.attention = layers.Attention()
        self.concat = layers.Concatenate(axis=1)
        self.flatten = layers.Flatten()
        
        # Classification head
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes, activation='softmax', name='predictions')
        
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Tuple of (image_tensor, feature_vector)
            
        Returns:
            Classification probabilities
        """
        images, explicit_features = inputs
        
        # Stream 1: RGB
        x_rgb = self.rgb_backbone(images, training=training)
        x_rgb = self.rgb_pool(x_rgb)
        x_rgb = self.rgb_proj(x_rgb)
        
        # Stream 2: Frequency
        x_freq = self.frequency_net(images, training=training)
        x_freq = self.freq_pool(x_freq)
        x_freq = self.freq_proj(x_freq)
        
        # Stream 3: Hand-crafted Features
        x_feat = self.feature_net(explicit_features, training=training)
        x_feat = self.feat_proj(x_feat)
        
        # Stack for attention: [batch, 3, fusion_dim]
        # Reshape to allow attention mechanism to weigh the 3 streams
        rgb_expanded = tf.expand_dims(x_rgb, 1)
        freq_expanded = tf.expand_dims(x_freq, 1)
        feat_expanded = tf.expand_dims(x_feat, 1)
        
        stacked = self.concat([rgb_expanded, freq_expanded, feat_expanded])
        
        # Self-attention over the 3 modalities.
        #
        # Keras 3's `layers.Attention` expects `[query, value, key]` for shape
        # inference (e.g., `model.summary()`), even though runtime often allows
        # `[query, value]` with `key=value`. Pass all three explicitly so the
        # model is fully traceable and summary-friendly.
        attended = self.attention([stacked, stacked, stacked], training=training)
        
        # Global pooling over the streams (average or max) - or just flatten
        # We flatten: [batch, 3 * fusion_dim] -> dense
        fused = self.flatten(attended)
        
        # Final classification
        x = self.dropout(fused, training=training)
        return self.classifier(x)
