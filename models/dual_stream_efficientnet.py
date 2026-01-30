"""
Dual-stream EfficientNet architecture for deepfake detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
from .frequency_net import FrequencyNet
from .attention_fusion import AttentionFusion


class DualStreamEfficientNet(keras.Model):
    """
    Dual-stream EfficientNet model with frequency analysis branch.
    """
    
    def __init__(self, 
                 backbone_name: str = 'efficientnet-b4',
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 frequency_branch: bool = True,
                 attention_fusion: bool = True,
                 **kwargs):
        """
        Initialize DualStreamEfficientNet.
        
        Args:
            backbone_name: EfficientNet backbone name
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            frequency_branch: Whether to use frequency analysis branch
            attention_fusion: Whether to use attention-based fusion
        """
        super().__init__(**kwargs)
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.frequency_branch = frequency_branch
        self.attention_fusion = attention_fusion
        
        # RGB stream backbone
        base_model = keras.applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        base_model.trainable = True
        
        self.rgb_backbone = base_model
        
        # Frequency stream
        if frequency_branch:
            self.frequency_net = FrequencyNet()
        else:
            self.frequency_net = None
        
        # Attention fusion module
        if attention_fusion:
            self.attention_fusion = AttentionFusion()
        else:
            self.attention_fusion = None
        
        # Classification head
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes, activation='softmax', name='predictions')
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input image tensor
            training: Whether in training mode
            
        Returns:
            Classification logits
        """
        # RGB stream
        rgb_features = self.rgb_backbone(inputs, training=training)
        
        # Frequency stream
        if self.frequency_branch:
            freq_features = self.frequency_net(inputs, training=training)
        else:
            freq_features = None
        
        # Fusion
        if self.attention_fusion and freq_features is not None:
            fused_features = self.attention_fusion([rgb_features, freq_features], training=training)
        else:
            fused_features = rgb_features
        
        # Classification
        x = self.global_pool(fused_features)
        x = self.dropout(x, training=training)
        outputs = self.classifier(x)
        
        return outputs
    
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'backbone_name': self.backbone_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'frequency_branch': self.frequency_branch,
            'attention_fusion': self.attention_fusion
        })
        return config








