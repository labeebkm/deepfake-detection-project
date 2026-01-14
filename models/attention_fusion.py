"""
Attention-based fusion module for combining RGB and frequency features.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional


class AttentionFusion(keras.Model):
    """
    Attention-based fusion module.
    """
    
    def __init__(self, 
                 attention_dim: int = 256,
                 num_heads: int = 8,
                 **kwargs):
        """
        Initialize AttentionFusion.
        
        Args:
            attention_dim: Attention dimension
            num_heads: Number of attention heads
        """
        super().__init__(**kwargs)
        
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=attention_dim // num_heads
        )
        
        # Layer normalization
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(attention_dim, activation='relu'),
            layers.Dense(attention_dim)
        ])
        
        # Feature fusion
        self.fusion_conv = layers.Conv2D(attention_dim, (1, 1), padding='same')
        self.fusion_bn = layers.BatchNormalization()
    
    def call(self, inputs: List[tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: List of feature tensors [rgb_features, freq_features]
            training: Whether in training mode
            
        Returns:
            Fused features
        """
        rgb_features, freq_features = inputs
        
        # Ensure same spatial dimensions
        if rgb_features.shape[1:3] != freq_features.shape[1:3]:
            freq_features = tf.image.resize(freq_features, rgb_features.shape[1:3])
        
        # Concatenate features
        combined = tf.concat([rgb_features, freq_features], axis=-1)
        
        # Apply attention
        # Reshape for attention: (batch, height*width, channels)
        batch_size = tf.shape(combined)[0]
        h, w = tf.shape(combined)[1], tf.shape(combined)[2]
        channels = combined.shape[-1]
        
        combined_flat = tf.reshape(combined, [batch_size, h * w, channels])
        
        # Multi-head self-attention
        attended = self.multi_head_attention(combined_flat, combined_flat)
        attended = self.layer_norm1(attended + combined_flat, training=training)
        
        # Feed-forward
        ffn_output = self.ffn(attended)
        attended = self.layer_norm2(attended + ffn_output, training=training)
        
        # Reshape back to spatial
        attended = tf.reshape(attended, [batch_size, h, w, self.attention_dim])
        
        # Final fusion convolution
        fused = self.fusion_conv(attended)
        fused = self.fusion_bn(fused, training=training)
        fused = tf.nn.relu(fused)
        
        return fused







