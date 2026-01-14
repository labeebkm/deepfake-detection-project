"""
Model factory for creating different model architectures.
"""

import tensorflow as tf
from typing import Dict, Optional
from .dual_stream_efficientnet import DualStreamEfficientNet


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_name: str, config: Optional[Dict] = None) -> tf.keras.Model:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model architecture
            config: Model configuration dictionary
            
        Returns:
            Model instance
        """
        config = config or {}
        
        if model_name == 'dual_stream_efficientnet':
            return DualStreamEfficientNet(
                backbone_name=config.get('backbone', 'efficientnet-b4'),
                num_classes=config.get('num_classes', 2),
                dropout_rate=config.get('dropout_rate', 0.5),
                frequency_branch=config.get('frequency_branch', {}).get('enabled', True),
                attention_fusion=config.get('attention_fusion', {}).get('enabled', True)
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    @staticmethod
    def load_model(model_path: str) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model instance
        """
        return tf.keras.models.load_model(model_path)







