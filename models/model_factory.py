"""
Model factory for creating different model architectures.
"""

import tensorflow as tf
from typing import Dict, Optional
from .three_stream_net import ThreeStreamEfficientNet


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
        
        if model_name == 'three_stream_efficientnet':
            return ThreeStreamEfficientNet(
                backbone_name=config.get('backbone', 'efficientnet-b4'),
                pretrained=config.get('pretrained', True),
                num_classes=config.get('num_classes', 2),
                dropout_rate=config.get('dropout_rate', 0.5),
                feature_dim=config.get('feature_stream', {}).get('input_dim', 128),
                dct_size=config.get('frequency_stream', {}).get('dct_size', 8),
                num_filters=config.get('frequency_stream', {}).get('num_filters', 64),
            )
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Only 'three_stream_efficientnet' is supported."
            )
    
    @staticmethod
    def load_model(model_path: str) -> tf.keras.Model:
        """
        Load a saved TensorFlow model (SavedModel format).
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model instance
        """
        return tf.keras.models.load_model(model_path)

    @staticmethod
    def create_and_load_weights(
        model_name: str,
        config: Optional[Dict],
        weights_path: str,
    ) -> tf.keras.Model:
        """
        Create the model in code, build it with a dummy forward pass, then load weights.

        This is the most reliable approach for subclassed models (like ThreeStreamEfficientNet).
        """
        config = dict(config or {})
        # We are about to load trained weights; do not download ImageNet weights during model construction.
        config["pretrained"] = False
        model = ModelFactory.create_model(model_name, config)

        feature_dim = config.get("feature_stream", {}).get("input_dim", 128)
        dummy_images = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
        dummy_features = tf.zeros([1, feature_dim], dtype=tf.float32)
        model((dummy_images, dummy_features), training=False)

        model.load_weights(weights_path)
        return model


