"""
Deepfake detection inference utilities.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import cv2


class DeepfakeDetector:
    """Deepfake detection inference class."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize DeepfakeDetector.
        
        Args:
            model_path: Path to saved model
            confidence_threshold: Confidence threshold for predictions
        """
        self.model = tf.keras.models.load_model(model_path)
        self.confidence_threshold = confidence_threshold
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Predict if image is deepfake.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        
        # Get class probabilities
        real_prob = float(predictions[0][0])
        fake_prob = float(predictions[0][1])
        
        # Determine prediction
        is_fake = fake_prob > self.confidence_threshold
        confidence = fake_prob if is_fake else real_prob
        
        return {
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'real_probability': real_prob,
            'fake_probability': fake_prob
        }
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Predict batch of images.
        
        Args:
            images: Batch of images
            
        Returns:
            Array of predictions
        """
        processed = self._preprocess_batch(images)
        predictions = self.model.predict(processed, verbose=0)
        return predictions
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single image."""
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def _preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess batch of images."""
        processed = []
        for img in images:
            processed.append(cv2.resize(img, (224, 224)))
        
        processed = np.array(processed, dtype=np.float32) / 255.0
        return processed








