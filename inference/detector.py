"""
Deepfake detection inference utilities.
"""

import numpy as np
from typing import Dict, Tuple
import cv2
import yaml

from models.model_factory import ModelFactory

from training.feature_generator import FeatureGenerator


class DeepfakeDetector:
    """Deepfake detection inference class."""
    
    def __init__(
        self,
        weights_path: str,
        config_path: str,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize DeepfakeDetector.
        
        Args:
            weights_path: Path to saved weights (e.g. best_model.weights.h5)
            config_path: Path to config.yaml (used to rebuild the model)
            confidence_threshold: Confidence threshold for predictions
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "three_stream_efficientnet")

        self.model = ModelFactory.create_and_load_weights(
            model_name=model_name,
            config=model_cfg,
            weights_path=weights_path,
        )
        self.confidence_threshold = confidence_threshold
        feature_dim = model_cfg.get("feature_stream", {}).get("input_dim", 128)
        self.feature_generator = FeatureGenerator(feature_dim=feature_dim)
    
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
            "is_fake": bool(is_fake),
            "confidence": float(confidence),
            "real_probability": real_prob,
            "fake_probability": fake_prob,
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
    
    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Ensure the input is HxWx3 RGB."""
        if image is None:
            raise ValueError("Input image is None")

        if len(image.shape) == 2:
            # Grayscale -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single image for the three-stream model.

        Returns:
            (image_batch, feature_batch)
        """
        image = self._ensure_rgb(image)

        # Match training: resize first, then compute features on uint8 resized image.
        image_resized = cv2.resize(image, (224, 224))

        # Explicit features
        features = self.feature_generator.extract(image_resized)  # (feature_dim,)
        feature_batch = np.expand_dims(features.astype(np.float32), axis=0)

        # Model image input
        image_float = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_float, axis=0)

        return image_batch, feature_batch
    
    def _preprocess_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess a batch of images for the three-stream model."""
        image_batches = []
        feature_batches = []

        for img in images:
            img = self._ensure_rgb(img)
            img_resized = cv2.resize(img, (224, 224))

            feature_batches.append(self.feature_generator.extract(img_resized))
            image_batches.append(img_resized.astype(np.float32) / 255.0)

        image_batch = np.array(image_batches, dtype=np.float32)
        feature_batch = np.array(feature_batches, dtype=np.float32)

        return image_batch, feature_batch


