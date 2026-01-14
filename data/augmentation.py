"""
Data augmentation utilities using TensorFlow and Albumentations.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class Augmentor:
    """Data augmentation utilities."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Augmentor.
        
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Initialize Albumentations if available
        if ALBUMENTATIONS_AVAILABLE:
            self.albumentations_transform = self._create_albumentations_pipeline()
        else:
            self.albumentations_transform = None
    
    def _create_albumentations_pipeline(self) -> Optional[A.Compose]:
        """Create Albumentations augmentation pipeline."""
        if not ALBUMENTATIONS_AVAILABLE:
            return None
        
        transforms = []
        
        if self.config.get('horizontal_flip', 0.5) > 0:
            transforms.append(A.HorizontalFlip(p=self.config.get('horizontal_flip', 0.5)))
        
        if self.config.get('rotation', 0) > 0:
            transforms.append(A.Rotate(limit=self.config.get('rotation', 15), p=0.5))
        
        if self.config.get('brightness', 0) > 0:
            transforms.append(A.RandomBrightness(limit=self.config.get('brightness', 0.2), p=0.5))
        
        if self.config.get('contrast', 0) > 0:
            transforms.append(A.RandomContrast(limit=self.config.get('contrast', 0.2), p=0.5))
        
        if self.config.get('saturation', 0) > 0:
            transforms.append(A.RandomSaturation(limit=self.config.get('saturation', 0.2), p=0.5))
        
        if self.config.get('hue', 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=self.config.get('hue', 10),
                sat_shift_limit=self.config.get('saturation', 20),
                val_shift_limit=self.config.get('brightness', 20),
                p=0.5
            ))
        
        if self.config.get('gaussian_noise', 0) > 0:
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=self.config.get('gaussian_noise', 0.1)))
        
        if self.config.get('blur', 0) > 0:
            transforms.append(A.GaussianBlur(blur_limit=3, p=self.config.get('blur', 0.1)))
        
        if self.config.get('jpeg_compression', 0) > 0:
            transforms.append(A.ImageCompression(quality_lower=70, quality_upper=100, p=self.config.get('jpeg_compression', 0.1)))
        
        if transforms:
            return A.Compose(transforms)
        return None
    
    def augment_tf(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply TensorFlow-based augmentation.
        
        Args:
            image: Input image tensor
            label: Input label tensor
            
        Returns:
            Augmented (image, label) tuple
        """
        if not self.enabled:
            return image, label
        
        # Random horizontal flip
        if self.config.get('horizontal_flip', 0.5) > 0:
            image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        if self.config.get('rotation', 0) > 0:
            angle = tf.random.uniform([], -self.config.get('rotation', 15), self.config.get('rotation', 15))
            image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))
        
        # Random brightness
        if self.config.get('brightness', 0) > 0:
            image = tf.image.random_brightness(image, max_delta=self.config.get('brightness', 0.2))
        
        # Random contrast
        if self.config.get('contrast', 0) > 0:
            image = tf.image.random_contrast(image, lower=1-self.config.get('contrast', 0.2), 
                                            upper=1+self.config.get('contrast', 0.2))
        
        # Random saturation
        if self.config.get('saturation', 0) > 0:
            image = tf.image.random_saturation(image, lower=1-self.config.get('saturation', 0.2),
                                              upper=1+self.config.get('saturation', 0.2))
        
        # Random hue
        if self.config.get('hue', 0) > 0:
            image = tf.image.random_hue(image, max_delta=self.config.get('hue', 0.1))
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def augment_albumentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Albumentations-based augmentation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image as numpy array
        """
        if not self.enabled or self.albumentations_transform is None:
            return image
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply augmentation
        augmented = self.albumentations_transform(image=image)
        image = augmented['image']
        
        # Convert back to float32 if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        return image







