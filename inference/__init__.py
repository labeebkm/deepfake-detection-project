"""
Inference utilities for deepfake detection.
"""

from .detector import DeepfakeDetector
from .api import create_app

__all__ = [
    "DeepfakeDetector",
    "create_app",
]







