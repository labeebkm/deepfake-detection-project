"""
Model architectures for deepfake detection.
"""

from .frequency_net import FrequencyNet
from .three_stream_net import ThreeStreamEfficientNet
from .model_factory import ModelFactory

__all__ = [
    "FrequencyNet",
    "ThreeStreamEfficientNet",
    "ModelFactory",
]
