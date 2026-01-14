"""
Face utility functions.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def extract_face_region(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                       margin: float = 0.25) -> np.ndarray:
    """
    Extract face region from image.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        margin: Margin around face
        
    Returns:
        Extracted face region
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Add margin
    margin_x = int((x2 - x1) * margin)
    margin_y = int((y2 - y1) * margin)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    
    face = image[y1:y2, x1:x2]
    return face







