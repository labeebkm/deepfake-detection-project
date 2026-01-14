"""
Frequency domain utility functions using DCT.
"""

import numpy as np
import tensorflow as tf
from scipy.fft import dct


def apply_dct(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Apply 2D DCT transformation.
    
    Args:
        image: Input image array
        block_size: DCT block size (for block-based DCT, if needed)
        
    Returns:
        DCT coefficients
    """
    # Apply 2D DCT
    dct_2d = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    return np.abs(dct_2d)

