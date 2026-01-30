"""
Feature generator for extracting advanced statistical and texture features.
"""

import numpy as np
import cv2
from scipy.stats import skew, kurtosis
import tensorflow as tf

class FeatureGenerator:
    """
    Extracts explicit features from images to feed into the Feature Stream.
    Features include:
    1. ELA (Error Level Analysis) statistics
    2. Texture features (LBP histograms)
    3. Color statistics (YCrCb moments)
    4. Frequency domain coherence
    """
    
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: RGB image array (0-1 or 0-255)
            
        Returns:
            Normalized feature vector
        """
        # Ensure image is 0-255 uint8 for CV2 operations
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image
            
        features = []
        
        # 1. ELA Features (Error Level Analysis)
        ela_features = self._extract_ela(img_uint8)
        features.extend(ela_features)
        
        # 2. Texture Features (LBP)
        lbp_features = self._extract_lbp(img_uint8)
        features.extend(lbp_features)
        
        # 3. Color Statistics
        color_features = self._extract_color_stats(img_uint8)
        features.extend(color_features)
        
        # Feature vector construction
        feature_vector = np.array(features, dtype=np.float32)
        
        # Padding or truncation to fixed dimension
        if len(feature_vector) < self.feature_dim:
            feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.feature_dim]
            
        # Normalize
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector

    def _extract_ela(self, image: np.ndarray) -> list:
        """
        Perform Error Level Analysis to detect compression inconsistencies.
        """
        # Save validation image at 90% quality
        _, encoded_img = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Calculate difference
        diff = np.abs(image.astype(np.float32) - decoded_img.astype(np.float32))
        
        # Calculate statistics of difference
        ela_mean = np.mean(diff)
        ela_std = np.std(diff)
        ela_max = np.max(diff)
        
        # Channel-wise stats
        b_mean, g_mean, r_mean = np.mean(diff, axis=(0,1))
        
        return [ela_mean, ela_std, ela_max, b_mean, g_mean, r_mean]

    def _extract_lbp(self, image: np.ndarray) -> list:
        """
        Extract Local Binary Pattern histograms for texture analysis.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Improved LBP (Uniform)
        radius = 1
        n_points = 8 * radius
        
        # Simple Python implementation for LBP to avoid heavy dependencies if skimage not available
        # Or using basic gradient logic as proxy for texture complexity
        
        # Calculate gradients as simple texture descriptors
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Histogram of gradients (HOG-like)
        hist_mag, _ = np.histogram(mag, bins=10, density=True)
        hist_ang, _ = np.histogram(ang, bins=10, density=True)
        
        return list(hist_mag) + list(hist_ang)

    def _extract_color_stats(self, image: np.ndarray) -> list:
        """
        Extract statistical moments from YCrCb color space.
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        stats = []
        for i in range(3):
            # Mean, Std, Skewness, Kurtosis for each channel
            channel = ycrcb[:,:,i].flatten()
            stats.append(np.mean(channel))
            stats.append(np.std(channel))
            stats.append(skew(channel) if len(channel) > 0 else 0)
            stats.append(kurtosis(channel) if len(channel) > 0 else 0)
            
        return stats




#Just for testing - delete below code after testing
if __name__ == "__main__":
    import cv2

    img = cv2.imread(r"C:\Users\HP\Documents\QUEST\deepfake project\dataset\real\real_00006.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fg = FeatureGenerator(feature_dim=128)
    features = fg.extract(img)

    print("Feature vector shape:", features.shape)
    print("First 10 features:", features[:10])
