"""
Automated artifact detection for deepfake images.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from scipy.fft import dct
import tensorflow as tf


class ArtifactDetector:
    """Detect various artifacts in deepfake images."""
    
    def __init__(self):
        """Initialize ArtifactDetector."""
        pass
    
    def detect_face_warping(self, image: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> Dict:
        """
        Detect unnatural face warping artifacts.
        
        Args:
            image: Input image array
            face_landmarks: Optional facial landmarks
            
        Returns:
            Dictionary with warping detection results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Detect irregular patterns using Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        # Calculate line angles
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
        
        # Calculate angle variance (high variance indicates warping)
        angle_variance = np.var(angles) if angles else 0
        
        # Detect texture inconsistencies
        # Use local binary patterns or Gabor filters
        lbp = self._local_binary_pattern(gray)
        texture_variance = np.var(lbp)
        
        result = {
            "edge_density": float(edge_density),
            "angle_variance": float(angle_variance),
            "texture_variance": float(texture_variance),
            "warping_score": float(edge_density * angle_variance / 1000),  # Normalized score
            "has_warping": edge_density > 0.1 and angle_variance > 500
        }
        
        return result
    
    def detect_lighting_inconsistency(self, image: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Detect inconsistent lighting patterns.
        
        Args:
            image: Input image array
            face_region: Optional (x, y, width, height) face bounding box
            
        Returns:
            Dictionary with lighting inconsistency results
        """
        # Convert to LAB color space for better lighting analysis
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]  # L channel represents lightness
        else:
            l_channel = image
        
        # Extract face region if provided
        if face_region:
            x, y, w, h = face_region
            l_channel = l_channel[y:y+h, x:x+w]
        
        # Calculate lighting gradient
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate statistics
        mean_gradient = np.mean(gradient_magnitude)
        std_gradient = np.std(gradient_magnitude)
        
        # Detect abrupt changes (inconsistent lighting)
        threshold = mean_gradient + 2 * std_gradient
        abrupt_changes = np.sum(gradient_magnitude > threshold) / gradient_magnitude.size
        
        # Calculate lighting distribution
        hist, _ = np.histogram(l_channel, bins=256, range=(0, 256))
        hist_normalized = hist / hist.sum()
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        
        result = {
            "mean_gradient": float(mean_gradient),
            "std_gradient": float(std_gradient),
            "abrupt_changes": float(abrupt_changes),
            "lighting_entropy": float(entropy),
            "inconsistency_score": float(abrupt_changes * std_gradient / 100),
            "has_inconsistency": abrupt_changes > 0.05
        }
        
        return result
    
    def detect_blending_artifacts(self, image: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Detect blending boundary artifacts.
        
        Args:
            image: Input image array
            face_region: Optional face bounding box
            
        Returns:
            Dictionary with blending artifact results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Extract face region if provided
        if face_region:
            x, y, w, h = face_region
            gray = gray[y:y+h, x:x+w]
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to detect boundaries
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        
        # Find contours (potential blending boundaries)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate boundary metrics
        if contours:
            # Find largest contour (likely face boundary)
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            
            # Circularity (blending artifacts often create irregular boundaries)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Boundary smoothness
            # Calculate curvature
            if len(largest_contour) > 2:
                curvature = self._calculate_curvature(largest_contour)
                smoothness = 1.0 / (1.0 + np.std(curvature))
            else:
                smoothness = 0
        else:
            circularity = 0
            smoothness = 0
            perimeter = 0
            area = 0
        
        # Detect color inconsistencies at boundaries
        # Use gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradients at boundaries indicate blending artifacts
        boundary_gradient = np.mean(gradient_magnitude[edges > 0]) if np.any(edges > 0) else 0
        
        result = {
            "circularity": float(circularity),
            "smoothness": float(smoothness),
            "boundary_gradient": float(boundary_gradient),
            "num_contours": len(contours),
            "blending_score": float((1 - smoothness) * boundary_gradient / 100),
            "has_blending_artifacts": smoothness < 0.7 or boundary_gradient > 50
        }
        
        return result
    
    def detect_jpeg_artifacts(self, image: np.ndarray) -> Dict:
        """
        Detect JPEG compression artifacts.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with JPEG artifact results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply DCT to detect block artifacts
        # JPEG compression creates 8x8 block patterns
        h, w = gray.shape
        block_size = 8
        
        # Pad image to be divisible by block_size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Calculate DCT for each block
        dct_blocks = []
        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = padded[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block.astype(np.float32))
                dct_blocks.append(dct_block)
        
        dct_blocks = np.array(dct_blocks)
        
        # Calculate high-frequency energy (compression artifacts)
        # High frequencies are in the bottom-right of DCT block
        hf_energy = []
        for dct_block in dct_blocks:
            # Extract high-frequency components
            hf_region = dct_block[4:, 4:]
            energy = np.sum(np.abs(hf_region))
            hf_energy.append(energy)
        
        hf_energy = np.array(hf_energy)
        
        # Calculate block boundary artifacts
        # Detect discontinuities at block boundaries
        block_boundaries = 0
        for i in range(block_size, padded.shape[0], block_size):
            diff = np.abs(padded[i, :] - padded[i-1, :])
            block_boundaries += np.sum(diff > 10)
        for j in range(block_size, padded.shape[1], block_size):
            diff = np.abs(padded[:, j] - padded[:, j-1])
            block_boundaries += np.sum(diff > 10)
        
        block_boundary_score = block_boundaries / (padded.shape[0] * padded.shape[1])
        
        result = {
            "mean_hf_energy": float(np.mean(hf_energy)),
            "std_hf_energy": float(np.std(hf_energy)),
            "block_boundary_score": float(block_boundary_score),
            "jpeg_score": float(block_boundary_score * np.mean(hf_energy) / 1000),
            "has_jpeg_artifacts": block_boundary_score > 0.01 or np.mean(hf_energy) > 100
        }
        
        return result
    
    def detect_frequency_inconsistencies(self, image: np.ndarray) -> Dict:
        """
        Detect frequency domain inconsistencies using DCT.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with frequency inconsistency results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply 2D DCT
        dct_2d = dct(dct(gray.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')
        magnitude = np.abs(dct_2d)
        
        # DCT is real-valued, so no phase component
        # Use gradient of magnitude instead for coherence
        magnitude_gradient = np.gradient(magnitude)
        
        # Calculate frequency spectrum statistics
        # Real images typically have smooth frequency distributions
        # Deepfakes may have abrupt changes
        
        # Divide into low, mid, and high frequency regions
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency (center region)
        low_freq_radius = min(h, w) // 4
        low_freq_mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 <= low_freq_radius**2
        low_freq_mask[mask] = True
        
        # High frequency (outer region)
        high_freq_radius = min(h, w) // 2
        high_freq_mask = np.zeros((h, w), dtype=bool)
        mask = (x - center_w)**2 + (y - center_h)**2 > high_freq_radius**2
        high_freq_mask[mask] = True
        
        # Mid frequency (between low and high)
        mid_freq_mask = ~(low_freq_mask | high_freq_mask)
        
        # Calculate energy in each region
        low_freq_energy = np.mean(magnitude[low_freq_mask])
        mid_freq_energy = np.mean(magnitude[mid_freq_mask])
        high_freq_energy = np.mean(magnitude[high_freq_mask])
        
        # Calculate energy ratios
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        low_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
        mid_ratio = mid_freq_energy / total_energy if total_energy > 0 else 0
        high_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Detect inconsistencies (unusual frequency distribution)
        # Real images typically have more low-frequency energy
        inconsistency_score = abs(low_ratio - 0.6) + abs(high_ratio - 0.1)  # Expected ratios
        
        # Calculate magnitude coherence (inconsistent patterns indicate artifacts)
        magnitude_coherence = 1.0 / (1.0 + np.std(magnitude_gradient))
        
        result = {
            "low_freq_ratio": float(low_ratio),
            "mid_freq_ratio": float(mid_ratio),
            "high_freq_ratio": float(high_ratio),
            "magnitude_coherence": float(magnitude_coherence),
            "inconsistency_score": float(inconsistency_score),
            "has_inconsistencies": inconsistency_score > 0.3 or magnitude_coherence < 0.5
        }
        
        return result
    
    def detect_all_artifacts(self, image: np.ndarray, face_region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Detect all types of artifacts.
        
        Args:
            image: Input image array
            face_region: Optional face bounding box
            
        Returns:
            Dictionary with all artifact detection results
        """
        results = {
            "warping": self.detect_face_warping(image),
            "lighting": self.detect_lighting_inconsistency(image, face_region),
            "blending": self.detect_blending_artifacts(image, face_region),
            "jpeg": self.detect_jpeg_artifacts(image),
            "frequency": self.detect_frequency_inconsistencies(image)
        }
        
        # Calculate overall artifact score
        scores = [
            results["warping"]["warping_score"],
            results["lighting"]["inconsistency_score"],
            results["blending"]["blending_score"],
            results["jpeg"]["jpeg_score"],
            results["frequency"]["inconsistency_score"]
        ]
        
        results["overall_score"] = float(np.mean(scores))
        results["is_fake"] = results["overall_score"] > 0.3
        
        return results
    
    def _local_binary_pattern(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        # Simplified LBP implementation
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if x < h and y < w:
                        if image[x, y] >= center:
                            code |= (1 << k)
                lbp[i, j] = code
        
        return lbp
    
    def _calculate_curvature(self, contour: np.ndarray) -> np.ndarray:
        """Calculate curvature of a contour."""
        if len(contour) < 3:
            return np.array([])
        
        # Calculate first and second derivatives
        dx = np.gradient(contour[:, 0, 0])
        dy = np.gradient(contour[:, 0, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)**(3/2)
        
        return curvature

