"""
Data difficulty estimation based on EDA findings.
"""

import numpy as np
from typing import Dict, List, Optional
from .artifact_detector import ArtifactDetector
from .statistical_tests import StatisticalTester


class DataDifficultyEstimator:
    """
    Estimate dataset difficulty based on EDA findings.
    """
    
    def __init__(self):
        """Initialize DataDifficultyEstimator."""
        self.artifact_detector = ArtifactDetector()
        self.statistical_tester = StatisticalTester()
    
    def estimate_difficulty(self, real_images: List[np.ndarray], 
                           fake_images: List[np.ndarray]) -> Dict:
        """
        Estimate dataset difficulty based on artifact visibility and statistical differences.
        
        Args:
            real_images: List of real image arrays
            fake_images: List of fake image arrays
            
        Returns:
            Dictionary with difficulty metrics
        """
        # Sample images for analysis
        sample_size = min(100, len(real_images), len(fake_images))
        real_sample = real_images[:sample_size]
        fake_sample = fake_images[:sample_size]
        
        # 1. Artifact visibility score
        artifact_scores = []
        for img in fake_sample:
            artifacts = self.artifact_detector.detect_all_artifacts(img)
            artifact_scores.append(artifacts['overall_score'])
        
        avg_artifact_score = np.mean(artifact_scores)
        artifact_visibility = avg_artifact_score  # Higher = more visible = easier
        
        # 2. Statistical separability
        real_pixels = np.concatenate([img.flatten() for img in real_sample])
        fake_pixels = np.concatenate([img.flatten() for img in fake_sample])
        
        ks_result = self.statistical_tester.kolmogorov_smirnov_test(real_pixels, fake_pixels)
        statistical_separability = ks_result['statistic']  # Higher = more separable = easier
        
        # 3. Feature variance (higher variance = harder)
        real_variance = np.var([np.mean(img) for img in real_sample])
        fake_variance = np.var([np.mean(img) for img in fake_sample])
        avg_variance = (real_variance + fake_variance) / 2
        variance_score = 1.0 / (1.0 + avg_variance / 1000)  # Normalize
        
        # 4. Class overlap (estimated from pixel distribution overlap)
        real_mean = np.mean(real_pixels)
        fake_mean = np.mean(fake_pixels)
        real_std = np.std(real_pixels)
        fake_std = np.std(fake_pixels)
        
        # Calculate overlap using normal distribution approximation
        overlap = self._calculate_distribution_overlap(
            real_mean, real_std, fake_mean, fake_std
        )
        overlap_score = 1.0 - overlap  # Lower overlap = easier
        
        # Combine metrics into difficulty score
        # Difficulty: 0 (easy) to 1 (hard)
        difficulty_score = 1.0 - (
            0.3 * artifact_visibility +
            0.3 * statistical_separability +
            0.2 * variance_score +
            0.2 * overlap_score
        )
        
        # Normalize to [0, 1]
        difficulty_score = max(0.0, min(1.0, difficulty_score))
        
        # Determine difficulty level
        if difficulty_score < 0.3:
            difficulty_level = "Easy"
        elif difficulty_score < 0.6:
            difficulty_level = "Medium"
        else:
            difficulty_level = "Hard"
        
        return {
            'difficulty_score': float(difficulty_score),
            'difficulty_level': difficulty_level,
            'artifact_visibility': float(artifact_visibility),
            'statistical_separability': float(statistical_separability),
            'variance_score': float(variance_score),
            'overlap_score': float(overlap_score),
            'recommendations': self._get_difficulty_recommendations(difficulty_score)
        }
    
    def _calculate_distribution_overlap(self, mean1: float, std1: float,
                                       mean2: float, std2: float) -> float:
        """Calculate overlap between two normal distributions."""
        # Simplified overlap calculation
        # Using Bhattacharyya coefficient approximation
        mean_diff = abs(mean1 - mean2)
        avg_std = (std1 + std2) / 2
        
        if avg_std == 0:
            return 1.0 if mean_diff == 0 else 0.0
        
        # Normalized distance
        normalized_dist = mean_diff / avg_std
        
        # Overlap decreases with distance
        overlap = np.exp(-normalized_dist / 2)
        
        return float(overlap)
    
    def _get_difficulty_recommendations(self, difficulty_score: float) -> List[str]:
        """Get recommendations based on difficulty score."""
        recommendations = []
        
        if difficulty_score < 0.3:
            recommendations.append("Dataset is relatively easy. Consider using simpler models or focusing on efficiency.")
            recommendations.append("Artifacts are clearly visible. Model should achieve high accuracy.")
        elif difficulty_score < 0.6:
            recommendations.append("Dataset has moderate difficulty. Use robust architectures with data augmentation.")
            recommendations.append("Consider ensemble methods for better performance.")
        else:
            recommendations.append("Dataset is challenging. Use advanced architectures and extensive data augmentation.")
            recommendations.append("Consider using self-supervised learning or transfer learning from larger datasets.")
            recommendations.append("Focus on frequency domain features and artifact-specific detection.")
        
        return recommendations


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for deepfake artifact detection.
    """
    
    def __init__(self):
        """Initialize FeatureImportanceAnalyzer."""
        self.artifact_detector = ArtifactDetector()
    
    def analyze_feature_importance(self, real_images: List[np.ndarray],
                                   fake_images: List[np.ndarray]) -> Dict:
        """
        Analyze which artifact features are most important for detection.
        
        Args:
            real_images: List of real image arrays
            fake_images: List of fake image arrays
            
        Returns:
            Dictionary with feature importance scores
        """
        sample_size = min(50, len(real_images), len(fake_images))
        real_sample = real_images[:sample_size]
        fake_sample = fake_images[:sample_size]
        
        # Analyze artifacts in fake images
        artifact_scores = {
            'warping': [],
            'lighting': [],
            'blending': [],
            'jpeg': [],
            'frequency': []
        }
        
        for img in fake_sample:
            artifacts = self.artifact_detector.detect_all_artifacts(img)
            artifact_scores['warping'].append(artifacts['warping']['warping_score'])
            artifact_scores['lighting'].append(artifacts['lighting']['inconsistency_score'])
            artifact_scores['blending'].append(artifacts['blending']['blending_score'])
            artifact_scores['jpeg'].append(artifacts['jpeg']['jpeg_score'])
            artifact_scores['frequency'].append(artifacts['frequency']['inconsistency_score'])
        
        # Calculate importance (mean score for each artifact type)
        importance_scores = {
            feature: float(np.mean(scores))
            for feature, scores in artifact_scores.items()
        }
        
        # Normalize to [0, 1]
        max_score = max(importance_scores.values()) if importance_scores.values() else 1.0
        if max_score > 0:
            importance_scores = {
                k: v / max_score for k, v in importance_scores.items()
            }
        
        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'importance_scores': importance_scores,
            'sorted_features': sorted_features,
            'most_important': sorted_features[0][0] if sorted_features else None,
            'recommendations': self._get_feature_recommendations(sorted_features)
        }
    
    def _get_feature_recommendations(self, sorted_features: List[tuple]) -> List[str]:
        """Get recommendations based on feature importance."""
        recommendations = []
        
        if sorted_features:
            top_feature = sorted_features[0][0]
            recommendations.append(f"Focus on {top_feature} artifacts - they are most prevalent in this dataset.")
            
            if 'frequency' in [f[0] for f in sorted_features[:2]]:
                recommendations.append("Frequency domain features are important. Consider using frequency analysis branch in model.")
            
            if 'blending' in [f[0] for f in sorted_features[:2]]:
                recommendations.append("Blending artifacts are significant. Focus on boundary detection.")
        
        return recommendations








