"""
Statistical testing functions for deepfake detection analysis.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


class StatisticalTester:
    """Statistical hypothesis testing for deepfake detection."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize StatisticalTester.
        
        Args:
            alpha: Significance level for tests
        """
        self.alpha = alpha
    
    def kolmogorov_smirnov_test(self, real_data: np.ndarray, fake_data: np.ndarray) -> Dict:
        """
        Perform Kolmogorov-Smirnov test to compare distributions.
        
        Args:
            real_data: Array of real image features
            fake_data: Array of fake image features
            
        Returns:
            Dictionary with test results
        """
        # Flatten if needed
        if len(real_data.shape) > 1:
            real_data = real_data.flatten()
        if len(fake_data.shape) > 1:
            fake_data = fake_data.flatten()
        
        statistic, p_value = stats.ks_2samp(real_data, fake_data)
        
        result = {
            "test": "Kolmogorov-Smirnov",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "significant": p_value < self.alpha,
            "interpretation": "Distributions are significantly different" if p_value < self.alpha else "No significant difference in distributions"
        }
        
        return result
    
    def chi_square_test(self, real_hist: np.ndarray, fake_hist: np.ndarray) -> Dict:
        """
        Perform Chi-square test for color distribution comparison.
        
        Args:
            real_hist: Histogram of real images
            fake_hist: Histogram of fake images
            
        Returns:
            Dictionary with test results
        """
        # Ensure same length
        min_len = min(len(real_hist), len(fake_hist))
        real_hist = real_hist[:min_len]
        fake_hist = fake_hist[:min_len]
        
        # Create contingency table
        observed = np.array([real_hist, fake_hist])
        
        statistic, p_value, dof, expected = stats.chi2_contingency(observed)
        
        result = {
            "test": "Chi-square",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "alpha": self.alpha,
            "significant": p_value < self.alpha,
            "interpretation": "Distributions are significantly different" if p_value < self.alpha else "No significant difference in distributions"
        }
        
        return result
    
    def t_test(self, real_features: np.ndarray, fake_features: np.ndarray) -> Dict:
        """
        Perform independent samples t-test.
        
        Args:
            real_features: Feature array from real images
            fake_features: Feature array from fake images
            
        Returns:
            Dictionary with test results
        """
        # Flatten if needed
        if len(real_features.shape) > 1:
            real_features = real_features.flatten()
        if len(fake_features.shape) > 1:
            fake_features = fake_features.flatten()
        
        statistic, p_value = stats.ttest_ind(real_features, fake_features)
        
        result = {
            "test": "Independent Samples t-test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "significant": p_value < self.alpha,
            "interpretation": "Means are significantly different" if p_value < self.alpha else "No significant difference in means"
        }
        
        return result
    
    def mann_whitney_test(self, real_features: np.ndarray, fake_features: np.ndarray) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            real_features: Feature array from real images
            fake_features: Feature array from fake images
            
        Returns:
            Dictionary with test results
        """
        # Flatten if needed
        if len(real_features.shape) > 1:
            real_features = real_features.flatten()
        if len(fake_features.shape) > 1:
            fake_features = fake_features.flatten()
        
        statistic, p_value = stats.mannwhitneyu(real_features, fake_features, alternative='two-sided')
        
        result = {
            "test": "Mann-Whitney U",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "significant": p_value < self.alpha,
            "interpretation": "Distributions are significantly different" if p_value < self.alpha else "No significant difference in distributions"
        }
        
        return result
    
    def anova_test(self, groups: List[np.ndarray], group_names: Optional[List[str]] = None) -> Dict:
        """
        Perform one-way ANOVA test for multiple groups.
        
        Args:
            groups: List of arrays, one for each group
            group_names: Optional list of group names
            
        Returns:
            Dictionary with test results
        """
        # Flatten all groups
        flattened_groups = [g.flatten() if len(g.shape) > 1 else g for g in groups]
        
        statistic, p_value = stats.f_oneway(*flattened_groups)
        
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(len(groups))]
        
        result = {
            "test": "One-way ANOVA",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": self.alpha,
            "significant": p_value < self.alpha,
            "num_groups": len(groups),
            "group_names": group_names,
            "interpretation": "At least one group mean is significantly different" if p_value < self.alpha else "No significant difference between group means"
        }
        
        return result
    
    def test_pixel_distributions(self, real_images: List[np.ndarray], fake_images: List[np.ndarray]) -> Dict:
        """
        Test pixel value distributions between real and fake images.
        
        Args:
            real_images: List of real image arrays
            fake_images: List of fake image arrays
            
        Returns:
            Dictionary with test results
        """
        # Extract pixel values
        real_pixels = np.concatenate([img.flatten() for img in real_images])
        fake_pixels = np.concatenate([img.flatten() for img in fake_images])
        
        # Perform multiple tests
        results = {
            "ks_test": self.kolmogorov_smirnov_test(real_pixels, fake_pixels),
            "t_test": self.t_test(real_pixels, fake_pixels),
            "mann_whitney": self.mann_whitney_test(real_pixels, fake_pixels)
        }
        
        return results
    
    def test_color_distributions(self, real_images: List[np.ndarray], fake_images: List[np.ndarray], 
                                 channel: int = 0) -> Dict:
        """
        Test color channel distributions.
        
        Args:
            real_images: List of real image arrays
            fake_images: List of fake image arrays
            channel: Channel index (0=R, 1=G, 2=B)
            
        Returns:
            Dictionary with test results
        """
        # Extract channel values
        if len(real_images[0].shape) == 3:
            real_channel = np.concatenate([img[:, :, channel].flatten() for img in real_images])
            fake_channel = np.concatenate([img[:, :, channel].flatten() for img in fake_images])
        else:
            real_channel = np.concatenate([img.flatten() for img in real_images])
            fake_channel = np.concatenate([img.flatten() for img in fake_images])
        
        # Create histograms
        real_hist, _ = np.histogram(real_channel, bins=256, range=(0, 256))
        fake_hist, _ = np.histogram(fake_channel, bins=256, range=(0, 256))
        
        # Perform tests
        results = {
            "chi_square": self.chi_square_test(real_hist, fake_hist),
            "ks_test": self.kolmogorov_smirnov_test(real_channel, fake_channel),
            "t_test": self.t_test(real_channel, fake_channel)
        }
        
        return results
    
    def test_feature_means(self, real_features: pd.DataFrame, fake_features: pd.DataFrame) -> pd.DataFrame:
        """
        Test feature means across all features.
        
        Args:
            real_features: DataFrame of real image features
            fake_features: DataFrame of fake image features
            
        Returns:
            DataFrame with test results for each feature
        """
        results = []
        
        for col in real_features.columns:
            if col in fake_features.columns:
                real_vals = real_features[col].values
                fake_vals = fake_features[col].values
                
                t_result = self.t_test(real_vals, fake_vals)
                mw_result = self.mann_whitney_test(real_vals, fake_vals)
                
                results.append({
                    "feature": col,
                    "t_statistic": t_result["statistic"],
                    "t_p_value": t_result["p_value"],
                    "t_significant": t_result["significant"],
                    "mw_statistic": mw_result["statistic"],
                    "mw_p_value": mw_result["p_value"],
                    "mw_significant": mw_result["significant"],
                    "real_mean": float(np.mean(real_vals)),
                    "fake_mean": float(np.mean(fake_vals)),
                    "real_std": float(np.std(real_vals)),
                    "fake_std": float(np.std(fake_vals))
                })
        
        return pd.DataFrame(results)
    
    def compare_datasets(self, datasets: Dict[str, Dict], feature_name: str = "pixel_values") -> Dict:
        """
        Compare multiple datasets using ANOVA.
        
        Args:
            datasets: Dictionary mapping dataset names to feature arrays
            feature_name: Name of feature being compared
            
        Returns:
            Dictionary with comparison results
        """
        groups = list(datasets.values())
        group_names = list(datasets.keys())
        
        anova_result = self.anova_test(groups, group_names)
        
        # Pairwise comparisons
        pairwise_results = {}
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names):
                if i < j:
                    pair_key = f"{name1}_vs_{name2}"
                    t_result = self.t_test(groups[i], groups[j])
                    pairwise_results[pair_key] = t_result
        
        return {
            "anova": anova_result,
            "pairwise": pairwise_results
        }


def compare_distributions(real_images: List[np.ndarray], 
                          fake_images: List[np.ndarray],
                          test_type: str = 'ks',
                          alpha: float = 0.05) -> Dict:
    """
    Compare distributions between real and fake images.
    
    Args:
        real_images: List of real image arrays
        fake_images: List of fake image arrays
        test_type: Type of test ('ks', 't_test', 'mann_whitney', 'all')
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    tester = StatisticalTester(alpha=alpha)
    
    # Extract pixel values
    real_pixels = np.concatenate([img.flatten() for img in real_images])
    fake_pixels = np.concatenate([img.flatten() for img in fake_images])
    
    results = {}
    
    if test_type in ['ks', 'all']:
        results['kolmogorov_smirnov'] = tester.kolmogorov_smirnov_test(real_pixels, fake_pixels)
    
    if test_type in ['t_test', 'all']:
        results['t_test'] = tester.t_test(real_pixels, fake_pixels)
    
    if test_type in ['mann_whitney', 'all']:
        results['mann_whitney'] = tester.mann_whitney_test(real_pixels, fake_pixels)
    
    # Return p-value from first test if single test requested
    if test_type != 'all' and results:
        first_result = list(results.values())[0]
        return first_result.get('p_value', None)
    
    return results

