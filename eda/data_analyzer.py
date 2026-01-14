"""
Statistical analysis functions for deepfake detection datasets.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import tensorflow as tf
from PIL import Image
import cv2
from tqdm import tqdm


class DataAnalyzer:
    """Comprehensive data analysis for deepfake detection datasets."""
    
    def __init__(self, dataset_path: str, config: Optional[Dict] = None):
        """
        Initialize DataAnalyzer.
        
        Args:
            dataset_path: Path to dataset root directory
            config: Optional configuration dictionary
        """
        self.dataset_path = Path(dataset_path)
        self.config = config or {}
        self.stats = {}
        
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """
        Analyze dataset directory structure and organization.
        
        Returns:
            Dictionary with dataset structure information
        """
        structure = {
            "total_images": 0,
            "classes": {},
            "splits": {},
            "methods": {},
            "compressions": {},
            "file_formats": Counter(),
            "corrupted_files": []
        }
        
        # Traverse dataset directory
        for root, dirs, files in os.walk(self.dataset_path):
            # Check for class labels (real/fake)
            if "real" in root.lower() or "fake" in root.lower():
                class_name = "real" if "real" in root.lower() else "fake"
                if class_name not in structure["classes"]:
                    structure["classes"][class_name] = 0
                
                # Count images
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                structure["classes"][class_name] += len(image_files)
                structure["total_images"] += len(image_files)
                
                # Check file formats
                for img_file in image_files:
                    ext = Path(img_file).suffix.lower()
                    structure["file_formats"][ext] += 1
        
        return structure
    
    def analyze_image_statistics(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze image statistics (resolution, aspect ratio, color channels).
        
        Args:
            sample_size: Number of images to sample (None for all)
            
        Returns:
            Dictionary with image statistics
        """
        resolutions = []
        aspect_ratios = []
        color_channels = []
        file_sizes = []
        
        image_files = self._get_all_image_files()
        if sample_size:
            image_files = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)
        
        print(f"Analyzing {len(image_files)} images...")
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Get file size
                file_size = os.path.getsize(img_path)
                file_sizes.append(file_size)
                
                # Load image
                img = Image.open(img_path)
                width, height = img.size
                resolutions.append((width, height))
                aspect_ratios.append(width / height if height > 0 else 0)
                
                # Check color channels
                if img.mode == 'RGB':
                    color_channels.append(3)
                elif img.mode == 'L':
                    color_channels.append(1)
                else:
                    color_channels.append(len(img.mode))
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        stats = {
            "resolutions": {
                "mean": np.mean([r[0] for r in resolutions]),  # width
                "std": np.std([r[0] for r in resolutions]),
                "min": (min([r[0] for r in resolutions]), min([r[1] for r in resolutions])),
                "max": (max([r[0] for r in resolutions]), max([r[1] for r in resolutions])),
                "distribution": resolutions
            },
            "aspect_ratios": {
                "mean": np.mean(aspect_ratios),
                "std": np.std(aspect_ratios),
                "min": np.min(aspect_ratios),
                "max": np.max(aspect_ratios),
                "distribution": aspect_ratios
            },
            "color_channels": {
                "distribution": Counter(color_channels),
                "most_common": Counter(color_channels).most_common(1)[0][0]
            },
            "file_sizes": {
                "mean": np.mean(file_sizes),
                "std": np.std(file_sizes),
                "min": np.min(file_sizes),
                "max": np.max(file_sizes),
                "distribution": file_sizes
            }
        }
        
        return stats
    
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """
        Analyze class distribution (real vs fake).
        
        Returns:
            Dictionary with class distribution statistics
        """
        class_counts = {"real": 0, "fake": 0}
        class_by_method = {}
        class_by_compression = {}
        
        for root, dirs, files in os.walk(self.dataset_path):
            # Determine class
            is_real = "real" in root.lower()
            is_fake = "fake" in root.lower()
            
            if not (is_real or is_fake):
                continue
            
            class_name = "real" if is_real else "fake"
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            count = len(image_files)
            
            class_counts[class_name] += count
            
            # Extract method if present in path
            path_parts = Path(root).parts
            for part in path_parts:
                if any(method.lower() in part.lower() for method in ["deepfakes", "face2face", "faceswap", "neuraltextures"]):
                    method = part
                    if method not in class_by_method:
                        class_by_method[method] = {"real": 0, "fake": 0}
                    class_by_method[method][class_name] += count
                    break
            
            # Extract compression level
            for part in path_parts:
                if part.startswith("c") and part[1:].isdigit():
                    compression = part
                    if compression not in class_by_compression:
                        class_by_compression[compression] = {"real": 0, "fake": 0}
                    class_by_compression[compression][class_name] += count
                    break
        
        total = sum(class_counts.values())
        distribution = {
            "counts": class_counts,
            "percentages": {k: (v / total * 100) if total > 0 else 0 for k, v in class_counts.items()},
            "total": total,
            "imbalance_ratio": max(class_counts.values()) / min(class_counts.values()) if min(class_counts.values()) > 0 else float('inf'),
            "by_method": class_by_method,
            "by_compression": class_by_compression
        }
        
        return distribution
    
    def analyze_face_detection_success(self, face_detector) -> Dict[str, Any]:
        """
        Analyze face detection success rate.
        
        Args:
            face_detector: Face detection function
            
        Returns:
            Dictionary with face detection statistics
        """
        image_files = self._get_all_image_files()
        sample_size = min(1000, len(image_files))
        sampled_files = np.random.choice(image_files, sample_size, replace=False)
        
        success_count = 0
        failed_files = []
        bounding_box_stats = []
        
        print(f"Testing face detection on {sample_size} images...")
        for img_path in tqdm(sampled_files, desc="Face detection"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                faces = face_detector(img)
                if faces and len(faces) > 0:
                    success_count += 1
                    # Get bounding box statistics
                    for face in faces:
                        if isinstance(face, dict) and 'box' in face:
                            bbox = face['box']
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            area = width * height
                            bounding_box_stats.append({
                                "width": width,
                                "height": height,
                                "area": area,
                                "aspect_ratio": width / height if height > 0 else 0
                            })
                else:
                    failed_files.append(str(img_path))
            except Exception as e:
                failed_files.append(str(img_path))
                continue
        
        stats = {
            "total_tested": sample_size,
            "success_count": success_count,
            "success_rate": success_count / sample_size if sample_size > 0 else 0,
            "failed_files": failed_files[:10],  # Store first 10 failures
            "bounding_box_stats": {
                "mean_area": np.mean([b["area"] for b in bounding_box_stats]) if bounding_box_stats else 0,
                "mean_width": np.mean([b["width"] for b in bounding_box_stats]) if bounding_box_stats else 0,
                "mean_height": np.mean([b["height"] for b in bounding_box_stats]) if bounding_box_stats else 0,
            } if bounding_box_stats else {}
        }
        
        return stats
    
    def calculate_dataset_quality_score(self) -> Dict[str, Any]:
        """
        Calculate overall dataset quality score.
        
        Returns:
            Dictionary with quality metrics
        """
        structure = self.analyze_dataset_structure()
        class_dist = self.analyze_class_distribution()
        img_stats = self.analyze_image_statistics(sample_size=1000)
        
        # Quality factors
        factors = {
            "class_balance": 1.0 / (1.0 + abs(0.5 - class_dist["percentages"].get("real", 50) / 100)),
            "dataset_size": min(1.0, structure["total_images"] / 10000),  # Normalize to 10k images
            "resolution_consistency": 1.0 - (img_stats["resolutions"]["std"] / img_stats["resolutions"]["mean"]) if img_stats["resolutions"]["mean"] > 0 else 0,
            "format_consistency": len(structure["file_formats"]) == 1,  # Prefer single format
        }
        
        # Overall quality score (weighted average)
        weights = {
            "class_balance": 0.3,
            "dataset_size": 0.2,
            "resolution_consistency": 0.3,
            "format_consistency": 0.2
        }
        
        quality_score = sum(factors[k] * weights[k] for k in factors)
        
        return {
            "overall_score": quality_score,
            "factors": factors,
            "weights": weights,
            "recommendations": self._generate_quality_recommendations(factors, class_dist, structure)
        }
    
    def _get_all_image_files(self) -> List[Path]:
        """Get all image files in dataset."""
        image_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(Path(root) / file)
        return image_files
    
    def _generate_quality_recommendations(self, factors: Dict, class_dist: Dict, structure: Dict) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        if factors["class_balance"] < 0.7:
            recommendations.append("Dataset has class imbalance. Consider using class weights or data augmentation.")
        
        if factors["dataset_size"] < 0.5:
            recommendations.append("Dataset size is relatively small. Consider data augmentation or collecting more data.")
        
        if factors["resolution_consistency"] < 0.7:
            recommendations.append("Image resolutions vary significantly. Consider standardizing image sizes.")
        
        if not factors["format_consistency"]:
            recommendations.append("Multiple file formats detected. Consider converting to a single format (e.g., JPEG).")
        
        if structure["corrupted_files"]:
            recommendations.append(f"Found {len(structure['corrupted_files'])} corrupted files. Review and remove them.")
        
        return recommendations
    
    def save_statistics(self, output_path: str):
        """Save all statistics to JSON file."""
        stats = {
            "dataset_structure": self.analyze_dataset_structure(),
            "class_distribution": self.analyze_class_distribution(),
            "image_statistics": self.analyze_image_statistics(sample_size=1000),
            "quality_score": self.calculate_dataset_quality_score()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, Counter):
                return dict(obj)
            return obj
        
        stats = convert_to_serializable(stats)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {output_path}")


class DataQualityChecker:
    """
    Data quality assessment and checking utilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataQualityChecker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.analyzer = None
    
    def assess_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Assess dataset quality and return quality score.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with quality assessment results
        """
        # Initialize analyzer if needed
        if self.analyzer is None or self.analyzer.dataset_path != Path(dataset_path):
            self.analyzer = DataAnalyzer(dataset_path, self.config)
        
        # Calculate quality score
        quality_results = self.analyzer.calculate_dataset_quality_score()
        
        # Additional quality checks
        structure = self.analyzer.analyze_dataset_structure()
        class_dist = self.analyzer.analyze_class_distribution()
        
        # Compile assessment
        assessment = {
            'overall_score': quality_results['overall_score'],
            'quality_factors': quality_results['factors'],
            'recommendations': quality_results['recommendations'],
            'dataset_size': structure['total_images'],
            'class_balance': class_dist['imbalance_ratio'],
            'file_formats': dict(structure['file_formats']),
            'corrupted_files': len(structure['corrupted_files']),
            'quality_level': self._get_quality_level(quality_results['overall_score'])
        }
        
        return assessment
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description from score."""
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'

