"""
Dataset loading utilities using tf.data API.
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import yaml


class DatasetLoader:
    """Load and prepare datasets using tf.data API."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DatasetLoader.
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.dataset_config = self.config.get('dataset', {})
        self.image_size = tuple(self.dataset_config.get('image_size', [224, 224]))
        self.batch_size = self.dataset_config.get('batch_size', 32)
    
    def load_dataset_from_directory(self, data_dir: str, split: str = 'train') -> tf.data.Dataset:
        """
        Load dataset from directory structure.
        
        Expected structure:
        data_dir/
          real/
            image1.jpg
            image2.jpg
          fake/
            image1.jpg
            image2.jpg
        
        Args:
            data_dir: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            tf.data.Dataset
        """
        data_dir = Path(data_dir)
        
        # Get image files and labels
        image_files = []
        labels = []
        
        # Real images (label 0)
        real_dir = data_dir / 'real'
        if real_dir.exists():
            for img_file in real_dir.glob('*.jpg'):
                image_files.append(str(img_file))
                labels.append(0)
            for img_file in real_dir.glob('*.png'):
                image_files.append(str(img_file))
                labels.append(0)
        
        # Fake images (label 1)
        fake_dir = data_dir / 'fake'
        if fake_dir.exists():
            for img_file in fake_dir.glob('*.jpg'):
                image_files.append(str(img_file))
                labels.append(1)
            for img_file in fake_dir.glob('*.png'):
                image_files.append(str(img_file))
                labels.append(1)
        
        # Create dataset with explicit dtypes
        image_files = tf.convert_to_tensor(image_files, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
        
        # Map to load and preprocess images
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache, shuffle, batch, prefetch
        dataset = dataset.cache(f"cache/{split}.cache")
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _load_and_preprocess_image(self, file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess a single image.
        
        Args:
            file_path: Path to image file
            label: Image label
            
        Returns:
            Tuple of (image, label)
        """
        # Ensure correct dtypes
        file_path = tf.cast(file_path, tf.string)
        label = tf.cast(label, tf.int32)

        # Read image
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # Resize
        image = tf.image.resize(image, self.image_size)
        
        return image, label
    
    def create_train_val_test_split(self, data_dir: str, 
                                    train_split: float = 0.7,
                                    val_split: float = 0.15,
                                    test_split: float = 0.15) -> Dict[str, tf.data.Dataset]:
        """
        Create train/val/test splits.
        
        Args:
            data_dir: Root directory of dataset
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            
        Returns:
            Dictionary with 'train', 'val', 'test' datasets
        """
        # Load full dataset
        full_dataset = self.load_dataset_from_directory(data_dir, split='full')
        
        # Calculate split sizes
        dataset_size = sum(1 for _ in full_dataset)
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        
        # Split dataset
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size).take(val_size)
        test_dataset = full_dataset.skip(train_size + val_size)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    def load_from_tfrecord(self, tfrecord_path: str) -> tf.data.Dataset:
        """
        Load dataset from TFRecord files.
        
        Args:
            tfrecord_path: Path to TFRecord file or directory
            
        Returns:
            tf.data.Dataset
        """
        tfrecord_path = Path(tfrecord_path)
        
        if tfrecord_path.is_file():
            filenames = [str(tfrecord_path)]
        else:
            filenames = [str(f) for f in tfrecord_path.glob('*.tfrecord')]
        
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_tfrecord_example)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _parse_tfrecord_example(self, example_proto) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Parse TFRecord example.
        
        Args:
            example_proto: TFRecord example
            
        Returns:
            Tuple of (image, label)
        """
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_image(example['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)
        label = example['label']
        
        return image, label


