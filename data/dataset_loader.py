"""
Dataset loading utilities using tf.data API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from training.feature_generator import FeatureGenerator


class DatasetLoader:
    """Load and prepare datasets using tf.data API."""

    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

        dataset_cfg = self.config.get("dataset", {})
        self.image_size = tuple(dataset_cfg.get("image_size", [224, 224]))
        self.batch_size = int(dataset_cfg.get("batch_size", 32))

        self.feature_dim = (
            self.config.get("model", {})
            .get("feature_stream", {})
            .get("input_dim", 128)
        )
        self.feature_generator = FeatureGenerator(feature_dim=self.feature_dim)

    # --------------------------
    # Public API
    # --------------------------
    def load_dataset_from_directory(self, data_dir: str, split: str = "train") -> tf.data.Dataset:
        """
        Load a dataset from a directory structure:

            data_dir/
              real/*.jpg|png
              fake/*.jpg|png

        Returns:
            tf.data.Dataset yielding: ((image, features), label)
        """
        file_paths, labels = self._scan_real_fake_dir(Path(data_dir))
        return self._build_dataset(file_paths, labels, split=split)

    def create_train_val_test_split(
        self,
        data_dir: str,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        stratify: bool = True,
    ) -> Dict[str, tf.data.Dataset]:
        """
        Create train/val/test datasets with a deterministic (seeded) split.

        Splitting is done at the *file list* level to avoid incorrect splits caused by
        batching before `take/skip`.
        """
        total = train_split + val_split + test_split
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"train/val/test splits must sum to 1.0, got {total:.4f} "
                f"({train_split=}, {val_split=}, {test_split=})."
            )

        file_paths, labels = self._scan_real_fake_dir(Path(data_dir))
        if not file_paths:
            raise ValueError(
                f"No images found under '{data_dir}'. Expected 'real/' and 'fake/' subfolders."
            )

        rng = np.random.default_rng(seed)
        indices = np.arange(len(file_paths))

        if stratify:
            real_idx = indices[np.array(labels) == 0]
            fake_idx = indices[np.array(labels) == 1]
            rng.shuffle(real_idx)
            rng.shuffle(fake_idx)

            train_idx, val_idx, test_idx = [], [], []
            for class_idx in (real_idx, fake_idx):
                n = len(class_idx)
                n_train = int(train_split * n)
                n_val = int(val_split * n)

                train_idx.extend(class_idx[:n_train].tolist())
                val_idx.extend(class_idx[n_train : n_train + n_val].tolist())
                test_idx.extend(class_idx[n_train + n_val :].tolist())

            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)
        else:
            rng.shuffle(indices)
            n = len(indices)
            n_train = int(train_split * n)
            n_val = int(val_split * n)
            train_idx = indices[:n_train].tolist()
            val_idx = indices[n_train : n_train + n_val].tolist()
            test_idx = indices[n_train + n_val :].tolist()

        def select(idxs: List[int]) -> Tuple[List[str], List[int]]:
            return [file_paths[i] for i in idxs], [labels[i] for i in idxs]

        train_files, train_labels = select(train_idx)
        val_files, val_labels = select(val_idx)
        test_files, test_labels = select(test_idx)

        return {
            "train": self._build_dataset(train_files, train_labels, split="train"),
            "val": self._build_dataset(val_files, val_labels, split="val"),
            "test": self._build_dataset(test_files, test_labels, split="test"),
        }

    # --------------------------
    # Internals
    # --------------------------
    def _scan_real_fake_dir(self, data_dir: Path) -> Tuple[List[str], List[int]]:
        """Return (file_paths, labels) where labels are 0=real, 1=fake."""
        real_dir = data_dir / "real"
        fake_dir = data_dir / "fake"

        real_files: List[Path] = []
        fake_files: List[Path] = []

        if real_dir.exists():
            for ext in self.SUPPORTED_EXTS:
                real_files.extend(real_dir.glob(f"*{ext}"))
        if fake_dir.exists():
            for ext in self.SUPPORTED_EXTS:
                fake_files.extend(fake_dir.glob(f"*{ext}"))

        # Deterministic ordering before shuffling/splitting.
        real_files = sorted(real_files)
        fake_files = sorted(fake_files)

        file_paths = [str(p) for p in real_files] + [str(p) for p in fake_files]
        labels = [0] * len(real_files) + [1] * len(fake_files)

        return file_paths, labels

    def _build_dataset(self, file_paths: List[str], labels: List[int], split: str) -> tf.data.Dataset:
        """Build a tf.data dataset yielding ((image, features), label)."""
        file_paths_t = tf.convert_to_tensor(file_paths, dtype=tf.string)
        labels_t = tf.convert_to_tensor(labels, dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((file_paths_t, labels_t))

        ds = ds.map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

        if split == "train":
            buffer_size = min(len(file_paths), 10_000)
            if buffer_size > 1:
                ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _load_and_preprocess_image(
        self,
        file_path: tf.Tensor,
        label: tf.Tensor,
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Load an image file and extract explicit features."""
        file_path = tf.cast(file_path, tf.string)
        label = tf.cast(label, tf.int32)

        image_bytes = tf.io.read_file(file_path)
        image_uint8 = tf.image.decode_image(
            image_bytes, channels=3, expand_animations=False
        )

        # Model image input is float32 in [0, 1]
        image_float = tf.image.convert_image_dtype(image_uint8, tf.float32)
        image_float = tf.image.resize(image_float, self.image_size)

        # Explicit features are extracted in Python/NumPy (OpenCV + SciPy).
        # Match inference: compute features from the resized uint8 image.
        image_uint8 = tf.cast(
            tf.clip_by_value(image_float * 255.0, 0.0, 255.0),
            tf.uint8,
        )
        features = tf.numpy_function(
            func=self.feature_generator.extract,
            inp=[image_uint8],
            Tout=tf.float32,
        )
        features.set_shape([self.feature_dim])

        return (image_float, features), label
