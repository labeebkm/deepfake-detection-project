"""
Training utilities.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional
import yaml
from .losses import FocalLoss
from .callbacks import get_callbacks
from .metrics import get_metrics


class Trainer:
    """Training manager."""
    
    def __init__(self, config_path: str):
        """
        Initialize Trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config.get('training', {})
    
    def compile_model(self, model: keras.Model) -> keras.Model:
        """
        Compile model with loss, optimizer, and metrics.
        
        Args:
            model: Model to compile
            
        Returns:
            Compiled model
        """
        # Loss
        loss_config = self.config.get('loss', {})
        if loss_config.get('name') == 'focal_loss':
            loss = FocalLoss(
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0),
                label_smoothing=loss_config.get('label_smoothing', 0.1)
            )
        else:
            loss = 'sparse_categorical_crossentropy'
        
        # Optimizer
        lr = float(self.training_config.get('initial_learning_rate', 0.001))
        clipnorm = self.training_config.get("gradient_clip_norm")
        if clipnorm is not None:
            clipnorm = float(clipnorm)
            if clipnorm <= 0:
                clipnorm = None

        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        
        # Metrics
        metrics_config = self.config.get('metrics', ['accuracy'])
        metrics = get_metrics(metrics_config)
        
        # Compile
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model

    def _resolve_class_weight(self, train_dataset: tf.data.Dataset):
        """
        Resolve class weights from config, or compute them from the dataset when enabled.
        """
        class_weights_cfg = self.training_config.get("class_weights", {})
        if not class_weights_cfg.get("enabled", False):
            return None

        real_weight = class_weights_cfg.get("real")
        fake_weight = class_weights_cfg.get("fake")
        if real_weight is not None and fake_weight is not None:
            real_weight = float(real_weight)
            fake_weight = float(fake_weight)
            # Treat default placeholder weights (1.0/1.0) as "auto" mode.
            if not (real_weight == 1.0 and fake_weight == 1.0):
                return {
                    0: real_weight,
                    1: fake_weight,
                }

        real_count = 0
        fake_count = 0
        for _, labels in train_dataset:
            labels_np = labels.numpy().astype("int32").ravel()
            real_count += int((labels_np == 0).sum())
            fake_count += int((labels_np == 1).sum())

        if real_count == 0 or fake_count == 0:
            return None

        total = real_count + fake_count
        return {
            0: float(total / (2.0 * real_count)),
            1: float(total / (2.0 * fake_count)),
        }
    
    def train(self, model: keras.Model, train_dataset: tf.data.Dataset, 
             val_dataset: Optional[tf.data.Dataset] = None) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training history
        """
        # Get callbacks
        callbacks = get_callbacks(self.config, output_dir=self.config.get('paths', {}).get('logs', './logs'))
        
        # Mixed precision
        if self.training_config.get('mixed_precision', False):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        class_weight = self._resolve_class_weight(train_dataset)
        if class_weight is not None:
            print(f"Using class_weight={class_weight}")
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.training_config.get('epochs', 100),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return history

