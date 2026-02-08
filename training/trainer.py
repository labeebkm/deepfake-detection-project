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
        lr = self.training_config.get('initial_learning_rate', 0.001)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        
        # Metrics
        metrics_config = self.config.get('metrics', ['accuracy'])
        metrics = get_metrics(metrics_config)
        
        # Compile
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
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
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.training_config.get('epochs', 100),
            callbacks=callbacks,
            verbose=1
        )
        
        return history

