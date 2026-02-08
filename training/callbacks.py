"""
Training callbacks.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, List
import os


def get_callbacks(config: Dict, output_dir: str = "./logs") -> List[tf.keras.callbacks.Callback]:
    """
    Get training callbacks based on configuration.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for logs and checkpoints
        
    Returns:
        List of callbacks
    """
    callbacks = []
    callbacks_config = config.get('callbacks', {})
    
    # TensorBoard
    if callbacks_config.get('tensorboard', {}).get('enabled', True):
        log_dir = callbacks_config.get('tensorboard', {}).get('log_dir', os.path.join(output_dir, 'tensorboard'))
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
    
    # Model checkpoint
    if callbacks_config.get('model_checkpoint', {}).get('enabled', True):
        checkpoint_config = callbacks_config.get('model_checkpoint', {})
        save_dir = checkpoint_config.get('save_dir', os.path.join(output_dir, 'checkpoints'))
        os.makedirs(save_dir, exist_ok=True)
        
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                # Save weights only (Keras 3 requires .weights.h5 for H5 weight files).
                filepath=os.path.join(save_dir, "best_model.weights.h5"),
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_weights_only=True,
                mode='min',
                verbose=1
            )
        )
    
    # Early stopping
    early_stopping_config = callbacks_config.get("early_stopping")
    if early_stopping_config is None:
        # Backward-compatible: allow early_stopping under training config.
        early_stopping_config = config.get("training", {}).get("early_stopping", {})
    if early_stopping_config.get("enabled", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_loss"),
                patience=early_stopping_config.get("patience", 15),
                restore_best_weights=True,
                verbose=1
            )
        )
    
    # Reduce learning rate on plateau
    if callbacks_config.get('reduce_lr', {}).get('enabled', True):
        reduce_lr_config = callbacks_config.get('reduce_lr', {})
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_lr_config.get('factor', 0.5),
                patience=reduce_lr_config.get('patience', 5),
                min_lr=reduce_lr_config.get('min_lr', 1e-7),
                verbose=1
            )
        )
    
    return callbacks

