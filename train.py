# """
# Main training script.
# """

# import argparse
# import os
# import yaml
# import tensorflow as tf
# from pathlib import Path

# from models.model_factory import ModelFactory
# from data.dataset_loader import DatasetLoader
# from data.augmentation import Augmentor
# from training.trainer import Trainer


# def main():
#     parser = argparse.ArgumentParser(description='Train deepfake detection model')
#     parser.add_argument('--config', type=str, default='configs/config.yaml',
#                        help='Path to configuration file')
#     parser.add_argument('--data_dir', type=str, required=True,
#                        help='Path to dataset directory')
#     parser.add_argument('--resume', type=str, default=None,
#                        help='Path to checkpoint to resume from')
    
#     args = parser.parse_args()
    
#     # Load configuration
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Set mixed precision if enabled
#     if config.get('training', {}).get('mixed_precision', False):
#         tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
#     # Load datasets
#     print("Loading datasets...")
#     loader = DatasetLoader(args.config)
#     datasets = loader.create_train_val_test_split(
#         args.data_dir,
#         train_split=config['dataset']['train_split'],
#         val_split=config['dataset']['val_split'],
#         test_split=config['dataset']['test_split']
#     )
    
#     # Apply augmentation to training set
#     if config.get('augmentation', {}).get('enabled', True):
#         augmentor = Augmentor(config.get('augmentation', {}))
#         datasets['train'] = datasets['train'].map(
#             lambda x, y: augmentor.augment_tf(x, y),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
    
#     # Create model
#     print("Creating model...")
#     model = ModelFactory.create_model(
#         config['model']['name'],
#         config['model']
#     )
    
#     # Resume from checkpoint if specified
#     if args.resume:
#         print(f"Loading checkpoint from {args.resume}...")
#         model.load_weights(args.resume)
    
#     # Initialize trainer
#     trainer = Trainer(args.config)
    
#     # Compile model
#     model = trainer.compile_model(model)
    
#     # Print model summary
#     model.summary()
    
#     # Train
#     print("Starting training...")
#     history = trainer.train(
#         model,
#         datasets['train'],
#         datasets['val']
#     )
    
#     # Save final model
#     save_path = os.path.join(config.get('paths', {}).get('checkpoints', './checkpoints'), 'final_model.h5')
#     model.save(save_path)
#     print(f"Model saved to {save_path}")


# if __name__ == '__main__':
#     main()



"""
Main training script.
"""

import argparse
import os
import yaml
import tensorflow as tf

from models.model_factory import ModelFactory
from data.dataset_loader import DatasetLoader
from data.augmentation import Augmentor
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Mixed precision (optional)
    # ------------------------------------------------------------------
    if config.get('training', {}).get('mixed_precision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    loader = DatasetLoader(args.config)

    datasets = loader.create_train_val_test_split(
        args.data_dir,
        train_split=config['dataset']['train_split'],
        val_split=config['dataset']['val_split'],
        test_split=config['dataset']['test_split']
    )

    # ------------------------------------------------------------------
    # Apply augmentation ONLY to image (FIXED PART)
    # ------------------------------------------------------------------
    if config.get('augmentation', {}).get('enabled', True):
        augmentor = Augmentor(config.get('augmentation', {}))

        def augment_only_image(inputs, label):
            """
            inputs: (image, feature_vector)
            label: class label
            """
            image, features = inputs          # unpack
            image, label = augmentor.augment_tf(image, label)
            return (image, features), label   # re-pack

        datasets['train'] = datasets['train'].map(
            augment_only_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    print("Creating model...")
    model = ModelFactory.create_model(
        config['model']['name'],
        config['model']
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        model.load_weights(args.resume)

    # ------------------------------------------------------------------
    # Initialize trainer
    # ------------------------------------------------------------------
    trainer = Trainer(args.config)

    # Compile model
    model = trainer.compile_model(model)

    # Print model summary
    model.summary()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("Starting training...")
    history = trainer.train(
        model,
        datasets['train'],
        datasets['val']
    )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    save_dir = config.get('paths', {}).get('checkpoints', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'final_model.h5')
    model.save(save_path)

    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()







