"""
Model evaluation script.
"""

import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.model_factory import ModelFactory
from data.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test dataset
    print("Loading test dataset...")
    loader = DatasetLoader(args.config)
    test_dataset = loader.load_dataset_from_directory(args.data_dir, split='test')
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ModelFactory.load_model(args.model_path)
    
    # Evaluate
    print("Evaluating model...")
    results = model.evaluate(test_dataset, verbose=1)
    
    # Get predictions
    print("Generating predictions...")
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    print(f"\nConfusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    
    # Save results
    results_dict = {
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'confusion_matrix': cm.tolist()
    }
    
    import json
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nEvaluation results saved to {args.output_dir}")


if __name__ == '__main__':
    main()







