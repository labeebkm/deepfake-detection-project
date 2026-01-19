"""
Verification script for the new feature engineering pipeline.
"""
import os
import tensorflow as tf
import numpy as np
import cv2
from training.feature_generator import FeatureGenerator
from models.model_factory import ModelFactory
from models.three_stream_net import ThreeStreamEfficientNet

def test_feature_generator():
    print("Testing FeatureGenerator...")
    # Create dummy image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    gen = FeatureGenerator(feature_dim=128)
    features = gen.extract(img)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector norm: {np.linalg.norm(features)}")
    
    assert features.shape == (128,)
    print("FeatureGenerator passed!")
    return features

def test_model_build():
    print("\nTesting ThreeStreamEfficientNet...")
    
    # Create dummy inputs
    batch_size = 4
    images = tf.random.uniform((batch_size, 224, 224, 3))
    features = tf.random.uniform((batch_size, 128))
    
    # Build model
    model = ThreeStreamEfficientNet(
        backbone_name='efficientnet-b4',
        num_classes=2,
        dropout_rate=0.5,
        feature_dim=128
    )
    
    # Run forward pass
    outputs = model((images, features), training=False)
    
    print(f"Output shape: {outputs.shape}")
    assert outputs.shape == (batch_size, 2)
    print("ThreeStreamEfficientNet passed!")

def main():
    try:
        test_feature_generator()
        test_model_build()
        print("\nAll systems go!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        # print(f"\n❌ Test failed: {e}")
        raise e

if __name__ == "__main__":
    main()
