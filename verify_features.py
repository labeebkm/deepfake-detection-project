"""
Quick verification script for the feature engineering + three-stream model wiring.
"""

import numpy as np
import tensorflow as tf

from models.three_stream_net import ThreeStreamEfficientNet
from training.feature_generator import FeatureGenerator


def test_feature_generator(feature_dim: int = 128) -> np.ndarray:
    print("Testing FeatureGenerator...")
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    gen = FeatureGenerator(feature_dim=feature_dim)
    features = gen.extract(img)

    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector norm: {np.linalg.norm(features)}")

    assert features.shape == (feature_dim,)
    print("FeatureGenerator passed!")
    return features


def test_model_build(feature_dim: int = 128) -> None:
    print("\nTesting ThreeStreamEfficientNet...")

    batch_size = 4
    images = tf.random.uniform((batch_size, 224, 224, 3))
    features = tf.random.uniform((batch_size, feature_dim))

    model = ThreeStreamEfficientNet(
        backbone_name="efficientnet-b4",
        pretrained=False,  # keep verification fast/offline (no ImageNet weight download)
        num_classes=2,
        dropout_rate=0.5,
        feature_dim=feature_dim,
    )

    outputs = model((images, features), training=False)

    print(f"Output shape: {outputs.shape}")
    assert outputs.shape == (batch_size, 2)
    print("ThreeStreamEfficientNet passed!")


def main() -> None:
    try:
        test_feature_generator()
        test_model_build()
        print("\nAll systems go!")
    except Exception:
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

