import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from data.dataset_loader import DatasetLoader
from inference.detector import DeepfakeDetector
from models.model_factory import ModelFactory
from models.three_stream_net import ThreeStreamEfficientNet


def _write_random_images(dir_path: Path, prefix: str, n: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"{prefix}_{i:03d}.png")


def test_three_stream_forward_pass():
    model = ThreeStreamEfficientNet(
        backbone_name="efficientnet-b0",
        pretrained=False,  # keep unit tests offline/fast
        num_classes=2,
        dropout_rate=0.1,
        feature_dim=128,
        dct_size=8,
        num_filters=8,
    )

    images = tf.zeros([2, 224, 224, 3], dtype=tf.float32)
    features = tf.zeros([2, 128], dtype=tf.float32)
    out = model((images, features), training=False)

    assert out.shape == (2, 2)


def test_dataset_loader_split_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "real").mkdir(parents=True, exist_ok=True)
        (root / "fake").mkdir(parents=True, exist_ok=True)

        _write_random_images(root / "real", "real", 6)
        _write_random_images(root / "fake", "fake", 6)

        loader = DatasetLoader(None)
        splits = loader.create_train_val_test_split(
            str(root),
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            seed=123,
            stratify=True,
        )

        for name in ("train", "val", "test"):
            ds = splits[name]
            (imgs, feats), labels = next(iter(ds))
            assert imgs.dtype == tf.float32
            assert imgs.shape[-3:] == (224, 224, 3)
            assert feats.dtype == tf.float32
            assert feats.shape[-1] == 128
            assert labels.dtype == tf.int32


def test_detector_predict_smoke():
    # Build a tiny model, save weights, then verify DeepfakeDetector can reload + predict.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        weights_path = tmp_path / "model.weights.h5"
        config_path = tmp_path / "config.yaml"

        cfg = {
            "dataset": {"image_size": [224, 224], "batch_size": 2},
            "model": {
                "name": "three_stream_efficientnet",
                "backbone": "efficientnet-b0",
                "pretrained": False,
                "num_classes": 2,
                "dropout_rate": 0.1,
                "frequency_stream": {"dct_size": 8, "num_filters": 8},
                "feature_stream": {"input_dim": 128},
            },
        }

        import yaml

        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        model = ModelFactory.create_model(cfg["model"]["name"], cfg["model"])
        model((tf.zeros([1, 224, 224, 3]), tf.zeros([1, 128])), training=False)
        model.save_weights(str(weights_path))

        detector = DeepfakeDetector(
            weights_path=str(weights_path),
            config_path=str(config_path),
            confidence_threshold=0.5,
        )

        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = detector.predict(img)

        assert set(result.keys()) == {
            "is_fake",
            "confidence",
            "real_probability",
            "fake_probability",
        }
        assert 0.0 <= result["real_probability"] <= 1.0
        assert 0.0 <= result["fake_probability"] <= 1.0
