"""
Visualization utilities for model explainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from typing import Optional, Union


def _prepare_image_tensor(preprocessed_image: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
    image_tensor = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)
    if image_tensor.shape.rank == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)
    if image_tensor.shape.rank != 4:
        raise ValueError(
            "preprocessed_image must have shape [H, W, C] or [B, H, W, C], "
            f"got rank={image_tensor.shape.rank}"
        )
    return image_tensor


def _prepare_feature_tensor(
    classifier_model: tf.keras.Model,
    feature_vector: Optional[Union[np.ndarray, tf.Tensor]],
    batch_size: tf.Tensor,
) -> tf.Tensor:
    if feature_vector is not None:
        feature_tensor = tf.convert_to_tensor(feature_vector, dtype=tf.float32)
        if feature_tensor.shape.rank == 1:
            feature_tensor = tf.expand_dims(feature_tensor, axis=0)
        return feature_tensor

    feature_dim = getattr(classifier_model, "feature_dim", None)
    if feature_dim is None:
        feature_dim = 128

    return tf.zeros((batch_size, int(feature_dim)), dtype=tf.float32)


def _normalize_cam(cam: tf.Tensor) -> np.ndarray:
    cam_np = cam.numpy().astype(np.float32)
    cam_np = np.maximum(cam_np, 0)
    denom = float(cam_np.max())
    if denom > 0:
        cam_np /= denom
    cam_np = cv2.resize(cam_np, (380, 380), interpolation=cv2.INTER_LINEAR)
    return np.clip(cam_np, 0.0, 1.0).astype(np.float32)


def generate_gradcam(
    model_backbone: tf.keras.Model,
    preprocessed_image: Union[np.ndarray, tf.Tensor],
    layer_name: str = "top_conv",
    *,
    classifier_model: Optional[tf.keras.Model] = None,
    feature_vector: Optional[Union[np.ndarray, tf.Tensor]] = None,
    class_idx: Optional[int] = 1,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap.

    Args:
        model_backbone: Backbone model containing the target conv layer (e.g., EfficientNet).
        preprocessed_image: Preprocessed image tensor/array, shape [H, W, C] or [B, H, W, C].
        layer_name: Target convolutional layer name.
        classifier_model: Full classifier model used for final class probabilities.
        feature_vector: Optional second input for multi-input models.
        class_idx: Target class index for Grad-CAM. If None, uses predicted class.

    Returns:
        Normalized Grad-CAM heatmap as float32 array in [0, 1], resized to 380x380.
    """
    classifier_model = classifier_model or model_backbone
    image_tensor = _prepare_image_tensor(preprocessed_image)

    try:
        target_layer = model_backbone.get_layer(layer_name)
    except Exception as exc:
        raise ValueError(
            f"Layer '{layer_name}' not found in backbone '{model_backbone.name}'."
        ) from exc

    # Path 1: Three-stream subclassed model (no stable `inputs` attribute).
    # We compute predictions through the same forward path while exposing target conv activations.
    if hasattr(classifier_model, "rgb_backbone") and classifier_model.rgb_backbone is model_backbone:
        feature_tensor = _prepare_feature_tensor(
            classifier_model=classifier_model,
            feature_vector=feature_vector,
            batch_size=tf.shape(image_tensor)[0],
        )

        # Extract both target conv activations and final RGB backbone output in one pass.
        rgb_extractor = tf.keras.Model(
            inputs=model_backbone.inputs,
            outputs=[target_layer.output, model_backbone.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, rgb_backbone_output = rgb_extractor(image_tensor, training=False)

            # Rebuild forward pass from RGB backbone output onward.
            x_rgb = classifier_model.rgb_pool(rgb_backbone_output)
            x_rgb = classifier_model.rgb_proj(x_rgb)

            x_freq = classifier_model.frequency_net(image_tensor, training=False)
            x_freq = classifier_model.freq_pool(x_freq)
            x_freq = classifier_model.freq_proj(x_freq)

            x_feat = classifier_model.feature_net(feature_tensor, training=False)
            x_feat = classifier_model.feat_proj(x_feat)

            rgb_expanded = tf.expand_dims(x_rgb, 1)
            freq_expanded = tf.expand_dims(x_freq, 1)
            feat_expanded = tf.expand_dims(x_feat, 1)

            stacked = classifier_model.concat([rgb_expanded, freq_expanded, feat_expanded])
            attended = classifier_model.attention([stacked, stacked, stacked], training=False)
            fused = classifier_model.flatten(attended)
            x = classifier_model.dropout(fused, training=False)
            predictions = classifier_model.classifier(x)

            target_class = int(tf.argmax(predictions[0]).numpy()) if class_idx is None else int(class_idx)
            loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradients are None. Check model connectivity and target layer.")

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
        return _normalize_cam(cam)

    # Path 2: Functional/sequential models where inputs are available.
    model_inputs = getattr(classifier_model, "inputs", None)
    model_output = getattr(classifier_model, "output", None)
    if model_inputs is not None and model_output is not None:
        grad_model = tf.keras.models.Model(
            inputs=model_inputs,
            outputs=[target_layer.output, model_output],
        )

        feed_inputs = image_tensor
        num_inputs = len(model_inputs) if isinstance(model_inputs, list) else 1
        if num_inputs > 1:
            feature_tensor = _prepare_feature_tensor(
                classifier_model=classifier_model,
                feature_vector=feature_vector,
                batch_size=tf.shape(image_tensor)[0],
            )
            feed_inputs = [image_tensor, feature_tensor]

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(feed_inputs, training=False)
            target_class = int(tf.argmax(predictions[0]).numpy()) if class_idx is None else int(class_idx)
            loss = predictions[:, target_class]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradients are None. Check model connectivity and target layer.")

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
        return _normalize_cam(cam)

    raise ValueError(
        "Unable to build Grad-CAM graph for this model type. "
        "Pass a functional model, or pass classifier_model with rgb_backbone path."
    )


def overlay_heatmap(original_image_numpy: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on top of an RGB image.

    Args:
        original_image_numpy: Original RGB image.
        cam: Grad-CAM heatmap in [0, 1].

    Returns:
        RGB uint8 overlay image.
    """
    if original_image_numpy is None:
        raise ValueError("original_image_numpy must not be None")

    image = np.asarray(original_image_numpy)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}")

    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.resize(image, (380, 380), interpolation=cv2.INTER_AREA)

    cam_norm = np.asarray(cam, dtype=np.float32)
    cam_norm = np.clip(cam_norm, 0.0, 1.0)
    cam_u8 = np.uint8(255 * cam_norm)

    heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.6, heatmap_rgb, 0.4, 0)
    return overlay


def plot_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    layer_name: str,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Backward-compatible Grad-CAM helper.
    """
    target_class = 1 if class_idx is None else int(class_idx)
    return generate_gradcam(
        model_backbone=model,
        preprocessed_image=image,
        layer_name=layer_name,
        classifier_model=model,
        class_idx=target_class,
    )


def plot_attention_maps(attention_weights: np.ndarray, image: np.ndarray) -> plt.Figure:
    """
    Plot attention maps.

    Args:
        attention_weights: Attention weights
        image: Input image

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(attention_weights, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    return fig
