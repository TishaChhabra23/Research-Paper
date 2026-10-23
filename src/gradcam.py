"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visual interpretability utilities.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DEFAULT_CLASSES = ["glioma", "meningioma", "pituitary", "normal"]


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for the given image batch using the model's backbone features.

    Args:
        img_array: Preprocessed image batch with shape (1, H, W, C).
        model: Trained Keras model with backbone at model.layers[1] and custom head.
        pred_index: Index of class to compute gradients for (defaults to highest predicted class).

    Returns:
        np.ndarray: Normalized heatmap with values in [0, 1].
    """
    backbone = model.layers[1]

    l_gap = model.get_layer("gap")
    l_gmp = model.get_layer("gmp")
    l_concat = model.get_layer("dual_pool")
    l_bn = model.get_layer("bn_pool")
    l_dense = model.get_layer("dense_256")
    l_drop = model.get_layer("drop_dense")
    l_out = model.get_layer("predictions")

    with tf.GradientTape() as tape:
        conv_outputs = backbone(img_array, training=False)
        tape.watch(conv_outputs)

        x = l_bn(l_concat([l_gap(conv_outputs), l_gmp(conv_outputs)]), training=False)
        x = l_drop(l_dense(x), training=False)
        predictions = l_out(x)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap_rel = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-9)
    return heatmap_rel.numpy()


def overlay_gradcam(
    img_path: str,
    model: keras.Model,
    img_size: Tuple[int, int] = (300, 300),
    class_names: List[str] = DEFAULT_CLASSES,
    save_path: Optional[str] = None,
    alpha: float = 0.45,
    preprocess_fn = None
) -> Tuple[np.ndarray, str, float]:
    """
    Generates and optionally saves a 3-panel visualization (Original, Grad-CAM, Overlay).

    Args:
        img_path: Path to target MRI image file.
        model: Trained Keras model.
        img_size: (Height, Width) input size.
        class_names: List of class labels.
        save_path: Optional file path to save figure.
        alpha: Heatmap blend ratio (alpha * heatmap + (1 - alpha) * original).
        preprocess_fn: Optional preprocessing function.

    Returns:
        Tuple[np.ndarray, str, float]: (overlay_image, predicted_class, confidence)
    """
    img = load_img(img_path, target_size=img_size)
    raw_arr = img_to_array(img)
    if preprocess_fn:
        proc_arr = preprocess_fn(raw_arr.copy())
    else:
        proc_arr = raw_arr.copy() / 255.0
    img_batch = np.expand_dims(proc_arr, axis=0)

    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    pred_prob = float(preds[pred_idx])

    heatmap = make_gradcam_heatmap(img_batch, model, pred_index=pred_idx)
    heatmap_resized = cv2.resize(heatmap, (img_size[1], img_size[0]))
    jet_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB) / 255.0

    orig_norm = raw_arr / 255.0
    overlay = np.clip(alpha * jet_heatmap + (1.0 - alpha) * orig_norm, 0, 1)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, im, t in zip(axes, [orig_norm, jet_heatmap, overlay], ["Original MRI", "Grad-CAM Heatmap", "Overlay"]):
            ax.imshow(im)
            ax.set_title(t, fontsize=12, fontweight="bold")
            ax.axis("off")

        true_folder = os.path.basename(os.path.dirname(img_path))
        plt.suptitle(
            f"Predicted: {pred_class.upper()} ({pred_prob*100:.2f}%) | Reference: {true_folder.upper()}",
            fontweight="bold",
            fontsize=13
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return overlay, pred_class, pred_prob
