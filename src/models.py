"""
Model architectures for Brain Tumor MRI classification.
Supports EfficientNet, DenseNet121, and ResNet50V2 with dual pooling,
batch normalization, and L2 regularization as proposed in our IEEE ICONICA 2026 paper.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB3, DenseNet121, ResNet50V2


def build_model(
    architecture: str = "efficientnet",
    num_classes: int = 4,
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    weights: str = "imagenet",
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.40
) -> Tuple[Model, Model]:
    """
    Builds a transfer learning model with dual pooling (GAP + GMP).

    Args:
        architecture (str): One of 'efficientnet', 'densenet', 'resnet'.
        num_classes (int): Number of target classes (default 4).
        input_shape (Tuple[int, int, int]): Dimensions of input image (H, W, C).
        weights (str): Pre-trained weights ('imagenet' or None).
        weight_decay (float): L2 regularization factor.
        dropout_rate (float): Dropout probability.

    Returns:
        Tuple[Model, Model]: (full_model, backbone_model)
    """
    arch = architecture.lower()
    if arch in ["efficientnet", "efficientnetb3", "efficientnetv2"]:
        backbone = EfficientNetB3(include_top=False, weights=weights, input_shape=input_shape)
        model_name = "EfficientNetB3_BrainTumor"
    elif arch in ["densenet", "densenet121"]:
        backbone = DenseNet121(include_top=False, weights=weights, input_shape=input_shape)
        model_name = "DenseNet121_BrainTumor"
    elif arch in ["resnet", "resnet50v2"]:
        backbone = ResNet50V2(include_top=False, weights=weights, input_shape=input_shape)
        model_name = "ResNet50V2_BrainTumor"
    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'. Choose from 'efficientnet', 'densenet', 'resnet'.")

    # Freeze backbone for Phase 1
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = backbone(inputs, training=False)

    # Dual Pooling: Global Average + Global Max Pooling
    avg_pool = layers.GlobalAveragePooling2D(name="gap")(x)
    max_pool = layers.GlobalMaxPooling2D(name="gmp")(x)
    x = layers.Concatenate(name="dual_pool")([avg_pool, max_pool])
    x = layers.BatchNormalization(name="bn_pool")(x)

    # Classification Head
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay),
        name="dense_256"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name=model_name)
    return model, backbone


def unfreeze_backbone(backbone: Model, num_layers_to_unfreeze: int = 50) -> None:
    """
    Unfreezes the top N layers of the backbone for Phase 2 fine-tuning.
    """
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
