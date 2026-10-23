"""
Data loading, dataset splitting, and augmentation utilities.
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess

DEFAULT_CLASSES = ["glioma", "meningioma", "pituitary", "normal"]


def split_dataset(
    src_root: str,
    dst_root: str,
    classes: List[str] = DEFAULT_CLASSES,
    train_r: float = 0.70,
    val_r: float = 0.15,
    test_r: float = 0.15,
    seed: int = 42
) -> None:
    """
    Performs stratified random train / val / test split.

    Args:
        src_root: Directory containing class subfolders.
        dst_root: Destination directory for 'train', 'val', 'test'.
        classes: List of class folder names.
        train_r: Training split ratio (default 0.70).
        val_r: Validation split ratio (default 0.15).
        test_r: Test split ratio (default 0.15).
        seed: Random seed for reproducibility.
    """
    assert abs((train_r + val_r + test_r) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    random.seed(seed)

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(dst_root, split, cls), exist_ok=True)

    print("Splitting dataset...")
    for cls in classes:
        src_dir = os.path.join(src_root, cls)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Class folder '{cls}' not found in '{src_root}'")

        images = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_r)
        n_val = int(n_total * val_r)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for img in img_list:
                shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_root, split_name, cls, img))

        print(f"  {cls:<12} -> Train: {len(train_imgs):>4} | Val: {len(val_imgs):>4} | Test: {len(test_imgs):>4}")
    print("Split completed successfully.\n")


def get_preprocess_fn(architecture: str):
    """Returns the matching Keras preprocess_input function."""
    arch = architecture.lower()
    if "densenet" in arch:
        return dense_preprocess
    elif "resnet" in arch:
        return resnet_preprocess
    else:
        return eff_preprocess


def get_data_generators(
    split_data_dir: str,
    img_size: Tuple[int, int] = (300, 300),
    batch_size: int = 32,
    architecture: str = "efficientnet",
    seed: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Creates ImageDataGenerators for training, validation, test (standard), and test (augmented for TTA).
    """
    preprocess_fn = get_preprocess_fn(architecture)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    # Augmented datagen for Test-Time Augmentation (TTA)
    tta_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(split_data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(split_data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(split_data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen, tta_datagen
