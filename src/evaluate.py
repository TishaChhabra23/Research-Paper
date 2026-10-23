"""
Evaluation utilities including Test-Time Augmentation (TTA), confusion matrix generation,
and comprehensive multi-class metrics computation.
"""

import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def predict_tta(
    model,
    test_generator,
    tta_datagen,
    steps: int = 8
) -> np.ndarray:
    """
    Computes ensemble predictions using Test-Time Augmentation (TTA).

    Args:
        model: Trained Keras model.
        test_generator: Standard test generator (un-augmented, shuffle=False).
        tta_datagen: ImageDataGenerator configured for subtle test augmentations.
        steps: Number of TTA passes (default 8).

    Returns:
        np.ndarray: Averaged probability predictions with shape (N, num_classes).
    """
    n_samples = test_generator.samples
    batch_size = test_generator.batch_size
    n_batches = int(np.ceil(n_samples / batch_size))

    # Pass 0: Standard unaugmented prediction
    test_generator.reset()
    base_probs = model.predict(test_generator, verbose=0)
    all_probs = [base_probs]

    print(f"Running TTA ({steps} passes)...")
    for pass_idx in range(1, steps):
        pass_probs = []
        test_generator.reset()
        for b_idx in range(n_batches):
            batch_x, _ = test_generator[b_idx]
            # Apply slight augmentation
            aug_x = np.empty_like(batch_x)
            for i in range(len(batch_x)):
                aug_x[i] = tta_datagen.random_transform(batch_x[i])
            pred_b = model.predict(aug_x, verbose=0)
            pass_probs.append(pred_b)
        all_probs.append(np.vstack(pass_probs))

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plots and saves a formatted confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=True
    )
    plt.title("Confusion Matrix (TTA Predictions)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_classification_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Generates a structured dictionary containing precision, recall, F1, and ROC-AUC.
    """
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = None

    report["macro_auc"] = macro_auc
    return report
