"""
Standalone Command-Line Interface (CLI) for Brain Tumor MRI inference and Grad-CAM visualization.

Usage:
    python src/predict.py --image path/to/mri_scan.jpg --model weights.keras
    python src/predict.py --image path/to/mri_scan.jpg --architecture efficientnet --gradcam
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.models import build_model
from src.data_loader import get_preprocess_fn, DEFAULT_CLASSES
from src.gradcam import overlay_gradcam

BAR_COLORS = ["#e63946", "#f4a261", "#9d4edd", "#457b9d"]


def run_inference(
    image_path: str,
    model_path: str = None,
    architecture: str = "efficientnet",
    img_size: tuple = (300, 300),
    classes: list = DEFAULT_CLASSES,
    show_gradcam: bool = False,
    output_dir: str = "./outputs"
):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    preprocess_fn = get_preprocess_fn(architecture)

    print(f"Loading model ({architecture})...")
    if model_path and os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        print(f"No checkpoint provided. Instantiating architecture '{architecture}' with ImageNet initialization.")
        model, _ = build_model(architecture=architecture, num_classes=len(classes), input_shape=(*img_size, 3))

    img = load_img(image_path, target_size=img_size)
    raw_arr = img_to_array(img)
    proc_arr = preprocess_fn(raw_arr.copy())
    batch = np.expand_dims(proc_arr, axis=0)

    print("Running inference...")
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    print("\n" + "=" * 45)
    print("       BRAIN TUMOR MRI PREDICTION")
    print("=" * 45)
    print(f"Top Prediction : {pred_class.upper()} ({confidence * 100:.2f}%)")
    print("-" * 45)
    print("Class Probabilities:")
    for cls_name, p in zip(classes, probs):
        bar = "█" * int(p * 30)
        print(f"  {cls_name:<12} : {p*100:>6.2f}% | {bar}")
    print("=" * 45 + "\n")

    # Plot confidence bar
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax_img.imshow(raw_arr / 255.0)
    ax_img.set_title(f"Predicted: {pred_class.upper()} ({confidence*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    y_pos = np.arange(len(classes))
    bars = ax_bar.barh(y_pos, probs, color=BAR_COLORS, height=0.55)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([c.upper() for c in classes], fontsize=10, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Confidence", fontsize=10)
    ax_bar.set_title("Class Probabilities", fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="x", alpha=0.3)

    for bar, p in zip(bars, probs):
        ax_bar.text(min(p + 0.02, 0.90), bar.get_y() + bar.get_height() / 2, f"{p*100:.1f}%", va="center", fontweight="bold")

    plot_path = os.path.join(output_dir, "inference_result.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Inference plot saved to: {plot_path}")

    if show_gradcam:
        gradcam_path = os.path.join(output_dir, "gradcam_result.png")
        try:
            overlay_gradcam(
                img_path=image_path,
                model=model,
                img_size=img_size,
                class_names=classes,
                save_path=gradcam_path,
                preprocess_fn=preprocess_fn
            )
            print(f"Grad-CAM visualization saved to: {gradcam_path}")
        except Exception as e:
            print(f"Grad-CAM could not be computed on this layer configuration: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Class Brain Tumor MRI Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI image file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .keras model weights")
    parser.add_argument("--architecture", type=str, default="efficientnet", choices=["efficientnet", "densenet", "resnet"])
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM interpretability overlay")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save generated plots")

    args = parser.parse_args()
    run_inference(
        image_path=args.image,
        model_path=args.model,
        architecture=args.architecture,
        show_gradcam=args.gradcam,
        output_dir=args.output_dir
    )
