from pathlib import Path
import json
import random
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "merged_dataset"
ANALYSIS_DIR = DATASET_DIR / "analysis"

RGB_DIR = DATASET_DIR / "rgb"
LABELS_DIR = DATASET_DIR / "labels"

SPLIT_INFO_JSON = DATASET_DIR / "split_info.json"
CLASS_MAPPING_JSON = DATASET_DIR / "class_mapping.json"


def ensure_analysis_dir():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_rgb_file(image_id: str):
    matches = sorted(RGB_DIR.glob(f"{image_id}.*"))
    if not matches:
        return None
    return matches[0]


def load_yolo_labels(label_path: Path):
    labels = []

    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_idx, x_center, y_center, width, height = parts

            labels.append(
                {
                    "class_idx": int(float(class_idx)),
                    "x_center": float(x_center),
                    "y_center": float(y_center),
                    "width": float(width),
                    "height": float(height),
                }
            )

    return labels


def get_all_image_ids_from_split():
    split_info = load_json(SPLIT_INFO_JSON)

    image_ids = []

    for split in ("train", "val", "test"):
        image_ids.extend(split_info.get(split, []))

    return image_ids, split_info


def yolo_to_pixel_bbox(label, image_width: int, image_height: int):
    x_center = label["x_center"] * image_width
    y_center = label["y_center"] * image_height
    width = label["width"] * image_width
    height = label["height"] * image_height

    x = x_center - width / 2
    y = y_center - height / 2

    return x, y, width, height


def visualize_bbox_samples(num_samples: int = 10, seed: int = 42):
    ensure_analysis_dir()

    image_ids, _ = get_all_image_ids_from_split()

    valid_image_ids = [
        image_id for image_id in image_ids
        if find_rgb_file(image_id) is not None
    ]

    if not valid_image_ids:
        raise RuntimeError("No valid images found.")

    random.seed(seed)
    sampled_ids = random.sample(valid_image_ids, min(num_samples, len(valid_image_ids)))

    rows = 2
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(22, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= len(sampled_ids):
            continue

        image_id = sampled_ids[i]
        rgb_path = find_rgb_file(image_id)
        label_path = LABELS_DIR / f"{image_id}.txt"

        image = np.array(Image.open(rgb_path).convert("RGB"))
        image_height, image_width = image.shape[:2]

        labels = load_yolo_labels(label_path)

        ax.imshow(image)

        for label in labels:
            x, y, w, h = yolo_to_pixel_bbox(label, image_width, image_height)

            class_idx = label["class_idx"]

            rect = Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )

            ax.add_patch(rect)

            ax.text(
                x,
                y - 3,
                f"cls {class_idx}",
                color="white",
                fontsize=8,
                bbox=dict(
                    facecolor="red",
                    alpha=0.7,
                    edgecolor="none",
                    pad=1,
                ),
            )

        ax.set_title(f"Image {image_id} | objects: {len(labels)}")

    fig.suptitle("Random Samples With YOLO Bounding Boxes", fontsize=14)
    fig.tight_layout()

    out_path = ANALYSIS_DIR / "bbox_samples_yolo.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved bbox visualization: {out_path}")


def create_statistics_plots():
    ensure_analysis_dir()

    image_ids, split_info = get_all_image_ids_from_split()

    objects_per_image = []
    class_counts = defaultdict(int)
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []

    split_counts = {
        split: len(ids)
        for split, ids in split_info.items()
    }

    for image_id in image_ids:
        label_path = LABELS_DIR / f"{image_id}.txt"
        labels = load_yolo_labels(label_path)

        objects_per_image.append(len(labels))

        for label in labels:
            class_idx = label["class_idx"]

            class_counts[class_idx] += 1
            bbox_widths.append(label["width"])
            bbox_heights.append(label["height"])
            bbox_areas.append(label["width"] * label["height"])

    # 1. Objects per image
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(len(objects_per_image)), objects_per_image)
    ax1.set_title("Number of Objects per Image")
    ax1.set_xlabel("Image index")
    ax1.set_ylabel("Number of objects")
    ax1.grid(axis="y", alpha=0.25)

    out1 = ANALYSIS_DIR / "objects_per_image.png"
    fig1.tight_layout()
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)

    # 2. Class distribution
    class_ids = sorted(class_counts.keys())
    class_values = [class_counts[c] for c in class_ids]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar([str(c) for c in class_ids], class_values)
    ax2.set_title("Object Count per Class")
    ax2.set_xlabel("Class index")
    ax2.set_ylabel("Number of objects")
    ax2.grid(axis="y", alpha=0.25)

    out2 = ANALYSIS_DIR / "class_distribution.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)

    # 3. Bbox area distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(bbox_areas, bins=30)
    ax3.set_title("Bounding Box Area Distribution")
    ax3.set_xlabel("Normalized bbox area")
    ax3.set_ylabel("Count")
    ax3.grid(axis="y", alpha=0.25)

    out3 = ANALYSIS_DIR / "bbox_area_distribution.png"
    fig3.tight_layout()
    fig3.savefig(out3, dpi=200)
    plt.close(fig3)

    print(f"[OK] Saved object-count chart: {out1}")
    print(f"[OK] Saved class-distribution chart: {out2}")
    print(f"[OK] Saved bbox-area chart: {out3}")

    print("\n[STATS SUMMARY]")
    print(f"Images total: {len(image_ids)}")
    print(f"Train images: {split_counts.get('train', 0)}")
    print(f"Val images: {split_counts.get('val', 0)}")
    print(f"Test images: {split_counts.get('test', 0)}")

    print(
        f"Objects per image -> "
        f"min: {min(objects_per_image)}, "
        f"max: {max(objects_per_image)}, "
        f"mean: {np.mean(objects_per_image):.2f}"
    )

    print(f"Total objects: {sum(objects_per_image)}")

    for class_id in class_ids:
        print(f"Class {class_id}: {class_counts[class_id]} objects")

    print(
        f"Bbox area -> "
        f"min: {min(bbox_areas):.6f}, "
        f"max: {max(bbox_areas):.6f}, "
        f"mean: {np.mean(bbox_areas):.6f}"
    )


if __name__ == "__main__":
    visualize_bbox_samples(num_samples=10)
    create_statistics_plots()