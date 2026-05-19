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
MASK_DIR = DATASET_DIR / "mask"

SCENE_GT_JSON = DATASET_DIR / "scene_gt.json"
SCENE_GT_INFO_JSON = DATASET_DIR / "scene_gt_info.json"

CLEAN_SCENE_GT_JSON = DATASET_DIR / "scene_gt_clean.json"
CLEAN_SCENE_GT_INFO_JSON = DATASET_DIR / "scene_gt_info_clean.json"
INVALID_BBOX_REPORT_JSON = ANALYSIS_DIR / "invalid_bbox_report.json"



def load_json_dict(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")

    return data



def ensure_analysis_dir():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)



def scene_key_to_image_id(scene_key: str) -> str:
    return f"{int(scene_key):06d}"



def find_rgb_file(scene_key: str):
    image_id = scene_key_to_image_id(scene_key)
    candidates = sorted(RGB_DIR.glob(f"{image_id}.*"))
    if not candidates:
        return None
    return candidates[0]



def get_common_scene_keys(scene_gt: dict, scene_gt_info: dict):
    keys = set(scene_gt.keys()) & set(scene_gt_info.keys())
    keys = [k for k in keys if find_rgb_file(k) is not None]
    return sorted(keys, key=int)



def random_color(seed_value: int):
    rng = random.Random(seed_value)
    return (rng.random(), rng.random(), rng.random())


def is_invalid_bbox(bbox):
    """
    Checks if bbox is unusable.
    Invalid means missing/malformed, all values are -1, or width/height are not positive.
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return True

    if any(not isinstance(value, (int, float)) for value in bbox):
        return True

    x, y, width, height = bbox
    if all(value == -1 for value in bbox):
        return True

    if width <= 0 or height <= 0:
        return True

    return False


def clean_invalid_bbox_objects():
    """
    Removes objects with invalid bbox_obj from scene_gt and scene_gt_info copies.
    Writes cleaned JSON files and a report listing scene/object details.
    """
    ensure_analysis_dir()

    scene_gt = load_json_dict(SCENE_GT_JSON)
    scene_gt_info = load_json_dict(SCENE_GT_INFO_JSON)
    scene_keys = sorted(set(scene_gt.keys()) & set(scene_gt_info.keys()), key=int)

    clean_scene_gt = {}
    clean_scene_gt_info = {}
    invalid_report = []

    for scene_key in scene_keys:
        gt_objects = scene_gt.get(scene_key, [])
        info_objects = scene_gt_info.get(scene_key, [])
        object_count = min(len(gt_objects), len(info_objects))

        clean_gt_objects = []
        clean_info_objects = []

        for obj_index in range(object_count):
            gt_obj = gt_objects[obj_index]
            info_obj = info_objects[obj_index]
            bbox = info_obj.get("bbox_obj")

            if is_invalid_bbox(bbox):
                invalid_report.append(
                    {
                        "scene": int(scene_key),
                        "object_index": obj_index,
                        "obj_id": gt_obj.get("obj_id"),
                        "bbox_obj": bbox,
                    }
                )
                continue

            clean_gt_objects.append(gt_obj)
            clean_info_objects.append(info_obj)

        clean_scene_gt[scene_key] = clean_gt_objects
        clean_scene_gt_info[scene_key] = clean_info_objects

    with CLEAN_SCENE_GT_JSON.open("w", encoding="utf-8") as f:
        json.dump(clean_scene_gt, f)

    with CLEAN_SCENE_GT_INFO_JSON.open("w", encoding="utf-8") as f:
        json.dump(clean_scene_gt_info, f)

    with INVALID_BBOX_REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(invalid_report, f, indent=2)

    print(f"[OK] Saved cleaned scene_gt: {CLEAN_SCENE_GT_JSON}")
    print(f"[OK] Saved cleaned scene_gt_info: {CLEAN_SCENE_GT_INFO_JSON}")
    print(f"[OK] Saved invalid bbox report: {INVALID_BBOX_REPORT_JSON}")
    print(f"[CLEAN] Invalid objects removed: {len(invalid_report)}")

    if invalid_report:
        print("\n[INVALID BBOX OBJECTS]")
        for item in invalid_report:
            print(
                f"Scene {item['scene']}, object_index {item['object_index']}, "
                f"obj_id {item['obj_id']}, bbox {item['bbox_obj']}"
            )


def create_cleaned_statistics_plots():
    """Creates statistics plots from JSON files with invalid bbox objects removed."""
    create_statistics_plots(
        scene_gt_path=CLEAN_SCENE_GT_JSON,
        scene_gt_info_path=CLEAN_SCENE_GT_INFO_JSON,
        output_suffix="_cleaned",
    )

    with INVALID_BBOX_REPORT_JSON.open("r", encoding="utf-8") as f:
        invalid_report = json.load(f)

    print(f"[CLEAN] Incorrect objects in original data: {len(invalid_report)}")


def visualize_bbox_samples(num_samples: int = 10, seed: int = 42):
    """
    Saves figure with random samples and bounding boxes.
    """
    ensure_analysis_dir()

    scene_gt = load_json_dict(SCENE_GT_JSON)
    scene_gt_info = load_json_dict(SCENE_GT_INFO_JSON)
    scene_keys = get_common_scene_keys(scene_gt, scene_gt_info)

    if not scene_keys:
        raise RuntimeError("No valid scenes found for bbox visualization.")

    sample_size = min(num_samples, len(scene_keys))
    random.seed(seed)
    sampled_keys = random.sample(scene_keys, sample_size)

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(22, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= sample_size:
            continue

        scene_key = sampled_keys[i]
        rgb_path = find_rgb_file(scene_key)
        image = np.array(Image.open(rgb_path).convert("RGB"))
        gt_objects = scene_gt.get(scene_key, [])
        gt_info_objects = scene_gt_info.get(scene_key, [])

        ax.imshow(image)
        object_count = min(len(gt_objects), len(gt_info_objects))

        for obj_idx in range(object_count):
            gt_obj = gt_objects[obj_idx]
            info_obj = gt_info_objects[obj_idx]

            bbox = info_obj.get("bbox_obj")
            obj_id = gt_obj.get("obj_id")

            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            color = random_color(int(obj_id) if obj_id is not None else obj_idx)
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x,
                y - 3,
                f"cls {obj_id}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
            )

        ax.set_title(f"Scene {scene_key}")

    fig.suptitle("Random Samples With Bounding Boxes", fontsize=14)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "bbox_samples.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved bbox visualization: {out_path}")



def overlay_masks_on_image(rgb_image: np.ndarray, mask_files, alpha: float = 0.7):
    """Returns RGB image with clearly visible black mask overlays."""
    result = rgb_image.astype(np.float32) / 255.0
    mask_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for mask_path in mask_files:
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        if not mask.any():
            continue

        result[mask] = (1.0 - alpha) * result[mask] + alpha * mask_color

    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result



def visualize_mask_samples(num_samples: int = 10, seed: int = 42):
    """
    Saves figure with random samples and mask overlays.
    """
    ensure_analysis_dir()

    scene_gt = load_json_dict(SCENE_GT_JSON)
    scene_gt_info = load_json_dict(SCENE_GT_INFO_JSON)
    scene_keys = get_common_scene_keys(scene_gt, scene_gt_info)

    if not scene_keys:
        raise RuntimeError("No valid scenes found for mask visualization.")

    sample_size = min(num_samples, len(scene_keys))
    random.seed(seed + 1)
    sampled_keys = random.sample(scene_keys, sample_size)

    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(22, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= sample_size:
            continue

        scene_key = sampled_keys[i]
        image_id = scene_key_to_image_id(scene_key)
        rgb_path = find_rgb_file(scene_key)
        rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
        mask_files = sorted(MASK_DIR.glob(f"{image_id}_*.*"))

        overlay = overlay_masks_on_image(rgb_image, mask_files)
        ax.imshow(overlay)
        ax.set_title(f"Scene {scene_key} | masks: {len(mask_files)}")

    fig.suptitle("Random Samples With Mask Overlays", fontsize=14)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / "mask_samples.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved mask visualization: {out_path}")


def create_statistics_plots(
    scene_gt_path: Path = SCENE_GT_JSON,
    scene_gt_info_path: Path = SCENE_GT_INFO_JSON,
    output_suffix: str = "",
):
    """
    Creates:
    1) Scene-by-scene plot of number of objects per image
    2) Mean visibility per object class (obj_id)
    """
    ensure_analysis_dir()

    scene_gt = load_json_dict(scene_gt_path)
    scene_gt_info = load_json_dict(scene_gt_info_path)
    scene_keys = get_common_scene_keys(scene_gt, scene_gt_info)

    if not scene_keys:
        raise RuntimeError("No valid scenes found for statistics.")

    # Count objects per scene by checking how many objects have obj_id.
    scene_numbers = [int(k) for k in scene_keys]
    objects_per_scene = [
        sum(1 for obj in scene_gt[k] if obj.get("obj_id") is not None)
        for k in scene_keys
    ]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    min_v = min(objects_per_scene)
    max_v = max(objects_per_scene)
    ax1.bar(scene_numbers, objects_per_scene, width=1.0, alpha=0.85)
    ax1.set_title("Number of Objects in Each Scene")
    ax1.set_xlabel("Scene number")
    ax1.set_ylabel("Number of objects")
    ax1.grid(alpha=0.25)
    hist_path = ANALYSIS_DIR / f"objects_per_scene{output_suffix}.png"
    fig1.tight_layout()
    fig1.savefig(hist_path, dpi=200)
    plt.close(fig1)

    # Mean visibility by class
    vis_sum = defaultdict(float)
    vis_count = defaultdict(int)

    for scene_key in scene_keys:
        gt_objects = scene_gt.get(scene_key, [])
        info_objects = scene_gt_info.get(scene_key, [])
        object_count = min(len(gt_objects), len(info_objects))

        for i in range(object_count):
            obj_id = gt_objects[i].get("obj_id")
            vis = info_objects[i].get("visib_fract")

            if obj_id is None:
                continue
            if not isinstance(vis, (int, float)):
                continue

            vis_sum[int(obj_id)] += float(vis)
            vis_count[int(obj_id)] += 1

    class_ids = sorted(vis_sum.keys())
    mean_vis = [vis_sum[c] / vis_count[c] for c in class_ids]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar([str(c) for c in class_ids], mean_vis)
    ax2.set_title("Mean Object Visibility per Class")
    ax2.set_xlabel("Class (obj_id)")
    ax2.set_ylabel("Mean visib_fract")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", alpha=0.25)
    vis_path = ANALYSIS_DIR / f"mean_visibility_per_class{output_suffix}.png"
    fig2.tight_layout()
    fig2.savefig(vis_path, dpi=200)
    plt.close(fig2)

    print(f"[OK] Saved object-count chart: {hist_path}")
    print(f"[OK] Saved visibility chart: {vis_path}")

    dataset_name = "cleaned dataset" if output_suffix else "original dataset"
    print(f"\n[STATS SUMMARY - {dataset_name}]")
    print(f"Scenes analyzed: {len(scene_keys)}")
    print(f"Objects per scene -> min: {min_v}, max: {max_v}, mean: {np.mean(objects_per_scene):.2f}")
    for class_id, mean_value in zip(class_ids, mean_vis):
        print(f"Class {class_id}: mean visibility = {mean_value:.4f} (n={vis_count[class_id]})")


if __name__ == "__main__":
    visualize_bbox_samples(num_samples=10)
    visualize_mask_samples(num_samples=10)
    create_statistics_plots()
    clean_invalid_bbox_objects()
    create_cleaned_statistics_plots()
