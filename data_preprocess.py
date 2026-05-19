from pathlib import Path
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "merged_dataset"
RGB_DIR = DATASET_DIR / "rgb"

CLEAN_SCENE_GT_JSON = DATASET_DIR / "scene_gt_clean.json"
CLEAN_SCENE_GT_INFO_JSON = DATASET_DIR / "scene_gt_info_clean.json"
SCENE_GT_JSON = DATASET_DIR / "scene_gt.json"
SCENE_GT_INFO_JSON = DATASET_DIR / "scene_gt_info.json"

IMAGE_SIZE = 640
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_annotation_files(dataset_dir: Path = DATASET_DIR, use_cleaned: bool = True):
    """
    Uses cleaned annotations by default. Falls back to original files if cleaned
    files do not exist.
    """
    dataset_dir = Path(dataset_dir)
    clean_scene_gt = dataset_dir / "scene_gt_clean.json"
    clean_scene_gt_info = dataset_dir / "scene_gt_info_clean.json"
    scene_gt = dataset_dir / "scene_gt.json"
    scene_gt_info = dataset_dir / "scene_gt_info.json"

    if use_cleaned and clean_scene_gt.exists() and clean_scene_gt_info.exists():
        return clean_scene_gt, clean_scene_gt_info

    return scene_gt, scene_gt_info


def image_path_from_scene(scene_key: str, rgb_dir: Path = RGB_DIR):
    image_id = f"{int(scene_key):06d}"
    matches = sorted(Path(rgb_dir).glob(f"{image_id}.*"))
    if not matches:
        return None
    return matches[0]


def is_valid_bbox(bbox):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    if any(not isinstance(v, (int, float)) for v in bbox):
        return False
    if all(v == -1 for v in bbox):
        return False

    _, _, width, height = bbox
    return width > 0 and height > 0


def letterbox_image(image: Image.Image, size: int = IMAGE_SIZE, fill=(114, 114, 114)):
    """
    YOLO-style resize: keeps aspect ratio and pads to a square image.
    Returns resized image, scale, pad_x, pad_y.
    """
    original_width, original_height = image.size
    scale = min(size / original_width, size / original_height)

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), fill)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y


def bbox_to_yolo(bbox, scale: float, pad_x: int, pad_y: int, image_size: int):
    """
    Converts bbox_obj [x, y, width, height] to YOLO format:
    [x_center, y_center, width, height], normalized to 0..1.
    """
    x, y, width, height = bbox

    x = x * scale + pad_x
    y = y * scale + pad_y
    width = width * scale
    height = height * scale

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_size), x + width)
    y2 = min(float(image_size), y + height)

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / image_size
    y_center = (y1 + y2) / 2.0 / image_size
    norm_width = clipped_width / image_size
    norm_height = clipped_height / image_size

    return [x_center, y_center, norm_width, norm_height]


def image_to_tensor(image: Image.Image, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


class YoloDetectionDataset(Dataset):
    """
    PyTorch Dataset for YOLO-style object detection.

    Returns:
    - image: float tensor [3, image_size, image_size]
    - labels: float tensor [num_objects, 5]
      labels columns: [class_idx, x_center, y_center, width, height]
    """

    def __init__(
        self,
        dataset_dir: Path = DATASET_DIR,
        image_size: int = IMAGE_SIZE,
        use_cleaned: bool = True,
        normalize: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.rgb_dir = self.dataset_dir / "rgb"
        self.image_size = image_size
        self.normalize = normalize

        scene_gt_path, scene_gt_info_path = choose_annotation_files(
            dataset_dir=self.dataset_dir,
            use_cleaned=use_cleaned,
        )
        self.scene_gt = load_json(scene_gt_path)
        self.scene_gt_info = load_json(scene_gt_info_path)

        self.scene_keys = self._collect_valid_scene_keys()
        self.class_ids = self._collect_class_ids()
        self.class_to_index = {obj_id: idx for idx, obj_id in enumerate(self.class_ids)}

    def _collect_valid_scene_keys(self):
        keys = sorted(set(self.scene_gt.keys()) & set(self.scene_gt_info.keys()), key=int)
        return [key for key in keys if image_path_from_scene(key, self.rgb_dir) is not None]

    def _collect_class_ids(self):
        class_ids = set()
        for objects in self.scene_gt.values():
            for obj in objects:
                obj_id = obj.get("obj_id")
                if obj_id is not None:
                    class_ids.add(int(obj_id))
        return sorted(class_ids)

    def __len__(self):
        return len(self.scene_keys)

    def __getitem__(self, index):
        scene_key = self.scene_keys[index]
        image_path = image_path_from_scene(scene_key, self.rgb_dir)
        image = Image.open(image_path).convert("RGB")
        image, scale, pad_x, pad_y = letterbox_image(image, self.image_size)

        labels = []
        gt_objects = self.scene_gt.get(scene_key, [])
        info_objects = self.scene_gt_info.get(scene_key, [])

        for gt_obj, info_obj in zip(gt_objects, info_objects):
            bbox = info_obj.get("bbox_obj")
            obj_id = gt_obj.get("obj_id")
            if obj_id is None or not is_valid_bbox(bbox):
                continue

            yolo_bbox = bbox_to_yolo(bbox, scale, pad_x, pad_y, self.image_size)
            if yolo_bbox is None:
                continue

            class_idx = self.class_to_index[int(obj_id)]
            labels.append([class_idx, *yolo_bbox])

        if self.normalize:
            image_tensor = image_to_tensor(image)
        else:
            image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1)

        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return image_tensor, labels_tensor


def yolo_collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), list(labels)


def create_dataloaders(
    batch_size: int = 8,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    image_size: int = IMAGE_SIZE,
    use_cleaned: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    dataset = YoloDetectionDataset(image_size=image_size, use_cleaned=use_cleaned)

    train_size = int(len(dataset) * train_fraction)
    val_size = int(len(dataset) * val_fraction)
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=yolo_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=yolo_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=yolo_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, dataset.class_to_index


def inspect_dataset():
    dataset = YoloDetectionDataset()
    image, labels = dataset[0]

    print(f"Images: {len(dataset)}")
    print(f"Class mapping obj_id -> class_idx: {dataset.class_to_index}")
    print(f"Image tensor shape: {tuple(image.shape)}")
    print(f"First sample labels shape: {tuple(labels.shape)}")
    print("Label columns: class_idx, x_center, y_center, width, height")


if __name__ == "__main__":
    random.seed(42)
    inspect_dataset()
