from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "merged_dataset"

RGB_DIR = DATASET_DIR / "rgb"
LABELS_DIR = DATASET_DIR / "labels"
SPLIT_INFO_JSON = DATASET_DIR / "split_info.json"
CLASS_MAPPING_JSON = DATASET_DIR / "class_mapping.json"

IMAGE_SIZE = 640
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

MODEL_TYPE = "yolo"
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_path_from_id(image_id: str, rgb_dir: Path = RGB_DIR):
    matches = sorted(rgb_dir.glob(f"{image_id}.*"))
    if not matches:
        raise FileNotFoundError(f"No image found for id: {image_id}")
    return matches[0]


def letterbox_image(image: Image.Image, size: int = IMAGE_SIZE, fill=(114, 114, 114)):
    original_width, original_height = image.size
    scale = min(size / original_width, size / original_height)

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), fill)

    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    return canvas


def image_to_tensor(image: Image.Image, model_type: str = MODEL_TYPE):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    if model_type == "yolo":
        return tensor

    if model_type == "faster_rcnn":
        mean_tensor = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(3, 1, 1)
        std_tensor = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean_tensor) / std_tensor

    raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


def load_yolo_labels(label_path: Path):
    labels = []

    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_idx, x_center, y_center, width, height = parts

            labels.append([
                float(class_idx),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            ])

    if not labels:
        return torch.zeros((0, 5), dtype=torch.float32)

    return torch.tensor(labels, dtype=torch.float32)


class YoloTxtDetectionDataset(Dataset):
   
    def __init__(
        self,
        split: str,
        dataset_dir: Path = DATASET_DIR,
        image_size: int = IMAGE_SIZE,
        normalize: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.rgb_dir = self.dataset_dir / "rgb"
        self.labels_dir = self.dataset_dir / "labels"
        self.image_size = image_size
        self.normalize = normalize

        split_info = load_json(self.dataset_dir / "split_info.json")

        if split not in split_info:
            raise ValueError(f"Unknown split: {split}. Use: train, val or test.")

        self.image_ids = split_info[split]

        class_mapping_path = self.dataset_dir / "class_mapping.json"
        self.class_mapping = load_json(class_mapping_path)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = image_path_from_id(image_id, self.rgb_dir)
        label_path = self.labels_dir / f"{image_id}.txt"

        image = Image.open(image_path).convert("RGB")
        image = letterbox_image(image, self.image_size)

        if self.normalize:
            image_tensor = image_to_tensor(image)
        else:
            image_tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)

        labels_tensor = load_yolo_labels(label_path)

        return image_tensor, labels_tensor


def yolo_collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images, dim=0)

    return images, list(labels)


def create_dataloaders(
    batch_size: int = 8,
    image_size: int = IMAGE_SIZE,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_dataset = YoloTxtDetectionDataset(
        split="train",
        image_size=image_size,
        normalize=normalize,
    )

    val_dataset = YoloTxtDetectionDataset(
        split="val",
        image_size=image_size,
        normalize=normalize,
    )

    test_dataset = YoloTxtDetectionDataset(
        split="test",
        image_size=image_size,
        normalize=normalize,
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

    class_mapping = train_dataset.class_mapping

    return train_loader, val_loader, test_loader, class_mapping


def inspect_dataset():
    train_dataset = YoloTxtDetectionDataset(split="train")

    image, labels = train_dataset[0]

    print(f"Train images: {len(train_dataset)}")
    print(f"Image tensor shape: {tuple(image.shape)}")
    print(f"First label tensor shape: {tuple(labels.shape)}")
    print("Label columns: class_idx, x_center, y_center, width, height")
    print(f"Class mapping: {train_dataset.class_mapping}")


if __name__ == "__main__":
    inspect_dataset()