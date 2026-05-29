from pathlib import Path
import json
import random
import shutil
import struct


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "merged_dataset"

RGB_FOLDER_NAME = "rgb"
SCENE_GT_JSON_NAME = "scene_gt.json"
SCENE_GT_INFO_JSON_NAME = "scene_gt_info.json"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SHUFFLE = True
SEED = 42


def get_source_dirs(data_dir: Path):
    if not data_dir.exists():
        print(f"Data folder not found: {data_dir}")
        return []

    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / RGB_FOLDER_NAME).exists()
    )


def load_json_dict(json_path: Path):
    if not json_path.exists():
        print(f"Missing JSON: {json_path}")
        return {}

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print(f"Warning: expected dict in {json_path}, got {type(data).__name__}. Ignoring.")
        return {}

    return data


def get_json_entry_by_image_id(json_dict, old_image_id: str):
    if old_image_id in json_dict:
        return json_dict[old_image_id]

    old_image_id_int = str(int(old_image_id))
    if old_image_id_int in json_dict:
        return json_dict[old_image_id_int]

    return None


def make_yolo_dirs(output_dir: Path):
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def get_image_size(image_path: Path):
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return get_png_size(image_path)
    if suffix in {".jpg", ".jpeg"}:
        return get_jpeg_size(image_path)
    raise ValueError(f"Unsupported image format: {image_path}")


def get_png_size(image_path: Path):
    with image_path.open("rb") as f:
        signature = f.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid PNG file: {image_path}")
        ihdr_length = f.read(4)
        ihdr_type = f.read(4)
        if len(ihdr_length) < 4 or ihdr_type != b"IHDR":
            raise ValueError(f"Invalid PNG IHDR: {image_path}")
        ihdr_data = f.read(13)
        if len(ihdr_data) < 13:
            raise ValueError(f"Truncated PNG IHDR: {image_path}")
        width, height = struct.unpack(">II", ihdr_data[:8])
        return int(width), int(height)


def get_jpeg_size(image_path: Path):
    with image_path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Invalid JPEG file: {image_path}")

        while True:
            marker_start = f.read(1)
            if not marker_start:
                break

            if marker_start != b"\xff":
                continue

            marker_code = f.read(1)
            if not marker_code:
                break

            while marker_code == b"\xff":
                marker_code = f.read(1)
                if not marker_code:
                    break

            if marker_code in {b"\xd8", b"\xd9"}:
                continue

            segment_length_bytes = f.read(2)
            if len(segment_length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"Invalid JPEG segment length in {image_path}")

            if marker_code in {
                b"\xc0", b"\xc1", b"\xc2", b"\xc3",
                b"\xc5", b"\xc6", b"\xc7",
                b"\xc9", b"\xca", b"\xcb",
                b"\xcd", b"\xce", b"\xcf",
            }:
                sof_data = f.read(segment_length - 2)
                if len(sof_data) < 5:
                    break
                height, width = struct.unpack(">HH", sof_data[1:5])
                return int(width), int(height)

            f.seek(segment_length - 2, 1)

    raise ValueError(f"Could not read JPEG size: {image_path}")


def is_valid_bbox(bbox):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    if any(not isinstance(v, (int, float)) for v in bbox):
        return False

    _, _, width, height = bbox
    return width > 0 and height > 0


def bbox_to_yolo(bbox, image_width: int, image_height: int):
    x, y, width, height = bbox

    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(image_width), float(x) + float(width))
    y2 = min(float(image_height), float(y) + float(height))

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    norm_width = clipped_width / image_width
    norm_height = clipped_height / image_height
    return x_center, y_center, norm_width, norm_height


def collect_records(source_dirs):
    records = []
    all_obj_ids = set()

    for source_dir in source_dirs:
        print(f"\n[COLLECT] Processing: {source_dir}")
        rgb_dir = source_dir / RGB_FOLDER_NAME

        if not rgb_dir.exists():
            print(f"[COLLECT] Skipped, missing rgb folder: {rgb_dir}")
            continue

        scene_gt = load_json_dict(source_dir / SCENE_GT_JSON_NAME)
        scene_gt_info = load_json_dict(source_dir / SCENE_GT_INFO_JSON_NAME)
        rgb_files = sorted(rgb_dir.glob("*.*"))

        for rgb_path in rgb_files:
            old_image_id = rgb_path.stem
            gt_entry = get_json_entry_by_image_id(scene_gt, old_image_id) or []
            gt_info_entry = get_json_entry_by_image_id(scene_gt_info, old_image_id) or []

            if len(gt_entry) != len(gt_info_entry):
                print(
                    f"[COLLECT] Warning: {source_dir.name}/{old_image_id} has different object counts "
                    f"(scene_gt={len(gt_entry)}, scene_gt_info={len(gt_info_entry)})."
                )

            for gt_obj in gt_entry:
                obj_id = gt_obj.get("obj_id")
                if obj_id is not None:
                    all_obj_ids.add(int(obj_id))

            image_width, image_height = get_image_size(rgb_path)
            records.append(
                {
                    "rgb_path": rgb_path,
                    "gt": gt_entry,
                    "gt_info": gt_info_entry,
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )

    return records, sorted(all_obj_ids)


def assign_split(index: int, train_end: int, val_end: int):
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def write_data_yaml(output_dir: Path, class_ids):
    class_names = [f"obj_{obj_id}" for obj_id in class_ids]

    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]

    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_to_yolo(source_dirs, output_dir: Path):
    if abs((TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION) - 1.0) > 1e-9:
        raise ValueError("TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION must be 1.0")

    make_yolo_dirs(output_dir)

    records, class_ids = collect_records(source_dirs)
    if not records:
        print("No input images found. Nothing to merge.")
        return

    class_to_idx = {obj_id: idx for idx, obj_id in enumerate(class_ids)}

    if SHUFFLE:
        random.Random(SEED).shuffle(records)

    total = len(records)
    train_end = int(total * TRAIN_FRACTION)
    val_end = train_end + int(total * VAL_FRACTION)

    copied_images = 0
    written_labels = 0
    skipped_boxes = 0

    for index, record in enumerate(records):
        split = assign_split(index, train_end, val_end)
        new_image_id = f"{index:06d}"
        source_image_path = record["rgb_path"]
        image_suffix = source_image_path.suffix.lower()

        target_image_path = output_dir / "images" / split / f"{new_image_id}{image_suffix}"
        shutil.copy2(source_image_path, target_image_path)
        copied_images += 1

        label_lines = []
        gt_objects = record["gt"]
        info_objects = record["gt_info"]
        image_width = record["image_width"]
        image_height = record["image_height"]

        for gt_obj, info_obj in zip(gt_objects, info_objects):
            obj_id = gt_obj.get("obj_id")
            bbox = info_obj.get("bbox_obj")

            if obj_id is None or not is_valid_bbox(bbox):
                skipped_boxes += 1
                continue

            yolo_box = bbox_to_yolo(bbox, image_width, image_height)
            if yolo_box is None:
                skipped_boxes += 1
                continue

            class_idx = class_to_idx[int(obj_id)]
            x_center, y_center, width, height = yolo_box
            label_lines.append(
                f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        label_path = output_dir / "labels" / split / f"{new_image_id}.txt"
        label_text = "\n".join(label_lines)
        if label_text:
            label_text += "\n"
        label_path.write_text(label_text, encoding="utf-8")
        written_labels += 1

    write_data_yaml(output_dir, class_ids)

    print(f"\n[DONE] Output directory: {output_dir}")
    print(f"[DONE] Images copied: {copied_images}")
    print(f"[DONE] Label files written: {written_labels}")
    print(f"[DONE] Classes (obj_id -> class_idx): {class_to_idx}")
    print(f"[DONE] Skipped invalid/clipped-out boxes: {skipped_boxes}")


if __name__ == "__main__":
    source_dirs = get_source_dirs(DATA_DIR)
    merge_to_yolo(source_dirs, OUTPUT_DIR)
