from pathlib import Path
import json
import shutil


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "merged_dataset"

RGB_FOLDER_NAME = "rgb"
MASK_FOLDER_NAME = "mask"
MASK_VISIB_FOLDER_NAME = "mask_visib"

SCENE_GT_JSON_NAME = "scene_gt.json"
SCENE_GT_INFO_JSON_NAME = "scene_gt_info.json"
BBOX_CLASS_JSON_NAME = "scene_bbox_obj_class.json"
VISIB_FRACT_JSON_NAME = "scene_visib_fract.json"


def get_source_dirs(data_dir: Path):
    if not data_dir.exists():
        print(f"Data folder not found: {data_dir}")
        return []

    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / RGB_FOLDER_NAME).exists()
    )



def make_image_dirs(output_dir: Path):
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask_visi").mkdir(parents=True, exist_ok=True)



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



def copy_matching_masks(
    source_mask_dir: Path,
    output_mask_dir: Path,
    old_image_id: str,
    new_image_id: str,
):

    if not source_mask_dir.exists():
        print(f"Missing mask folder: {source_mask_dir}")
        return

    pattern = f"{old_image_id}_*"

    for mask_path in sorted(source_mask_dir.glob(pattern)):
        object_id = mask_path.stem.split("_")[1]
        new_mask_name = f"{new_image_id}_{object_id}{mask_path.suffix}"

        target_path = output_mask_dir / new_mask_name
        shutil.copy2(mask_path, target_path)



def build_bbox_and_visibility_json(merged_scene_gt: dict, merged_scene_gt_info: dict):
    bbox_class_data = {}
    visib_fract_data = {}

    all_scene_keys = sorted(
        set(merged_scene_gt.keys()) | set(merged_scene_gt_info.keys()),
        key=int,
    )

    for scene_key in all_scene_keys:
        gt_objects = merged_scene_gt.get(scene_key, [])
        gt_info_objects = merged_scene_gt_info.get(scene_key, [])

        if len(gt_objects) != len(gt_info_objects):
            print(
                f"Warning: scene {scene_key} has different object counts "
                f"(scene_gt={len(gt_objects)}, scene_gt_info={len(gt_info_objects)})."
            )

        object_count = min(len(gt_objects), len(gt_info_objects))

        bbox_class_rows = []
        visib_fract_rows = []

        for obj_index in range(object_count):
            gt_obj = gt_objects[obj_index]
            gt_info_obj = gt_info_objects[obj_index]

            bbox_class_rows.append(
                {
                    "obj_id": gt_obj.get("obj_id"),
                    "bbox_obj": gt_info_obj.get("bbox_obj"),
                }
            )

            visib_fract_rows.append(
                {
                    "obj_id": gt_obj.get("obj_id"),
                    "visib_fract": gt_info_obj.get("visib_fract"),
                }
            )

        bbox_class_data[scene_key] = bbox_class_rows
        visib_fract_data[scene_key] = visib_fract_rows

    return bbox_class_data, visib_fract_data


def merge_images(source_dirs, output_dir: Path):
    make_image_dirs(output_dir)

    global_image_index = 0

    for source_dir in source_dirs:
        print(f"\n[IMAGES] Processing folder: {source_dir}")

        rgb_dir = source_dir / RGB_FOLDER_NAME
        mask_dir = source_dir / MASK_FOLDER_NAME
        mask_visib_dir = source_dir / MASK_VISIB_FOLDER_NAME

        if not rgb_dir.exists():
            print(f"[IMAGES] Skipped, missing rgb folder: {rgb_dir}")
            continue

        rgb_files = sorted(rgb_dir.glob("*.*"))

        for rgb_path in rgb_files:
            old_image_id = rgb_path.stem
            new_image_id = f"{global_image_index:06d}"

            new_rgb_name = f"{new_image_id}{rgb_path.suffix}"
            target_rgb_path = output_dir / "rgb" / new_rgb_name
            shutil.copy2(rgb_path, target_rgb_path)

            copy_matching_masks(
                source_mask_dir=mask_dir,
                output_mask_dir=output_dir / "mask",
                old_image_id=old_image_id,
                new_image_id=new_image_id,
            )

            copy_matching_masks(
                source_mask_dir=mask_visib_dir,
                output_mask_dir=output_dir / "mask_visi",
                old_image_id=old_image_id,
                new_image_id=new_image_id,
            )

            global_image_index += 1

    print(f"\n[IMAGES] Done. Copied RGB images: {global_image_index}")
    print(f"[IMAGES] Output saved to: {output_dir}")



def merge_json(source_dirs, output_dir: Path):
    merged_scene_gt = {}
    merged_scene_gt_info = {}
    global_image_index = 0

    for source_dir in source_dirs:
        print(f"\n[JSON] Processing folder: {source_dir}")

        rgb_dir = source_dir / RGB_FOLDER_NAME
        scene_gt = load_json_dict(source_dir / SCENE_GT_JSON_NAME)
        scene_gt_info = load_json_dict(source_dir / SCENE_GT_INFO_JSON_NAME)

        if not rgb_dir.exists():
            print(f"[JSON] Skipped, missing rgb folder: {rgb_dir}")
            continue

        rgb_files = sorted(rgb_dir.glob("*.*"))

        for rgb_path in rgb_files:
            old_image_id = rgb_path.stem
            new_json_key = str(global_image_index)

            gt_entry = get_json_entry_by_image_id(scene_gt, old_image_id)
            gt_info_entry = get_json_entry_by_image_id(scene_gt_info, old_image_id)

            if gt_entry is not None:
                merged_scene_gt[new_json_key] = gt_entry
            else:
                print(f"Warning: no {SCENE_GT_JSON_NAME} entry for image ID {old_image_id} in {source_dir}")

            if gt_info_entry is not None:
                merged_scene_gt_info[new_json_key] = gt_info_entry
            else:
                print(
                    f"Warning: no {SCENE_GT_INFO_JSON_NAME} entry for image ID {old_image_id} in {source_dir}"
                )

            global_image_index += 1

    with (output_dir / SCENE_GT_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(merged_scene_gt, f)

    with (output_dir / SCENE_GT_INFO_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(merged_scene_gt_info, f)

    print(f"\n[JSON] Done. Merged {SCENE_GT_JSON_NAME} entries: {len(merged_scene_gt)}")
    print(f"[JSON] Merged {SCENE_GT_INFO_JSON_NAME} entries: {len(merged_scene_gt_info)}")
    print(f"[JSON] Output saved to: {output_dir}")



def create_derived_json(output_dir: Path):
    merged_scene_gt = load_json_dict(output_dir / SCENE_GT_JSON_NAME)
    merged_scene_gt_info = load_json_dict(output_dir / SCENE_GT_INFO_JSON_NAME)

    bbox_class_data, visib_fract_data = build_bbox_and_visibility_json(
        merged_scene_gt,
        merged_scene_gt_info,
    )

    with (output_dir / BBOX_CLASS_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(bbox_class_data, f)

    with (output_dir / VISIB_FRACT_JSON_NAME).open("w", encoding="utf-8") as f:
        json.dump(visib_fract_data, f)

    print(f"\n[DERIVED] Done. Created {BBOX_CLASS_JSON_NAME} entries: {len(bbox_class_data)}")
    print(f"[DERIVED] Created {VISIB_FRACT_JSON_NAME} entries: {len(visib_fract_data)}")
    print(f"[DERIVED] Output saved to: {output_dir}")


if __name__ == "__main__":
    source_dirs = get_source_dirs(DATA_DIR)
    merge_images(source_dirs, OUTPUT_DIR)
    merge_json(source_dirs, OUTPUT_DIR)
    create_derived_json(OUTPUT_DIR)