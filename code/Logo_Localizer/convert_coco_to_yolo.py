# Convert COCO format datasets to YOLO format,
# compatible with YOLO11_Detect training and testing.

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml


def convert_coco_to_yolo(json_path, image_root, save_dir, split='train'):
    """
    Convert COCO JSON to YOLO format and organize the folder structure.
    """
    # Creating a YOLO standard directory structure
    # datasets/logo_yolo/images/train
    # datasets/logo_yolo/labels/train
    img_save_dir = Path(save_dir) / 'images' / split
    label_save_dir = Path(save_dir) / 'labels' / split
    img_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_category_id = None
    for cat in data['categories']:
        if 'logo' in cat['name'].lower():
            target_category_id = cat['id']
            break

    if target_category_id is None:
        print("Warning: No category with 'logo' in its name was found. The first category was used as the target by default.")
        target_category_id = data['categories'][0]['id']

    img_id_to_anns = {}
    for ann in data['annotations']:
        if ann['category_id'] == target_category_id:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

    # 4. Iterate through the images to convert and copy them
    print(f"Start the conversion {split}...")
    processed_count = 0
    missing_count = 0

    for img_info in tqdm(data['images']):
        img_id = img_info['id']
        file_name = img_info['file_name']  # For example: benign_sample_30k/vaxjo.se/shot.png

        src_path = os.path.join(image_root, file_name)

        if not os.path.exists(src_path):
            possible_path = os.path.join(image_root, file_name.split('/')[-1])
            if os.path.exists(possible_path):
                src_path = possible_path
            else:
                missing_count += 1
                continue

        # Get the width and height of the image (for normalizing coordinates)
        img_w = img_info['width']
        img_h = img_info['height']

        # Get all labels of this figure
        anns = img_id_to_anns.get(img_id, [])

        if not anns:
            continue

        # Generate annotation content in YOLO format
        yolo_lines = []
        for ann in anns:
            bbox = ann['bbox']  # x, y, w, h
            x, y, w, h = bbox

            # Convert to YOLO center point coordinates (normalized 0-1)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Limit the value to between 0 and 1 (to prevent out-of-bounds annotations)
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        new_file_name = file_name.replace('/', '_').replace('\\', '_')
        dst_img_path = img_save_dir / new_file_name
        dst_label_path = label_save_dir / new_file_name.replace('.png', '.txt').replace('.jpg', '.txt')

        shutil.copy2(src_path, dst_img_path)

        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        processed_count += 1

    print(f"{split} set processing complete: Success {processed_count}, Missing {missing_count}")


# ================= Configuration area =================
TRAIN_JSON = "./datasets/coco_train.json"
TEST_JSON = "./datasets/coco_test.json"
IMAGE_ROOT = "../../data/benign_sample_30k/"

OUTPUT_DIR = "./datasets/"

if __name__ == "__main__":
    convert_coco_to_yolo(TRAIN_JSON, IMAGE_ROOT, OUTPUT_DIR, split='train')
    convert_coco_to_yolo(TEST_JSON, IMAGE_ROOT, OUTPUT_DIR, split='val')
    print("All data conversion complete!")