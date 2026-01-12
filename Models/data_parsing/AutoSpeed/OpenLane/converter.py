import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm  # pip install tqdm
import json
import random

image_width = 1920
image_height = 1280


def move_images(input_dir, output_dir):
    input_dir = Path(input_dir)

    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Moving files", unit="file"):
        if file.is_file():
            target = Path(output_dir) / file.name
            target.parent.mkdir(parents=True, exist_ok=True)

            # If file with same name exists, rename
            if target.exists():
                print(f"File with same name already exists: {target}")

            # Move file
            shutil.move(str(file), str(target))

    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)


def convert_labels(input_dir, output_dir):
    input_dir = Path(input_dir)
    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Convert labels", unit="file"):
        parent_dir = file.parent
        base_name = file.name.split(".", 1)[0]
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            labels = []
            for box in data["result"]:
                id = box["id"] if "id" in box else box["attribute"]
                id = 3 if str(id) == "4" else id  # convert 4 → 3 (handles int or str)
                width = float(box["width"]) / image_width
                height = float(box["height"]) / image_height
                x = (float(box["x"]) + float(box["width"]) / 2) / image_width
                y = (float(box["y"]) + float(box["height"]) / 2) / image_height

                labels.append([id, x, y, width, height])
        target = Path(output_dir) / f"{base_name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for item in labels:
                f.write(" ".join(str(x) for x in item) + "\n")

        # delete original file
        file.unlink()

    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)


def expand_training_set(dataset_dir, fract=0.25):
    val_images_dir = dataset_dir + "/images/val"
    val_labels_dir = dataset_dir + "/labels/val"

    train_images_dir = dataset_dir + "/images/train"
    train_labels_dir = dataset_dir + "/labels/train"

    val_images = [f for f in Path(val_images_dir).rglob("*") if f.is_file()]
    random.shuffle(val_images)

    split_idx = int(len(val_images) * fract)
    # val_files = files[:split_idx]
    train_images = val_images[split_idx:]

    for image in tqdm(train_images, desc="Expand training dataset", unit="file"):
        if image.is_file():
            target_image = Path(train_images_dir) / image.name
            label = Path(val_labels_dir) / f"{image.stem}.txt"
            target_label = Path(train_labels_dir) / f"{image.stem}.txt"

            # Move file
            try:
                shutil.move(str(image), str(target_image))
                shutil.move(str(label), str(target_label))
            except Exception as e:
                print(f"Failed to move {image.name}: {e}")


def convert_labels_safe(input_dir, output_dir):
    input_dir = Path(input_dir)
    files = [f for f in input_dir.rglob("*.json") if f.is_file()]

    for file in tqdm(files, desc="Convert labels", unit="file"):
        rel_path = file.relative_to(input_dir)
        base_name = file.stem.replace(".jpg", "")   # remove .jpg.json

        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        labels = []
        for box in data["result"]:
            id = box["id"] if "id" in box else box.get("attribute", 0)
            id = 3 if str(id) == "4" else id  # convert 4 → 3

            width  = float(box["width"])  / image_width
            height = float(box["height"]) / image_height
            x = (float(box["x"]) + float(box["width"])  / 2) / image_width
            y = (float(box["y"]) + float(box["height"]) / 2) / image_height

            labels.append([id, x, y, width, height])

        # keep same folder structure
        target = Path(output_dir) / rel_path.parent / f"{base_name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            for item in labels:
                f.write(" ".join(f"{x:.6f}" for x in item) + "\n")



def convert(dataset_dir):
    # Labels paths (your real structure)
    input_train_labels = dataset_dir + "/cipo/cipo/training"
    input_val_labels   = dataset_dir + "/cipo/cipo/validation"

    # Output YOLO labels (new safe folder)
    output_train_labels = dataset_dir + "/labels_yolo/train"
    output_val_labels   = dataset_dir + "/labels_yolo/val"

    convert_labels_safe(input_train_labels, output_train_labels)
    convert_labels_safe(input_val_labels, output_val_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", help="dataset directory")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    convert(dataset_dir)
    # expand_training_set(dataset_dir)
