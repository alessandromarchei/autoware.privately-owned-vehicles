import cv2
import argparse
from pathlib import Path


def image_to_label_path(image_path: Path) -> Path:
    """
    Map:
      .../images/training/.../img.jpg   -> .../labels_yolo/train/.../img.txt
      .../images/validation/.../img.jpg -> .../labels_yolo/val/.../img.txt
    """
    parts = list(image_path.parts)

    if "images" not in parts:
        raise ValueError(f"'images' not found in path: {image_path}")

    idx = parts.index("images")

    # Replace root folder
    parts[idx] = "labels_yolo"

    # Replace split name
    if parts[idx + 1] == "training":
        parts[idx + 1] = "train"
    elif parts[idx + 1] == "validation":
        parts[idx + 1] = "val"
    else:
        raise ValueError(f"Unknown split folder: {parts[idx + 1]}")

    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def draw_bbox(image_path: str):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img_h, img_w = image.shape[:2]

    label_path = image_to_label_path(image_path)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines

            cls, x, y, w, h = map(float, parts)

            # Convert to pixel coordinates
            x_center_px = x * img_w
            y_center_px = y * img_h
            w_px = w * img_w
            h_px = h * img_h

            # Top-left and bottom-right
            x1 = int(x_center_px - w_px / 2)
            y1 = int(y_center_px - h_px / 2)
            x2 = int(x_center_px + w_px / 2)
            y2 = int(y_center_px + h_px / 2)

            # Clamp to image bounds (safety)
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2))
            y2 = max(0, min(img_h - 1, y2))

            # Draw rectangle + class id
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(
                image,
                f"class {int(cls)}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # save image with bbox in the folder where this script is located
    image_path = Path(__file__).parent / "test_image_bboxes.jpg"
    cv2.imwrite(str(image_path), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True, help="Path to image")
    args = parser.parse_args()

    draw_bbox(args.image_path)

#python Models/data_parsing/AutoSpeed/OpenLane/test_conversion.py -i=/home/sergey/DEV/AI/datasets/OpenLane/images/training/segment-12304907743194762419_1522_000_1542_000_with_camera_labels/151865300644567000.jpg