import json
from pathlib import Path
from PIL import Image
import shutil


def convert_crowd_human_to_yolo(
    annotation_file: str,
    image_dir: str,
    out_image_dir: str,
    out_label_dir: str,
    box_type: str = "vbox",  # "vbox", "fbox", or "hbox"
):
    """
    Convert CrowdHuman dataset file structure and annotations to YOLO format.
    CrowdHuman annotation format: Each line in the annotation file is a JSON object with the following structure:
    {
        "ID": "0--Parade_Parade_0_1000.jpg",
        "gtboxes": [
            {
                "tag": "person",
                "vbox": [x1, y1, w, h],
                "fbox": [x1, y1, w, h],
                "hbox": [x1, y1, w, h],
                "extra": {"ignore": 0}
            },
            ...
        ]
    }
    YOLO format: Each line in the label file corresponds to one bounding box in the format:
    <class_id> <x_center> <y_center> <width> <height>
    where all coordinates are normalized to [0, 1] relative to the image dimensions.

    Args:
        annotation_file (str): Path to the CrowdHuman annotation file (e.g., annotation_train.odgt).
        image_dir (str): Directory containing the original images.
        out_image_dir (str): Directory to save the copied images for YOLO format.
        out_label_dir (str): Directory to save the generated YOLO label files.
        box_type (str): Type of bounding box to use ("vbox", "fbox", or "hbox"). Default is "vbox".
    Returns:
        None
    """
    image_dir = Path(image_dir)
    out_image_dir = Path(out_image_dir)
    out_label_dir = Path(out_label_dir)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        record = json.loads(line)
        image_id = record["ID"]

        image_path = image_dir / image_id
        if not image_path.exists():
            image_path = image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            print(f"Missing image: {image_id}")
            continue

        with Image.open(image_path) as img:
            W, H = img.size

        yolo_lines = []

        for gt in record["gtboxes"]:
            if gt.get("tag") != "person":
                continue

            extra = gt.get("extra", {})
            if extra.get("ignore", 0) == 1:
                continue

            if box_type not in gt:
                continue

            x, y, w, h = gt[box_type]

            # Clip to image bounds.
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + w)
            y2 = min(H, y + h)

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 2 or bh <= 2:
                continue

            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            bw_norm = bw / W
            bh_norm = bh / H

            yolo_lines.append(
                f"0 {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}"
            )

        out_img_path = out_image_dir / image_path.name
        out_lbl_path = out_label_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, out_img_path)
        out_lbl_path.write_text("\n".join(yolo_lines), encoding="utf-8")