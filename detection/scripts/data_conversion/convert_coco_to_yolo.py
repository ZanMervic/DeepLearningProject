import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


PERSON_CATEGORY_ID = 1


def coco_xywh_to_yolo(x, y, w, h, img_w, img_h):
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(img_w), float(x) + float(w))
    y2 = min(float(img_h), float(y) + float(h))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 1 or bh <= 1:
        return None

    xc = (x1 + bw / 2.0) / img_w
    yc = (y1 + bh / 2.0) / img_h
    wn = bw / img_w
    hn = bh / img_h

    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
        return None

    return xc, yc, wn, hn


def convert_split(
    images_dir: Path,
    ann_json: Path,
    out_dir: Path,
    split_name: str,
    skip_iscrowd: bool = True,
    copy_images: bool = True,
):
    image_out_dir = out_dir / "images" / split_name
    label_out_dir = out_dir / "labels" / split_name

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("category_id") != PERSON_CATEGORY_ID:
            continue

        if skip_iscrowd and ann.get("iscrowd", 0) == 1:
            continue

        anns_by_image[ann["image_id"]].append(ann)

    saved_images = 0
    saved_boxes = 0
    missing_images = 0

    for image_id, anns in tqdm(anns_by_image.items(), desc=f"Converting {split_name}"):
        img_info = images_by_id.get(image_id)
        if img_info is None:
            continue

        file_name = img_info["file_name"]
        img_w = int(img_info["width"])
        img_h = int(img_info["height"])

        src_img = images_dir / file_name
        if not src_img.exists():
            missing_images += 1
            continue

        yolo_lines = []

        for ann in anns:
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            yolo_box = coco_xywh_to_yolo(*bbox, img_w=img_w, img_h=img_h)
            if yolo_box is None:
                continue

            xc, yc, bw, bh = yolo_box
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        dst_img = image_out_dir / file_name
        dst_lbl = label_out_dir / f"{Path(file_name).stem}.txt"

        if copy_images:
            shutil.copy2(src_img, dst_img)
        else:
            # Optional: create hardlinks to save disk space.
            try:
                dst_img.hardlink_to(src_img.resolve())
            except Exception:
                shutil.copy2(src_img, dst_img)

        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")

        saved_images += 1
        saved_boxes += len(yolo_lines)

    print(
        f"[{split_name}] saved_images={saved_images}, "
        f"saved_boxes={saved_boxes}, missing_images={missing_images}"
    )


def write_yaml(out_dir: Path):
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Wrote {out_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep-crowd", action="store_true")
    parser.add_argument("--hardlink", action="store_true")

    args = parser.parse_args()

    convert_split(
        images_dir=args.coco_root / "train2017",
        ann_json=args.coco_root / "annotations" / "instances_train2017.json",
        out_dir=args.out,
        split_name="train",
        skip_iscrowd=not args.keep_crowd,
        copy_images=not args.hardlink,
    )

    convert_split(
        images_dir=args.coco_root / "val2017",
        ann_json=args.coco_root / "annotations" / "instances_val2017.json",
        out_dir=args.out,
        split_name="val",
        skip_iscrowd=not args.keep_crowd,
        copy_images=not args.hardlink,
    )

    write_yaml(args.out)


if __name__ == "__main__":
    main()