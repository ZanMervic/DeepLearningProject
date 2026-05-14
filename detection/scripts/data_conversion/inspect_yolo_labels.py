#!/usr/bin/env python3
"""
Quick sanity-check utility: draw YOLO boxes on random images.

Usage:
  python inspect_yolo_labels.py \
    --dataset /path/to/yolo_dataset \
    --split train \
    --out /path/to/debug_drawn \
    --n 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def find_images(img_dir: Path) -> list[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(img_dir.glob(f"*{ext}"))
    return sorted(files)


def draw_one(image_path: Path, label_path: Path, out_path: Path) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = map(float, parts[:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    images = find_images(img_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")
    sample = random.sample(images, min(args.n, len(images)))
    ok = 0
    for p in sample:
        label = lbl_dir / f"{p.stem}.txt"
        if draw_one(p, label, args.out / p.name):
            ok += 1
    print(f"Wrote {ok} debug images to {args.out}")


if __name__ == "__main__":
    main()
