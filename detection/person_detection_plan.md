# Person Detection Plan for Deep Learning Project

## 1. Task Definition

This part of the project is not generic person detection.

The actual task is:

> Occlusion-robust **visible-person detection** for a mobile shooting / hit-localization pipeline.

The full pipeline is:

```text
captured image after player presses "shoot"
→ person detection
→ select candidate person near image center / crosshair
→ body-part segmentation / human parsing
→ determine hit body part
→ ReID using visible body parts
````

The detector should find visible people or visible person regions, including partially occluded people.

Important design decision:

```text
Use visible-person boxes for hit logic.
Do not use hallucinated full-body boxes as the main hit target.
```

Reason:

If only a leg is visible behind a wall, a full-body box may extend behind the wall. We do not want the system to count a hit through an occluder.

Therefore:

```text
visible box = used for hit detection and crop selection
full-body box = optional ablation / auxiliary context only
```

---

## 2. What Already Exists

This problem exists in the literature under names such as:

```text
occluded pedestrian detection
crowded human detection
partial-person detection
dense pedestrian detection
```

The closest datasets are:

* CrowdHuman
* WiderPerson
* OCHuman
* COCO persons

The closest model families are:

* YOLO26
* D-FINE
* RT-DETR / RT-DETRv2
* older two-stage detectors such as Faster R-CNN / Cascade R-CNN

For this project, YOLO26 is the most practical starting point because it has strong pretrained models, simple fine-tuning, and easier deployment.

---

## 3. Main Datasets

### 3.1 CrowdHuman — primary dataset

Use this first.

Why:

```text
- built for crowded / occluded human detection
- provides visible-region boxes
- provides full-body boxes
- provides head boxes
- large enough for real fine-tuning
```

Use:

```text
vbox = visible person box
```

Do not use `fbox` as the main training label unless doing a specific ablation.

Main use:

```text
YOLO26 pretrained on COCO
→ fine-tune on CrowdHuman vbox
```

Useful ablation:

```text
YOLO26 fine-tuned on CrowdHuman vbox
vs
YOLO26 fine-tuned on CrowdHuman fbox
```

Expected result:

```text
vbox should be better for hit validation.
fbox may create false hits through occluders.
```

---

### 3.2 COCO persons — useful but not first priority

COCO persons can be used because COCO contains person annotations and instance segmentation masks.

However, most YOLO26 pretrained weights are already trained on COCO, so adding COCO again is less important than adding occlusion-specific datasets.

Use COCO if:

```text
- you want more general person diversity
- you derive visible boxes from person segmentation masks
- you train a merged dataset later
```

Do not let COCO dominate the final training if your main target is occluded / partially visible people.

---

### 3.3 WiderPerson — optional diversity dataset

Use after CrowdHuman works.

Why:

```text
- many dense person / pedestrian scenes
- includes partially visible persons
- adds diversity beyond CrowdHuman
```

Suggested mapping:

```text
pedestrian → person
rider → person, optional
partially-visible person → person
ignore region → ignore / skip
crowd region → ignore / skip
```

Be careful with ignore regions. If ignored regions are treated as background, the model may learn wrong negatives.

---

### 3.4 OCHuman — hard occlusion dataset

Use mainly for hard evaluation or final fine-tuning.

Why:

```text
- heavily occluded humans
- useful stress test
- smaller than CrowdHuman / WiderPerson
```

Best use:

```text
derive visible boxes from instance masks, where available
```

Avoid blindly mixing OCHuman raw bboxes unless you verify that their box semantics match your target.

---

### 3.5 Custom phone-shot test set — mandatory

Public datasets are not enough to prove the app works.

Create a small custom test set:

```text
100–300 images
captured with phone-like camera
players partially visible
center/crosshair sometimes on person, sometimes not
different lighting, distance, poses, occluders
```

Label:

```text
visible person boxes
whether image center is on a visible person/body part
optional: body part at center
```

This is the final judge of whether the detector is useful for the app.

---

## 4. Model Plan

### 4.1 Mobile-friendly models

Use:

```text
YOLO26n
YOLO26s
```

Suggested experiments:

```text
YOLO26n @ 640
YOLO26n @ 960
YOLO26s @ 640
YOLO26s @ 960
```

Likely best practical options:

```text
YOLO26s @ 640 = good speed/accuracy balance
YOLO26n @ 960 = lightweight model with better small-object visibility
YOLO26s @ 960 = stronger but heavier
```

Because inference happens only when the player presses shoot, the model does not need to run every video frame. This allows a slightly larger model or larger input size.

---

### 4.2 High-accuracy reference models

Use one of these as an accuracy ceiling:

```text
YOLO26l
YOLO26x
D-FINE-L / D-FINE-X
RT-DETRv2
```

Recommended:

```text
YOLO26x = easiest high-accuracy comparison inside same framework
D-FINE = stronger research comparison if time allows
```

Do not start with D-FINE / RT-DETRv2 unless the YOLO pipeline already works.

---

## 5. Input Size Strategy

Input size matters because partially visible people may become very small after resizing.

Test:

```text
640
960
1280, optional
```

Compute roughly scales quadratically:

```text
960 vs 640  ≈ 2.25× more pixels
1280 vs 640 ≈ 4× more pixels
```

Recommended first tests:

```text
YOLO26s @ 640
YOLO26s @ 960
YOLO26n @ 960
```

If the detector misses small visible body parts, increasing input size may help more than using a larger model at lower resolution.

Use letterbox resizing, not naive stretching:

```text
preserve aspect ratio
resize image to fit model input
pad remaining area
run detector
map boxes back to original image coordinates
```

The center/crosshair logic must use original-image coordinates, not padded model coordinates.

---

## 6. Inference Logic

Recommended inference after the player presses shoot:

```text
1. Capture image.
2. Run visible-person detector on full image.
3. Find boxes whose visible box contains the image center.
4. If no box contains center, check slightly expanded visible boxes.
5. If multiple candidates exist, pass all plausible candidates to segmentation.
6. Let body-part segmentation decide whether the center is actually on visible body pixels.
7. If segmentation says background/occluder/wall, count as miss.
8. If segmentation says visible body part, continue to ReID.
```

Expanded-box rule:

```text
expanded_box = visible_box enlarged by 10–25%
```

Do not expand too much. Expansion is only to compensate for detector imprecision, not to infer invisible body area.

Optional fallback:

```text
If no detection near center:
    run detector again on a high-resolution center crop
```

This is better than using a small detector as a hard gate.

---

## 7. Fine-Tuning Strategy

### 7.1 Core idea

Fine-tuning means:

```text
start from pretrained YOLO26 weights
→ continue training on target dataset
→ adapt model to occlusion / visible-person boxes
```

Do not train from scratch.

Use:

```python
YOLO("yolo26s.pt")
```

not a randomly initialized model.

---

### 7.2 Dataset format

Ultralytics YOLO detection format:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Each label file:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0, 1]`.

For this project:

```text
class_id = 0
class name = person
```

Example label:

```text
0 0.512331 0.438221 0.130442 0.381245
```

---

### 7.3 data.yaml

Example:

```yaml
path: /absolute/path/to/person_detection_dataset

train: images/train
val: images/val

names:
  0: person
```

Before training, verify that:

```text
- image paths are correct
- label paths are correct
- class IDs match data.yaml
- every label coordinate is between 0 and 1
```

---

## 8. CrowdHuman Conversion Plan

CrowdHuman annotations contain multiple box types:

```text
vbox = visible-region box
fbox = full-body box
hbox = head box
```

For the main model, convert:

```text
vbox → YOLO person box
```

For the ablation model, convert:

```text
fbox → YOLO person box
```

Ignore:

```text
non-person tags
ignore boxes
invalid/tiny boxes
```

Conversion logic:

```python
import json
from pathlib import Path
from PIL import Image
import shutil


def convert_crowdhuman_to_yolo(
    annotation_file: str,
    image_dir: str,
    out_image_dir: str,
    out_label_dir: str,
    box_type: str = "vbox",  # "vbox", "fbox", or "hbox"
):
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
```

Example usage:

```python
convert_crowdhuman_to_yolo(
    annotation_file="CrowdHuman/annotation_train.odgt",
    image_dir="CrowdHuman/Images",
    out_image_dir="datasets/crowdhuman_vbox/images/train",
    out_label_dir="datasets/crowdhuman_vbox/labels/train",
    box_type="vbox",
)

convert_crowdhuman_to_yolo(
    annotation_file="CrowdHuman/annotation_val.odgt",
    image_dir="CrowdHuman/Images",
    out_image_dir="datasets/crowdhuman_vbox/images/val",
    out_label_dir="datasets/crowdhuman_vbox/labels/val",
    box_type="vbox",
)
```

---

## 9. Label Sanity Check

Before training, draw boxes on random images.

```python
import cv2
from pathlib import Path


def draw_yolo_labels(image_path, label_path, out_path):
    img = cv2.imread(str(image_path))
    H, W = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * W)
            y1 = int((yc - bh / 2) * H)
            x2 = int((xc + bw / 2) * W)
            y2 = int((yc + bh / 2) * H)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(str(out_path), img)
```

Check at least 50 images manually.

Common conversion errors:

```text
- using fbox when you intended vbox
- wrong image dimensions
- x/y swapped
- labels not normalized
- labels shifted because of wrong coordinate convention
- ignore regions included as real people
```

---

## 10. Main Training Commands

Install:

```bash
pip install ultralytics
```

Train YOLO26s on CrowdHuman visible boxes:

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(
    data="datasets/crowdhuman_vbox/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    patience=15,
    project="runs/person_detection",
    name="yolo26s_crowdhuman_vbox_640",
)
```

Train higher input size:

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(
    data="datasets/crowdhuman_vbox/data.yaml",
    epochs=50,
    imgsz=960,
    batch=8,
    device=0,
    patience=15,
    project="runs/person_detection",
    name="yolo26s_crowdhuman_vbox_960",
)
```

Train mobile-light model:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="datasets/crowdhuman_vbox/data.yaml",
    epochs=50,
    imgsz=960,
    batch=8,
    device=0,
    patience=15,
    project="runs/person_detection",
    name="yolo26n_crowdhuman_vbox_960",
)
```

If GPU memory fails:

```text
reduce batch size
reduce input size
switch from YOLO26s to YOLO26n
use gradient accumulation if needed
```

---

## 11. Validation

Validate pretrained baseline:

```python
from ultralytics import YOLO

baseline = YOLO("yolo26s.pt")

baseline.val(
    data="datasets/crowdhuman_vbox/data.yaml",
    imgsz=640,
    device=0,
)
```

Validate fine-tuned model:

```python
from ultralytics import YOLO

model = YOLO("runs/person_detection/yolo26s_crowdhuman_vbox_640/weights/best.pt")

model.val(
    data="datasets/crowdhuman_vbox/data.yaml",
    imgsz=640,
    device=0,
)
```

Compare:

```text
COCO-pretrained YOLO26s
vs
CrowdHuman-vbox-fine-tuned YOLO26s
```

---

## 12. Metrics to Report

### Standard metrics

```text
mAP50
mAP50-95
precision
recall
latency
model size
```

### Occlusion-specific metrics

Using CrowdHuman:

```text
visible_ratio = area(vbox) / area(fbox)
```

Report recall for:

```text
mostly visible: visible_ratio > 0.65
partially visible: 0.35 <= visible_ratio <= 0.65
heavily occluded: visible_ratio < 0.35
```

### App-specific metrics

Most important:

```text
center-target recall
```

Definition:

```text
If the center/crosshair lies on or near a true visible-person box,
did the detector return a matching detection?
```

Also useful:

```text
false hit through occluder
```

Definition:

```text
center is inside predicted box
but center is not inside true visible-person box
```

This directly tests why visible boxes are better than full-body boxes.

---

## 13. Required Experiments

### Experiment 1 — pretrained vs fine-tuned

```text
YOLO26s pretrained on COCO
vs
YOLO26s fine-tuned on CrowdHuman vbox
```

Goal:

```text
show that fine-tuning improves occluded / partial-person detection
```

---

### Experiment 2 — mobile model vs stronger model

```text
YOLO26n
vs
YOLO26s
```

Suggested settings:

```text
YOLO26n @ 960
YOLO26s @ 640
YOLO26s @ 960
```

Goal:

```text
find realistic mobile/accuracy tradeoff
```

---

### Experiment 3 — input size

```text
640 vs 960
```

Optional:

```text
1280
```

Goal:

```text
test whether larger input size improves small / partial-person recall
```

---

### Experiment 4 — visible box vs full-body box

Train:

```text
YOLO26s on CrowdHuman vbox
YOLO26s on CrowdHuman fbox
```

Goal:

```text
prove that visible boxes are safer for hit detection
```

Expected:

```text
vbox improves hit validity
fbox may create false positives through occluders
```

---

### Experiment 5 — custom phone-shot evaluation

Evaluate all final candidates on custom phone-shot images.

Goal:

```text
show whether public-dataset improvements transfer to your actual app
```

This should be treated as the final decision metric.

---

## 14. Optional: Merged Dataset Fine-Tuning

Do this only after the CrowdHuman pipeline works.

### 14.1 Why merge?

Possible benefits:

```text
more data
more scene diversity
more partial-person examples
better generalization
```

Possible risks:

```text
inconsistent box definitions
ignore-region mistakes
COCO easy examples diluting occlusion training
OCHuman too small to matter unless oversampled
domain mismatch
```

The goal is not simply "more data." The goal is more useful data with consistent labels.

---

### 14.2 Unified target definition

All datasets should be converted to:

```text
class: person
box: visible person region
```

Dataset mapping:

```text
CrowdHuman:
    use vbox

COCO:
    use person bbox or derive bbox from person segmentation mask
    skip iscrowd where appropriate

WiderPerson:
    pedestrian → person
    rider → optional person
    partially-visible person → person
    ignore/crowd regions → skip or ignore

OCHuman:
    preferably derive visible boxes from masks
    otherwise inspect bboxes carefully before use
```

Do not mix:

```text
CrowdHuman vbox
+ CrowdHuman fbox
+ loose OCHuman bbox
+ COCO mask-derived visible bbox
```

unless you explicitly know they have the same meaning.

---

### 14.3 Merged training options

#### Option A — train directly on merged dataset

```text
CrowdHuman vbox
+ COCO persons
+ WiderPerson
+ OCHuman visible boxes
```

Pros:

```text
simple
more data
```

Cons:

```text
risk of inconsistent labels
risk of easy examples dominating
```

---

#### Option B — staged fine-tuning

Recommended if time allows:

```text
Stage 1:
fine-tune on broad merged dataset:
CrowdHuman + COCO + WiderPerson

Stage 2:
short final fine-tune on hard data:
CrowdHuman + OCHuman + custom hard cases
```

Why:

```text
Stage 1 teaches broad person detection.
Stage 2 specializes back toward occlusion / visible-person hit detection.
```

---

#### Option C — ablation-based merge

Best for report:

```text
Model A: CrowdHuman only
Model B: CrowdHuman + COCO
Model C: CrowdHuman + COCO + WiderPerson
Model D: CrowdHuman + COCO + WiderPerson + OCHuman
```

Evaluate each separately on:

```text
CrowdHuman val
OCHuman hard set
custom phone-shot set
```

Keep the merged model only if it improves the custom test set.

---

### 14.4 Sampling strategy

Avoid letting easy datasets dominate.

Possible strategy:

```text
50% CrowdHuman
25% COCO / WiderPerson
25% OCHuman / custom hard cases
```

If OCHuman is too small:

```text
oversample OCHuman
or use it only in final hard-case fine-tuning
```

---

## 15. Mobile / Deployment Plan

After choosing the best mobile candidate:

```text
likely YOLO26n @ 960
or YOLO26s @ 640/960
```

Export:

```python
from ultralytics import YOLO

model = YOLO("runs/person_detection/best_model/weights/best.pt")

model.export(format="onnx")
model.export(format="tflite", int8=True)
model.export(format="coreml")
```

Compare exported model against PyTorch model:

```text
PyTorch best.pt
vs
ONNX
vs
TFLite/CoreML
```

Report:

```text
accuracy drop
latency
model size
```

Quantization can improve speed and size but may reduce accuracy. Always validate the exported model.

---

## 16. Final Recommended Scope

Minimum strong project scope:

```text
1. Fine-tune YOLO26s on CrowdHuman visible boxes.
2. Compare pretrained vs fine-tuned.
3. Compare YOLO26n vs YOLO26s.
4. Compare 640 vs 960 input size.
5. Compare visible-box vs full-body-box training.
6. Evaluate on a custom phone-shot center-hit test set.
```

Optional if time allows:

```text
7. Try merged dataset fine-tuning.
8. Try YOLO26x or D-FINE as an accuracy reference.
9. Export mobile model and benchmark ONNX/TFLite/CoreML.
```

Main expected contribution:

```text
A detector specialized for visible, partially occluded humans,
evaluated not only by generic mAP but also by center-hit recall
and false-hit-through-occlusion behavior.
```