# Person Detection Subproject Documentation

**Project context:** deep-learning-based mobile/near-mobile person detection for a larger camera-based game pipeline.

**Detection role in the full system:** given a single frame captured when the player presses “shoot”, detect visible people in the image, pass the relevant person/candidate region to body-part segmentation, and later support person re-identification. The detection model should work when only parts of the person are visible, because the downstream app needs to determine whether a visible body part was hit.

---

## 1. High-level pipeline and problem definition

The planned full pipeline is:

```text
Input image from phone camera
  → person detection
  → select candidate(s), especially around image center / crosshair
  → body-part segmentation of selected person
  → ReID using visible body regions
  → game decision: was the correct visible body part / person hit?
```

The detection subproblem is therefore not generic “detect all COCO objects”. It is specifically:

```text
Detect visible person regions robustly, including partial and occluded people, with a model small enough to be a plausible mobile candidate.
```

Important design decision: the detector should predict the **visible person region**, not hallucinate a full-body bounding box behind occluders. Earlier we considered whether a detector should infer the full body box even when only a leg is visible, because then checking whether the crosshair intersects the person would be easy. We rejected that as the main target because it would create physically incorrect game behavior: the player could appear to hit a person “through a wall” or behind an occluder. Therefore, the model target is the visible region/person extent that is actually observable in the image.

Another important design decision: the model does **not** need to run continuously in real time on every camera frame. The intended app can process a single captured frame after the shoot action. This relaxes latency requirements substantially and makes a somewhat heavier model or higher resolution more realistic, while still keeping mobile deployment in mind.

---

## 2. Model-family decision: why YOLO26 was used

### 2.1 Requirements

The detection model needs to satisfy several constraints:

- strong person-detection performance;
- robustness to occlusion and partial visibility;
- reasonably fast inference;
- exportability to mobile or edge formats;
- easy fine-tuning on custom YOLO-format datasets;
- manageable implementation effort for a course project.

### 2.2 CNN-style YOLO detector vs transformer detector

We considered modern transformer detectors and open-vocabulary detectors, especially:

- **Grounding DINO**: an open-set detector combining a transformer detector with language grounding. It can detect arbitrary text-specified categories such as “person” and is powerful for zero-shot or pseudo-labeling use cases. However, it is not primarily a small mobile detector and is unnecessarily general for a fixed one-class person-detection problem.
- **RF-DETR**: a recent real-time detection transformer using neural architecture search to find accuracy-latency tradeoffs. It is interesting as a modern comparison baseline, but switching the project to RF-DETR would add additional framework/setup risk and is not necessary for the core project objective.

We chose **Ultralytics YOLO26** as the main model family because it directly supports training, validation, export, and deployment workflows, and because the YOLO26 family includes multiple model sizes (`n`, `s`, `m`, `l`, `x`) with clear accuracy/speed tradeoffs. YOLO26 is also explicitly positioned by Ultralytics for edge and low-power deployment, with an end-to-end NMS-free design and export support to formats such as ONNX, CoreML, TFLite, TensorRT, and OpenVINO.

Relevant YOLO26 model-size facts from Ultralytics documentation:

| Model | Params | FLOPs @640 | COCO mAP50-95 |
|---|---:|---:|---:|
| YOLO26n | 2.4M | 5.4B | 40.9 |
| YOLO26s | 9.5M | 20.7B | 48.6 |
| YOLO26m | 20.4M | 68.2B | 53.1 |
| YOLO26l | 24.8M | 86.4B | 55.0 |
| YOLO26x | 55.7M | 193.9B | 57.5 |

This made `yolo26n` and `yolo26s` the most relevant models for the project. Larger models are useful as possible upper-bound references, but they are much less relevant for mobile deployment. The `x` model has roughly 36× the FLOPs of `n` and 9× the FLOPs of `s` at the same input size, which is hard to justify for the final app.

### 2.3 One-class detector, not “person vs not-person”

The model is trained as a **single-class detector**:

```yaml
names:
  0: person
```

We specifically did **not** add a separate `not-person` class. In object detection, background is learned implicitly from regions without labeled objects. Adding a `not-person` class would require arbitrarily annotating background objects, which is ill-defined and would likely degrade training.

Ultralytics automatically adapts the detection head from the COCO pretrained class count (`nc=80`) to the dataset class count (`nc=1`) during fine-tuning. This was confirmed in logs by messages such as:

```text
Overriding model.yaml nc=80 with nc=1
```

This means the model is not wasting its classification head on 80 COCO classes after fine-tuning. Most compute still remains in the backbone/neck feature extraction, but the task is correctly defined as person-only detection.

---

## 3. Dataset selection

The datasets were selected to cover different parts of the target problem: visible people, occlusion, crowds, pedestrians, and general person appearance.

### 3.1 CrowdHuman

CrowdHuman is highly relevant because it was built for crowded human detection and provides multiple box types, including visible boxes. We used the **visible bounding box (`vbox`)** annotations because our desired target is the visible person region, not the full body behind occluders.

Role in project:

```text
Main occlusion/crowd training dataset and baseline dataset.
```

Why useful:

- many crowded scenes;
- explicit visible-region annotation;
- directly aligned with the “do not shoot through walls/occluders” decision.

### 3.2 WiderPerson

WiderPerson adds pedestrian/person diversity and partial visibility. It is useful for generalizing beyond CrowdHuman’s distribution. During conversion, person-like categories were mapped to the single class `person`; ignore/crowd categories should not be treated as standard person boxes.

Role in project:

```text
Second training dataset for pedestrian and partial-person diversity.
```

### 3.3 COCO person subset / COCO2017 person

COCO provides broad, general-purpose person instances in varied scenes. It is less specialized for occluded visible-person detection, but it adds visual diversity and generalization.

Role in project:

```text
General person-detection diversity dataset.
```

Potential issue:

```text
COCO box style may differ from CrowdHuman visible boxes; this can create annotation-style mismatch.
```

### 3.4 OCHuman

OCHuman contains heavily occluded people and is especially relevant to the target app because partial visibility is central to the task. However, OCHuman was originally often used as a challenging validation/testing benchmark for occlusion robustness. This creates two valid uses:

| Use | Interpretation |
|---|---|
| Hold OCHuman out for evaluation | Measures generalization to a hard occlusion benchmark. |
| Include OCHuman in training | Makes sense for production if occlusion is central to the target use case. |

It is not “bad” to train on OCHuman. The only caveat is methodological: if OCHuman is included in training, then OCHuman validation performance is no longer a clean unseen-dataset generalization claim. For production, using all relevant occlusion data is reasonable; for a clean benchmark, keeping OCHuman held out is more convincing.

Role in project:

```text
Hard occlusion evaluation set and optional production-oriented training data.
```

---

## 4. Data preparation and label compatibility

### 4.1 Unified YOLO format

All datasets were converted to YOLO-compatible detection format:

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

Each label file contains normalized YOLO rows:

```text
class_id x_center y_center width height
```

All datasets use a single class:

```yaml
names:
  0: person
```

Therefore, the datasets are **class-compatible**. The main remaining concern is not class mismatch, but **bounding-box semantic mismatch**.

### 4.2 Bounding-box semantic differences

Although every dataset is mapped to `class 0 = person`, the meaning of a “person box” can differ:

| Dataset | Approximate box semantics |
|---|---|
| CrowdHuman | Visible person region (`vbox`) was used. |
| WiderPerson | Pedestrian/person/partial-person boxes; likely visible-person-like but not identical. |
| COCO person | General person instance boxes; can differ from visible-box conventions. |
| OCHuman | Heavily occluded people; depending on conversion, boxes may be mask-derived and very tight to visible regions. |

This is important because training on mixed datasets can improve generality but slightly reduce performance on the original dataset due to inconsistent localization targets. The effect is especially visible in stricter metrics such as mAP50-95.

### 4.3 Visual inspection

A visual inspection step was included because most detection-training failures come from silent label-conversion errors. The inspection script overlays YOLO labels on images so we can verify:

- boxes are aligned with people;
- boxes are not shifted or scaled incorrectly;
- class IDs are all `0`;
- ignore/crowd regions are not incorrectly labeled as normal persons;
- train/val folders have matching images and labels.

This was important before launching long HPC fine-tuning jobs.

---

## 5. Training setup

### 5.1 Fine-tuning approach

The models were initialized from pretrained YOLO26 checkpoints (`yolo26n.pt`, `yolo26s.pt`, etc.) and fine-tuned on the custom one-class person datasets.

Fine-tuning does not merely train a new final layer. Unless layers are explicitly frozen, Ultralytics updates the trainable weights throughout the model. This is useful because the model needs to adapt not only to the `person` class, but also to the target annotation style: visible/occluded person regions.

Training used validation during training for early stopping/model selection. Ultralytics saves:

```text
runs/person_detection/<run_name>/weights/best.pt
runs/person_detection/<run_name>/weights/last.pt
```

`best.pt` should be used for reporting/evaluation. `last.pt` is mainly useful for resuming interrupted training.

### 5.2 Input size

Experiments were run at both `imgsz=640` and `imgsz=960`. The completed training set included a systematic `yolo26n @640` dataset-composition ablation, selected `yolo26n @960` runs, and one stronger `yolo26s @960` fine-tuned model.

Reasoning:

- `640` is cheaper and closer to typical YOLO default workflows;
- `960` may help with small or partially visible persons because more pixels are available for the same object;
- compute scales approximately with image area, so `960` costs about `(960/640)^2 = 2.25×` more raw image compute than `640`.

### 5.3 Model sizes

Main focus:

```text
yolo26n: mobile candidate
yolo26s: stronger but still practical model
```

Larger models were discussed (`m`, `l`, `x`), but not prioritized because:

- they are much more compute-intensive;
- they are less realistic for mobile deployment;
- the project goal is not simply to maximize benchmark mAP with an impractically large model;
- dataset composition and evaluation design are more important for this project than brute-force model size.

### 5.4 Merged dataset training

Rather than physically copying datasets together, merged YAML files were created using lists of image folders:

```yaml
path: /d/hpc/projects/FRI/zm3587/dl/datasets

train:
  - crowdhuman/images/train
  - widerperson/images/train
  - coco2017/images/train

val:
  - crowdhuman/images/val
  - widerperson/images/val
  - coco2017/images/val

names:
  0: person
```

This avoids duplicating data and makes training recipes explicit.

The systematic dataset recipes were:

```text
CH
CH + WP
CH + WP + COCO
CH + WP + COCO + OCH
```

The scientific reason for this sequence is to test whether each additional dataset improves generalization or simply causes domain interference.

---

## 6. HPC and engineering issues solved

A significant part of the project was building a reliable training/evaluation workflow on HPC. This is worth mentioning in the report because it shows practical engineering work, not just running a notebook.

### 6.1 Conda and Python package isolation

An issue occurred where packages were silently imported from:

```text
~/.local/lib/python3.11/site-packages
```

instead of the intended conda environment. This caused inconsistencies between interactive shell behavior and Slurm job behavior. The solution was to set:

```bash
export PYTHONNOUSERSITE=1
```

and install all required packages into the conda environment itself. This made the environment reproducible and prevented user-site package leakage.

### 6.2 V100 compatibility issue

The HPC GPU nodes used Tesla V100S GPUs. A newer PyTorch/CUDA wheel was initially installed, but it did not include kernels for V100 compute capability (`sm_70`). The error was:

```text
CUDA error: no kernel image is available for execution on the device
```

The solution was to create a V100-compatible environment using a CUDA 11.8 PyTorch wheel, because the newer PyTorch build supported newer compute capabilities but not V100’s compute capability.

A clean environment was created for V100 training and tested inside an actual Slurm GPU allocation before training.

### 6.3 Slurm scripts

Reusable Slurm scripts were created for:

- single training jobs;
- parameterized training jobs;
- merged-dataset training;
- evaluating a model on one dataset;
- evaluating a model on all datasets.

The scripts use positional parameters with defaults so that many models/datasets can be tested without creating a separate `.sbatch` file for every run.

Example argument structure for training:

```text
MODEL IMGSZ DATASET_KEY BATCH NAME EPOCHS PATIENCE FREEZE
```

Example argument structure for evaluation:

```text
MODEL IMGSZ DATASET_GROUP BATCH NAME_PREFIX
```

### 6.4 Dataset cache race issue

One training job failed before training started due to an Ultralytics dataset cache issue:

```text
AssertionError during cache hash check
FileNotFoundError: labels/val.cache
```

This was diagnosed as a likely stale cache or race condition where multiple jobs touched the same dataset `.cache` files simultaneously. The solution was to delete cache files and rerun the failed job, preferably avoiding simultaneous first-time cache generation for multiple jobs using the same dataset.

### 6.5 Run directory management

Ultralytics creates new directories when a run name already exists, e.g.:

```text
yolo26s_crowdhuman_960
yolo26s_crowdhuman_960-2
```

This is intentional overwrite protection. The completed run must be identified by checking for:

```text
weights/best.pt
weights/last.pt
results.csv
```

The `.pt` files in the project root, such as `yolo26n.pt`, are pretrained starting checkpoints, not fine-tuned results. Fine-tuned models are stored inside the run folders.

---

## 7. Evaluation methodology

### 7.1 Why separate validation sets are better than merged validation only

Evaluating on one merged validation set gives a single average number, but it hides where the model works or fails. Therefore, each model was evaluated separately on:

```text
CrowdHuman validation
WiderPerson validation
COCO2017/person validation
OCHuman validation
```

This is more informative because it reveals whether a model improved broadly or only became specialized to one dataset.

### 7.2 Metrics used

The main reported metrics are:

| Metric | Interpretation |
|---|---|
| Precision | Of predicted detections, how many are correct? |
| Recall | Of true persons, how many were detected? |
| mAP50 | Detection AP at IoU threshold 0.50; lenient localization metric. |
| mAP50-95 | Average mAP over IoU thresholds 0.50 to 0.95; stricter and more sensitive to box quality. |
| Inference time | Approximate model inference speed in validation environment. |

For this app, **recall** is especially important. If the detector misses a partially visible person, downstream segmentation/ReID cannot recover that person. Some false positives are less harmful because later stages can reject them.

### 7.3 Evaluation scripts

The evaluator was modified to run one model against multiple dataset YAMLs and produce:

```text
CSV table of metrics
JSON details
Markdown summary table
per-dataset Ultralytics output folders with plots
```

This made it easier to compare models systematically rather than manually launching many separate evaluation jobs.

### 7.4 Completed experiment inventory

The original plan contained several possible experiment families: pretrained vs fine-tuned comparison, mobile model vs stronger model, input-size comparison, visible-box vs full-body-box ablation, custom phone-shot evaluation, merged-dataset fine-tuning, and optional high-accuracy reference models. The completed detection experiments focused most heavily on the **mobile-relevant YOLO26n models**, dataset-composition ablations, and cross-dataset evaluation.

Completed fine-tuning runs:

| # | Model | Input size | Training data | Purpose |
|---:|---|---:|---|---|
| 1 | YOLO26n | 640 | CrowdHuman | baseline visible-person fine-tune |
| 2 | YOLO26n | 640 | CrowdHuman + WiderPerson | test effect of adding pedestrian/partial-person diversity |
| 3 | YOLO26n | 640 | CrowdHuman + WiderPerson + COCO2017 person | test effect of adding general person diversity |
| 4 | YOLO26n | 640 | CrowdHuman + WiderPerson + COCO2017 person + OCHuman | test effect of adding hard occlusion data |
| 5 | YOLO26n | 960 | CrowdHuman | input-size comparison for the primary dataset |
| 6 | YOLO26n | 960 | CrowdHuman + WiderPerson + COCO2017 person + OCHuman | higher-resolution all-data mobile candidate |
| 7 | YOLO26s | 960 | CrowdHuman | stronger fine-tuned reference model |

Completed evaluation runs:

Each of the following model checkpoints was evaluated separately on all four validation datasets: CrowdHuman, WiderPerson, COCO2017 person, and OCHuman.

| # | Model / checkpoint type | Input size | Fine-tuned? | Training data | Evaluation sets | Purpose |
|---:|---|---:|---|---|---|---|
| 1 | YOLO26n | 640 | yes | CrowdHuman | all four datasets | CrowdHuman-only fine-tuned baseline |
| 2 | YOLO26n | 960 | yes | CrowdHuman | all four datasets | effect of higher input size on CrowdHuman fine-tune |
| 3 | YOLO26n | 640 | yes | CrowdHuman + WiderPerson | all four datasets | merged-data ablation |
| 4 | YOLO26n | 640 | yes | CrowdHuman + WiderPerson + COCO2017 person | all four datasets | merged-data ablation |
| 5 | YOLO26n | 640 | yes | CrowdHuman + WiderPerson + COCO2017 person + OCHuman | all four datasets | all-data ablation |
| 6 | YOLO26n | 960 | yes | CrowdHuman + WiderPerson + COCO2017 person + OCHuman | all four datasets | higher-resolution all-data candidate |
| 7 | YOLO26n | 640 | no | COCO-pretrained base checkpoint | all four datasets | pretrained baseline |
| 8 | YOLO26n | 960 | no | COCO-pretrained base checkpoint | all four datasets | pretrained baseline at higher input size |
| 9 | YOLO26s | 960 | yes | CrowdHuman | all four datasets | stronger fine-tuned reference |
| 10 | YOLO26s | 640 | no | COCO-pretrained base checkpoint | all four datasets | stronger pretrained baseline |
| 11 | YOLO26s | 960 | no | COCO-pretrained base checkpoint | all four datasets | stronger pretrained baseline at higher input size |
| 12 | YOLO26x | 960 | no | COCO-pretrained base checkpoint | all four datasets | high-capacity upper-bound reference without fine-tuning |

This means the final detection work deviated from the initial minimum plan in a useful direction: instead of only training one or two models, the project built a matrix of fine-tuned, pretrained, mobile-sized, stronger, and high-capacity reference checkpoints, then evaluated each across all public validation domains.

---

## 8. Current experimental results: YOLO26n @ 640

The most systematic completed result set is for `yolo26n` at input size `640`, trained on different dataset combinations.

Notation:

```text
CH = CrowdHuman
WP = WiderPerson
COCO = COCO2017 person subset
OCH = OCHuman
```

### 8.1 Full result table

| Train set | Eval set | mAP50 | mAP50-95 | Precision | Recall | Inference ms |
|---|---|---:|---:|---:|---:|---:|
| CH | CrowdHuman | 0.7599 | 0.4824 | 0.8037 | 0.6615 | 1.9946 |
| CH | WiderPerson | 0.6590 | 0.2739 | 0.7571 | 0.5942 | 1.1337 |
| CH | COCO2017 | 0.6460 | 0.4144 | 0.6856 | 0.5837 | 1.0383 |
| CH | OCHuman | 0.7939 | 0.5327 | 0.7818 | 0.7457 | 1.1348 |
| CH + WP | CrowdHuman | 0.7553 | 0.4721 | 0.8017 | 0.6543 | 1.6811 |
| CH + WP | WiderPerson | 0.7503 | 0.4676 | 0.8124 | 0.6386 | 2.3089 |
| CH + WP | COCO2017 | 0.6389 | 0.4050 | 0.6695 | 0.5955 | 1.0884 |
| CH + WP | OCHuman | 0.7815 | 0.5007 | 0.7691 | 0.7359 | 1.0902 |
| CH + WP + COCO | CrowdHuman | 0.7289 | 0.4594 | 0.7919 | 0.6183 | 1.5948 |
| CH + WP + COCO | WiderPerson | 0.7368 | 0.4597 | 0.8151 | 0.6217 | 2.1126 |
| CH + WP + COCO | COCO2017 | 0.7736 | 0.5388 | 0.8122 | 0.6712 | 1.3679 |
| CH + WP + COCO | OCHuman | 0.7764 | 0.5040 | 0.7948 | 0.7179 | 1.1383 |
| CH + WP + COCO + OCH | CrowdHuman | 0.7282 | 0.4592 | 0.7886 | 0.6182 | 1.7854 |
| CH + WP + COCO + OCH | WiderPerson | 0.7354 | 0.4596 | 0.8135 | 0.6184 | 1.1954 |
| CH + WP + COCO + OCH | COCO2017 | 0.7740 | 0.5400 | 0.8051 | 0.6720 | 1.1147 |
| CH + WP + COCO + OCH | OCHuman | 0.8743 | 0.7004 | 0.8615 | 0.7716 | 1.1440 |

### 8.2 Macro-average across validation datasets

| Train set | Avg mAP50 | Avg mAP50-95 | Avg Precision | Avg Recall |
|---|---:|---:|---:|---:|
| CH | 0.7147 | 0.4258 | 0.7570 | 0.6463 |
| CH + WP | 0.7315 | 0.4614 | 0.7632 | 0.6561 |
| CH + WP + COCO | 0.7539 | 0.4905 | 0.8035 | 0.6573 |
| CH + WP + COCO + OCH | 0.7780 | 0.5398 | 0.8172 | 0.6700 |

### 8.3 Main interpretation

The results show a clear and expected domain-specialization pattern.

Adding WiderPerson strongly improves WiderPerson validation performance:

```text
CH only on WP:        mAP50-95 = 0.2739
CH + WP on WP:        mAP50-95 = 0.4676
```

Adding COCO strongly improves COCO validation performance:

```text
CH + WP on COCO:      mAP50-95 = 0.4050
CH + WP + COCO:       mAP50-95 = 0.5388
```

Adding OCHuman strongly improves OCHuman validation performance:

```text
CH + WP + COCO on OCH:      mAP50-95 = 0.5040
CH + WP + COCO + OCH on OCH: mAP50-95 = 0.7004
```

However, adding more datasets slightly reduces CrowdHuman performance:

```text
CH only on CH:              mAP50-95 = 0.4824
CH + WP + COCO + OCH on CH: mAP50-95 = 0.4592
```

This is not surprising. The datasets have different image distributions and probably slightly different bounding-box annotation styles. The model learns a compromise across datasets rather than optimizing perfectly for CrowdHuman visible boxes.

### 8.4 Is this overfitting?

The pattern is not best described as simple overfitting. It is more likely a combination of:

- dataset domain shift;
- annotation-style differences;
- dataset imbalance;
- optimization interference between domains;
- different object scales and occlusion types;
- possible conversion/labeling noise.

Overfitting could play a role, especially for smaller datasets such as OCHuman, but the more important observation is **multi-domain tradeoff**: each added dataset improves its own domain, but the optimal detector behavior is not identical across all datasets.

---

## 9. Current conclusions

### 9.1 Scientific conclusion

The experiments support the idea that dataset composition has a major effect on occluded/visible-person detection. Training only on CrowdHuman gives the best CrowdHuman-specific performance, while adding WiderPerson, COCO, and OCHuman improves cross-dataset robustness and macro-average performance. This shows that the project is not just “fine-tune YOLO once”; it investigates how training data affects generalization for a specific target task.

### 9.2 Practical app conclusion

For the actual game/app, the best current candidate is likely:

```text
yolo26n or yolo26s trained on CH + WP + COCO + OCH
```

provided it also performs well on a custom phone-shot test set. Including OCHuman makes sense for production because heavily occluded people are part of the target use case.

For a cleaner benchmark claim, the report should distinguish:

```text
Generalization experiment:
  train without OCHuman, test on OCHuman.

Production-oriented model:
  train with OCHuman included, then test on custom app-like images.
```

---

## 10. Comparison with the original plan

The final detection work stayed aligned with the original project direction, but the emphasis shifted. The original plan defined the detector as **occlusion-robust visible-person detection** for a mobile hit-localization pipeline, with YOLO26 as the practical starting point, CrowdHuman visible boxes as the primary dataset, and merged datasets as an optional extension. This core direction was preserved.

### 10.1 Parts that stayed close to the original plan

The following original decisions were implemented directly:

- the task was treated as visible-person detection, not generic COCO object detection;
- CrowdHuman visible boxes were used as the primary fine-tuning target;
- YOLO26 was used because it offers practical fine-tuning, speed/accuracy scaling, and mobile/export relevance;
- the detector was trained as a one-class `person` model rather than a `person`/`not-person` classifier;
- `yolo26n` and `yolo26s` were prioritized because they are the most realistic mobile/near-mobile candidates;
- 640 and 960 input sizes were tested;
- merged-dataset training was implemented through YAML files rather than physically copying data;
- models were evaluated separately on each dataset instead of only on a merged validation set.

### 10.2 Parts that expanded beyond the original minimum plan

The original plan suggested merged-dataset fine-tuning as optional. In the completed work, this became one of the main experimental axes. The final setup systematically tested:

```text
CrowdHuman
CrowdHuman + WiderPerson
CrowdHuman + WiderPerson + COCO2017 person
CrowdHuman + WiderPerson + COCO2017 person + OCHuman
```

This was a useful expansion because the results showed clear dataset-domain effects: adding a dataset strongly improved performance on that dataset, while sometimes slightly lowering performance on previously dominant datasets. This gave the project a stronger experimental question: **how does training-data composition affect occluded visible-person detection?**

The completed work also expanded the evaluation side by including pretrained baselines and a high-capacity YOLO26x reference model. This was not central to the original minimum scope, but it helps contextualize how much fine-tuning helps compared with simply using a larger pretrained detector.

### 10.3 Parts not completed or intentionally deprioritized

Some planned or optional parts were not completed in the detection subproject:

| Planned item | Status | Reason / interpretation |
|---|---|---|
| CrowdHuman visible-box vs full-body-box ablation | Not completed | Useful idea, but dataset-composition experiments became the main focus. |
| Custom phone-shot test set | Not yet completed | Still the most important next step for proving app relevance. |
| Mobile export / quantized model validation | Not yet completed | Requires final model selection first. |
| Extensive hyperparameter tuning | Intentionally deprioritized | Too time-consuming for a ~50-hour project; threshold tuning is more realistic. |
| Transformer-detector training such as RF-DETR / Grounding DINO | Not implemented | Considered, but YOLO26 was more practical and mobile-relevant. |
| Larger fine-tuned YOLO26m/l/x models | Mostly not pursued | Less relevant for mobile; YOLO26x was used only as a pretrained reference. |

### 10.4 Overall deviation assessment

The project did **not** deviate away from the original goal. It remained focused on visible, occluded person detection for the downstream hit-localization pipeline. The main deviation was that the implementation became more systematic on dataset composition and cross-dataset evaluation, while some originally suggested ablations such as `vbox` vs `fbox` and custom phone-shot evaluation remain future work.

A concise report statement would be:

> The final detection work preserved the original task definition and model rationale, but shifted effort toward a more systematic dataset-composition study. This was a justified change because the major practical uncertainty was not whether YOLO can detect people, but which training data mixture best supports visible and occluded person detection across domains.
---

## 11. Recommended next steps

### 10.1 Create a custom app-like test set

This is the most important next step. Public validation sets are useful, but they do not fully match the final app. A small custom test set should include:

- phone-camera images;
- people near the center/crosshair;
- partial body visibility;
- occlusion by furniture/walls/other people;
- different distances and lighting;
- images without people;
- people at image edges;
- cases where only a hand/leg/torso is visible.

Even 100–300 manually labeled images would make the evaluation much more convincing.

### 10.2 Compare 640 vs 960

The next useful experiment is not necessarily a larger model. It is evaluating whether higher input resolution improves partial-person recall:

```text
yolo26n 640 vs yolo26n 960
yolo26s 640 vs yolo26s 960
```

The 960 models cost more compute but may detect small/partial people better.

### 10.3 Tune confidence threshold

Full hyperparameter tuning is probably too time-consuming for a 50-hour project. However, confidence-threshold tuning is cheap and directly relevant to the app.

Test thresholds such as:

```text
0.10, 0.15, 0.20, 0.25, 0.30, 0.40
```

Because false negatives are costly, the app may prefer a lower threshold with higher recall, allowing segmentation/ReID to reject false positives later.

### 10.4 Error analysis

For the final report, include qualitative failure categories:

- missed tiny/partial person;
- loose box;
- duplicate detections;
- false positive on background object;
- person visible but not centered;
- severe occlusion missed;
- incorrect localization around mask-derived boxes.

This makes the evaluation more convincing than metrics alone.

### 10.5 Mobile export

Export the best `n` model to a mobile-friendly format and verify parity:

- ONNX as a first deployment-neutral format;
- TFLite / CoreML / NCNN depending on target platform;
- FP16 or INT8 quantization if supported.

Export should be validated by comparing predictions/metrics between `.pt` and exported model.

---

## 12. Report narrative

A good report framing is:

> We implemented a person-detection subsystem for a mobile camera-based game where the detector must find visible people, including partially occluded persons. We chose YOLO26 because it provides a strong accuracy/speed tradeoff, supports small mobile-friendly models, and has mature training/export tooling. We converted multiple public person datasets into a unified one-class YOLO format and systematically studied how dataset composition affects cross-dataset detection performance. The results show that each dataset contributes useful domain-specific information, while also introducing mild cross-domain tradeoffs likely caused by annotation and distribution differences. The best public-validation macro-average was obtained by training on all datasets, but the final model should be selected using a custom app-like test set.

This demonstrates:

- clear problem definition;
- justified model choice;
- careful dataset selection;
- nontrivial data conversion;
- HPC engineering and reproducible scripts;
- systematic ablation experiments;
- thoughtful interpretation of results;
- realistic next steps for deployment.

---

## 13. Limitations

Current limitations:

- Public datasets do not perfectly match the final app distribution.
- Bounding-box semantics differ slightly across datasets.
- Dataset balancing was not explicitly controlled.
- The numeric result table currently documents the most complete `yolo26n @640` dataset-composition ablation; the additional 960, pretrained-baseline, YOLO26s, and YOLO26x reference evaluations should be summarized in the final report once their CSV/Markdown outputs are consolidated.
- OCHuman-as-training improves OCHuman validation, but then OCHuman is not a clean held-out benchmark.
- Inference times measured on HPC GPU do not directly equal mobile latency.
- Hyperparameters were mostly kept fixed; no extensive hyperparameter search was performed.

These are acceptable limitations for a ~50-hour course project, especially because the project already includes systematic dataset experiments and realistic engineering work.

---

## 14. References

- Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26
- Ultralytics YOLO dataset YAML documentation: https://academy.ultralytics.com/courses/train-your-first-yolo/write-the-data-yaml
- Ultralytics YOLO dataset preparation documentation: https://academy.ultralytics.com/courses/train-your-first-yolo/prepare-a-dataset
- Ultralytics object detection metrics documentation: https://docs.ultralytics.com/guides/yolo-performance-metrics
- Ultralytics export documentation: https://docs.ultralytics.com/modes/export
- CrowdHuman dataset: https://www.crowdhuman.org/
- Grounding DINO paper page: https://huggingface.co/papers/2303.05499
- RF-DETR paper page: https://huggingface.co/papers/2511.09554
- PyTorch previous versions / CUDA 11.8 wheels: https://pytorch.org/get-started/previous-versions/
- NVIDIA Tesla V100 product information: https://www.nvidia.com/en-gb/data-center/tesla-v100/
