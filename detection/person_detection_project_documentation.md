# Person Detection Subproject Documentation

**Project context:** deep-learning-based mobile / near-mobile person detection for a larger camera-based game pipeline.

**Detection role in the full system:** given a single frame captured when the player presses "shoot", detect visible people in the image, pass the relevant person / candidate region to body-part segmentation, and later support person re-identification. The detector should work even when only parts of the person are visible, because the downstream app needs to determine whether a visible body part was hit.

---

## 1. High-level pipeline and problem definition

The planned full pipeline is:

```text
Input image from phone camera
  -> person detection
  -> select candidate(s), especially around image center / crosshair
  -> body-part segmentation of selected person
  -> ReID using visible body regions
  -> game decision: was the correct visible body part / person hit?
```

The detection subproblem is therefore not generic "detect all COCO objects". It is specifically:

```text
Detect visible person regions robustly, including partial and occluded people, with a model small enough to be a plausible mobile candidate.
```

The key design decision is that the detector should predict the **visible person region**, not hallucinate a full-body box behind occluders. If only a leg is visible behind a wall, a full-body box would create physically wrong game behavior by allowing "hits through walls". Therefore, visible-region detection is the correct target for this project.

Another important design decision is that the detector does **not** need to run continuously on every camera frame. The app can process a single captured frame after the shoot action, which makes somewhat heavier models and higher resolutions more realistic while still keeping mobile deployment in mind.

---

## 2. Why YOLO26 was used

The detector needed to satisfy several constraints:

- strong person-detection performance
- robustness to occlusion and partial visibility
- reasonable inference speed
- exportability to mobile / edge formats
- easy fine-tuning on YOLO-format datasets
- manageable implementation effort for a course project

We considered transformer-style alternatives such as Grounding DINO and RF-DETR, but Ultralytics YOLO26 was the most practical choice because it already provides:

- pretrained checkpoints in several model sizes
- straightforward fine-tuning and validation workflows
- export tooling for ONNX, CoreML, TFLite, TensorRT, and OpenVINO
- a clear speed / accuracy scale from `n` to `x`

This made `yolo26n` and `yolo26s` the most relevant models for the project. Larger models such as `x` are useful as upper-bound references, but much less realistic for final deployment.

The model is trained as a **single-class detector**:

```yaml
names:
  0: person
```

We explicitly did **not** add a `not-person` class. Background is already learned implicitly in detection, and adding a `not-person` class would require arbitrary negative annotations that are not well defined.

---

## 3. Dataset selection

The datasets were selected to cover visible-person detection, crowds, pedestrians, general person diversity, and hard occlusion.

### 3.1 CrowdHuman

CrowdHuman is the most important dataset in the project because it provides visible-region boxes. We use the **visible bounding box (`vbox`)** annotations rather than full-body boxes.

Role in project:

```text
Main occlusion / crowd training dataset and main fine-tuning baseline.
```

### 3.2 WiderPerson

WiderPerson adds pedestrian and partial-visibility diversity beyond CrowdHuman.

Role in project:

```text
Second training dataset for pedestrian / partial-person diversity.
```

### 3.3 COCO2017 person subset

COCO adds broad general person appearance diversity, though it is less specialized for occlusion than CrowdHuman.

Role in project:

```text
General person-detection diversity dataset.
```

### 3.4 OCHuman

OCHuman is a hard occlusion dataset. It can be used either as a difficult evaluation benchmark or as production-oriented training data when heavy occlusion is central to the target use case.

Role in project:

```text
Hard occlusion evaluation set and optional production-oriented training data.
```

---

## 4. Data preparation and label compatibility

All datasets were converted to a unified YOLO detection format:

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

Each label file contains normalized rows of the form:

```text
class_id x_center y_center width height
```

All datasets use a single class:

```yaml
names:
  0: person
```

The datasets are class-compatible. The main remaining issue is **box semantic mismatch**:

- CrowdHuman uses visible-region boxes
- WiderPerson has pedestrian / visible-person-like boxes
- COCO uses general person boxes
- OCHuman can be especially tight around visible regions depending on the conversion

This matters because mixed training can improve robustness but also create localization-style compromises between datasets.

Visual inspection scripts were used to confirm that:

- boxes are aligned with people
- labels are normalized correctly
- class IDs are correct
- ignore / crowd regions are not accidentally treated as normal positives

---

## 5. Training setup

### 5.1 Fine-tuning approach

The models were initialized from pretrained YOLO26 checkpoints (`yolo26n.pt`, `yolo26s.pt`, `yolo26x.pt`) and then fine-tuned on the one-class person datasets.

Unless layers are explicitly frozen, Ultralytics fine-tuning updates trainable weights across the model, not just a final classification layer. This is useful because the detector must adapt not only to the `person` class but also to the visible-region annotation style.

### 5.2 Input size

Experiments were run at `imgsz=640` and `imgsz=960`.

Reasoning:

- `640` is cheaper and closer to common YOLO defaults
- `960` may help with small or partially visible people
- `960` costs about 2.25x the image-area compute of `640`

### 5.3 Model sizes

Main focus:

```text
yolo26n: mobile candidate
yolo26s: stronger but still plausible
```

`yolo26x` was used only as a stock upper-bound reference. A fine-tuning attempt for `x` was started but not completed.

### 5.4 Merged dataset training

Rather than physically copying all data into one folder, merged YAML files were created using lists of image folders.

The main merged recipes were:

```text
CHV
CHV + WP
CHV + WP + COCO
CHV + WP + COCO + OCH
```

where `CHV` means CrowdHuman visible boxes.

---

## 6. HPC and engineering issues solved

This subproject included a substantial amount of engineering work on the HPC cluster, not just notebook experimentation.

### 6.1 Conda and Python package isolation

Packages were initially leaking in from:

```text
~/.local/lib/python3.11/site-packages
```

The fix was:

```bash
export PYTHONNOUSERSITE=1
```

and installing required packages into the intended conda environment.

### 6.2 V100 compatibility issue

The HPC nodes used Tesla V100S GPUs. A newer PyTorch / CUDA wheel did not include compatible kernels for `sm_70`, causing:

```text
CUDA error: no kernel image is available for execution on the device
```

The fix was to build a V100-compatible environment with CUDA 11.8 PyTorch wheels and test it inside a real Slurm GPU allocation before launching training.

### 6.3 Slurm scripts

Reusable Slurm scripts were created for:

- single training jobs
- merged-dataset training jobs
- evaluating one model on one dataset
- evaluating one model on all datasets

### 6.4 Dataset cache race issue

One job failed due to an Ultralytics dataset cache problem:

```text
AssertionError during cache hash check
FileNotFoundError: labels/val.cache
```

This was traced to stale or concurrently modified cache files. The practical fix was to remove the cache and rerun the failed job.

### 6.5 Run directory management

Ultralytics creates new directories when a run name already exists, for example:

```text
yolo26s_chv_960
yolo26s_chv_960-2
```

The completed run must be identified by checking for:

```text
weights/best.pt
weights/last.pt
results.csv
```

For local organization, the output folders were normalized into a shorter naming scheme:

```text
stock = untouched official Ultralytics checkpoint
chv   = CrowdHuman visible-box fine-tune
wp    = WiderPerson
coco  = COCO2017 person subset
och   = OCHuman
```

Examples:

```text
outputs/training/yolo26n_chv_640
outputs/training/yolo26n_chv_wp_coco_och_960
outputs/evaluation/yolo26n_stock_640
outputs/evaluation/yolo26s_chv_960
```

The `.pt` files in the project root are **stock checkpoints** from Ultralytics, not fine-tuned results.

---

## 7. Evaluation methodology

### 7.1 Why separate validation sets are better than merged validation only

Evaluating on one merged validation set gives one average number, but hides where the model succeeds or fails. Therefore, each checkpoint was evaluated separately on:

```text
CrowdHuman validation
WiderPerson validation
COCO2017/person validation
OCHuman validation
```

### 7.2 Metrics used

| Metric | Interpretation |
|---|---|
| Precision | Of predicted detections, how many are correct? |
| Recall | Of true persons, how many were detected? |
| mAP50 | AP at IoU 0.50; lenient localization metric |
| mAP50-95 | Average AP from IoU 0.50 to 0.95; stricter localization metric |
| Inference time | Approximate model speed in the validation environment |

For the app, recall is especially important. If the detector misses a partially visible person, downstream segmentation / ReID cannot recover that target.

Important disclaimer on timing:

```text
The inference-time numbers in this document are not perfectly comparable across all checkpoints.
Most evaluations were run on a Tesla V100S-PCIE-32GB, but yolo26s_stock_640 was evaluated on an NVIDIA H100 PCIe.
Therefore, the timing columns are useful as rough reference numbers, but not as a perfectly controlled hardware-normalized speed benchmark.
```

### 7.3 Evaluation outputs

The evaluator produces:

- `summary.csv`
- `summary.json`
- `summary.md`
- per-dataset Ultralytics result folders with plots and example images

### 7.4 Completed experiment inventory

Completed fine-tuning runs:

| Run | Model | Input size | Training data | Purpose |
|---|---|---:|---|---|
| `yolo26n_chv_640` | YOLO26n | 640 | CrowdHuman visible boxes | main fine-tuned baseline |
| `yolo26n_chv_wp_640` | YOLO26n | 640 | CHV + WP | test added pedestrian / partial-person diversity |
| `yolo26n_chv_wp_coco_640` | YOLO26n | 640 | CHV + WP + COCO | test added general person diversity |
| `yolo26n_chv_wp_coco_och_640` | YOLO26n | 640 | CHV + WP + COCO + OCH | all-data mobile candidate |
| `yolo26n_chv_960` | YOLO26n | 960 | CrowdHuman visible boxes | input-size comparison |
| `yolo26n_chv_wp_coco_och_960` | YOLO26n | 960 | CHV + WP + COCO + OCH | higher-resolution all-data candidate |
| `yolo26s_chv_960` | YOLO26s | 960 | CrowdHuman visible boxes | stronger fine-tuned reference |

Completed cross-dataset evaluation runs:

| Run | Type | Input size | Fine-tuned? | Eval GPU | Purpose |
|---|---|---:|---|---|---|
| `yolo26n_stock_640` | YOLO26n stock checkpoint | 640 | no | Tesla V100S-PCIE-32GB | smallest stock reference |
| `yolo26n_stock_960` | YOLO26n stock checkpoint | 960 | no | Tesla V100S-PCIE-32GB | stock reference at higher input size |
| `yolo26n_chv_640` | YOLO26n fine-tune | 640 | yes | Tesla V100S-PCIE-32GB | `n` fine-tuned baseline |
| `yolo26n_chv_960` | YOLO26n fine-tune | 960 | yes | Tesla V100S-PCIE-32GB | `chv` input-size comparison |
| `yolo26n_chv_wp_640` | YOLO26n fine-tune | 640 | yes | Tesla V100S-PCIE-32GB | merged-data ablation |
| `yolo26n_chv_wp_coco_640` | YOLO26n fine-tune | 640 | yes | Tesla V100S-PCIE-32GB | merged-data ablation |
| `yolo26n_chv_wp_coco_och_640` | YOLO26n fine-tune | 640 | yes | Tesla V100S-PCIE-32GB | all-data mobile candidate |
| `yolo26n_chv_wp_coco_och_960` | YOLO26n fine-tune | 960 | yes | Tesla V100S-PCIE-32GB | higher-resolution all-data mobile candidate |
| `yolo26s_stock_640` | YOLO26s stock checkpoint | 640 | no | NVIDIA H100 PCIe | stronger stock reference |
| `yolo26s_stock_960` | YOLO26s stock checkpoint | 960 | no | Tesla V100S-PCIE-32GB | stronger stock reference at higher input size |
| `yolo26s_chv_960` | YOLO26s fine-tune | 960 | yes | Tesla V100S-PCIE-32GB | stronger fine-tuned reference |
| `yolo26x_stock_960` | YOLO26x stock checkpoint | 960 | no | Tesla V100S-PCIE-32GB | high-capacity upper-bound reference |

Older single-dataset evaluations and the incomplete `yolo26x` fine-tuning attempt were kept in `outputs/archive/` but are not part of the main final result matrix.

---

## 8. Current experimental results

### 8.1 Macro-average across all four validation datasets

The following table reports macro-averages across CrowdHuman, WiderPerson, COCO2017 person, and OCHuman validation sets.

| Run | Avg mAP50 | Avg mAP50-95 | Avg Precision | Avg Recall | Avg Inference ms |
|---|---:|---:|---:|---:|---:|
| `yolo26n_stock_640` | 0.6515 | 0.3793 | 0.7555 | 0.5704 | 1.2404 |
| `yolo26n_stock_960` | 0.6700 | 0.3737 | 0.7467 | 0.6031 | 2.1623 |
| `yolo26n_chv_640` | 0.7147 | 0.4259 | 0.7571 | 0.6462 | 1.3253 |
| `yolo26n_chv_960` | 0.7310 | 0.4444 | 0.7494 | 0.6756 | 2.4477 |
| `yolo26n_chv_wp_640` | 0.7315 | 0.4614 | 0.7632 | 0.6561 | 1.5421 |
| `yolo26n_chv_wp_coco_640` | 0.7539 | 0.4905 | 0.8035 | 0.6573 | 1.5534 |
| `yolo26n_chv_wp_coco_och_640` | 0.7780 | 0.5398 | 0.8172 | 0.6700 | 1.3099 |
| `yolo26n_chv_wp_coco_och_960` | 0.8088 | 0.5724 | 0.8293 | 0.6991 | 2.2645 |
| `yolo26s_stock_640` | 0.7105 | 0.4324 | 0.7753 | 0.6319 | 0.9923 |
| `yolo26s_stock_960` | 0.7162 | 0.4166 | 0.7584 | 0.6486 | 4.7255 |
| `yolo26s_chv_960` | 0.7523 | 0.4722 | 0.7508 | 0.7051 | 4.5752 |
| `yolo26x_stock_960` | 0.7466 | 0.4279 | 0.7642 | 0.6867 | 29.0182 |

The best overall macro-average comes from `yolo26n_chv_wp_coco_och_960`.

Timing note: the macro-average inference times above mix V100 and H100 evaluation runs. In the active 12-run matrix, `yolo26s_stock_640` is the only run evaluated on H100; the others were evaluated on V100.

### 8.2 Detailed `yolo26n @640` dataset-composition ablation

| Train set | Eval set | mAP50 | mAP50-95 | Precision | Recall | Inference ms |
|---|---|---:|---:|---:|---:|---:|
| CHV | CrowdHuman | 0.7599 | 0.4824 | 0.8037 | 0.6615 | 1.9946 |
| CHV | WiderPerson | 0.6590 | 0.2739 | 0.7571 | 0.5942 | 1.1337 |
| CHV | COCO2017 | 0.6460 | 0.4144 | 0.6856 | 0.5837 | 1.0383 |
| CHV | OCHuman | 0.7939 | 0.5327 | 0.7818 | 0.7457 | 1.1348 |
| CHV + WP | CrowdHuman | 0.7553 | 0.4721 | 0.8017 | 0.6543 | 1.6811 |
| CHV + WP | WiderPerson | 0.7503 | 0.4676 | 0.8124 | 0.6386 | 2.3089 |
| CHV + WP | COCO2017 | 0.6389 | 0.4050 | 0.6695 | 0.5955 | 1.0884 |
| CHV + WP | OCHuman | 0.7815 | 0.5007 | 0.7691 | 0.7359 | 1.0902 |
| CHV + WP + COCO | CrowdHuman | 0.7289 | 0.4594 | 0.7919 | 0.6183 | 1.5948 |
| CHV + WP + COCO | WiderPerson | 0.7368 | 0.4597 | 0.8151 | 0.6217 | 2.1126 |
| CHV + WP + COCO | COCO2017 | 0.7736 | 0.5388 | 0.8122 | 0.6712 | 1.3679 |
| CHV + WP + COCO | OCHuman | 0.7764 | 0.5040 | 0.7948 | 0.7179 | 1.1383 |
| CHV + WP + COCO + OCH | CrowdHuman | 0.7282 | 0.4592 | 0.7886 | 0.6182 | 1.7854 |
| CHV + WP + COCO + OCH | WiderPerson | 0.7354 | 0.4596 | 0.8135 | 0.6184 | 1.1954 |
| CHV + WP + COCO + OCH | COCO2017 | 0.7740 | 0.5400 | 0.8051 | 0.6720 | 1.1147 |
| CHV + WP + COCO + OCH | OCHuman | 0.8743 | 0.7004 | 0.8615 | 0.7716 | 1.1440 |

Macro-average for that ablation:

| Train set | Avg mAP50 | Avg mAP50-95 | Avg Precision | Avg Recall |
|---|---:|---:|---:|---:|
| CHV | 0.7147 | 0.4259 | 0.7571 | 0.6462 |
| CHV + WP | 0.7315 | 0.4614 | 0.7632 | 0.6561 |
| CHV + WP + COCO | 0.7539 | 0.4905 | 0.8035 | 0.6573 |
| CHV + WP + COCO + OCH | 0.7780 | 0.5398 | 0.8172 | 0.6700 |


### 8.3 Strongest checkpoints by evaluation domain

| Eval set | Best checkpoint | mAP50-95 | Notes |
|---|---|---:|---|
| CrowdHuman | `yolo26s_chv_960` | 0.5985 | best CH-specific result |
| WiderPerson | `yolo26n_chv_wp_coco_och_960` | 0.4888 | best merged-data result on WP |
| COCO2017 | `yolo26x_stock_960` | 0.6281 | strongest overall reference, but not mobile-relevant |
| OCHuman | `yolo26n_chv_wp_coco_och_640` | 0.7004 | best hard-occlusion result |

### 8.4 Interpretation

The results show a clear multi-domain tradeoff:

- CrowdHuman-only training gives the cleanest optimization for the CrowdHuman visible-box domain.
- Adding WiderPerson, COCO, and OCHuman improves the domains they represent, especially at the stricter mAP50-95 metric.
- The best balanced mobile-oriented checkpoint is `yolo26n_chv_wp_coco_och_960`.
- `yolo26s_chv_960` is the strongest CrowdHuman-specific model.
- `yolo26x_stock_960` is useful as an upper-bound reference, especially on COCO, but not as a realistic final deployment choice.

This pattern is better explained by domain shift, annotation-style mismatch, and dataset balancing tradeoffs than by simple overfitting alone.

---

## 9. Current conclusions

### 9.1 Scientific conclusion

The experiments support the idea that dataset composition has a major effect on occluded visible-person detection. Fine-tuning improves results over stock checkpoints, and progressively adding WiderPerson, COCO, and OCHuman improves cross-dataset robustness and macro-average performance. The project therefore goes beyond "fine-tune YOLO once" and becomes a study of how training-data composition, input size, and checkpoint choice affect generalization for the target task.

### 9.2 Practical app conclusion

For the actual app, the best current candidate is:

```text
yolo26n_chv_wp_coco_och_960
```

because it has the best macro-average across the four public validation domains while staying within a realistic mobile-oriented model family.

A second strong candidate is:

```text
yolo26s_chv_960
```

which is the strongest CrowdHuman-specific checkpoint and could still be acceptable if the final app can tolerate the heavier inference cost.

Both conclusions remain conditional on a custom phone-shot evaluation set. Public datasets are useful, but they are not the final judge of app relevance.

---

## 10. Comparison with the original plan

The final detection work stayed aligned with the original project direction. The detector remained focused on visible-person detection, CrowdHuman visible boxes remained the central fine-tuning target, and YOLO26 remained the practical model family.

The main shift was emphasis:

- the project expanded strongly into a systematic dataset-composition study
- cross-dataset evaluation became much more central than originally planned
- stock `n`, `s`, and `x` references were added to contextualize what fine-tuning actually contributes

The main planned items that are still incomplete are:

- visible-box vs full-body-box ablation
- custom phone-shot evaluation
- final mobile export / validation

---

## 11. Recommended next steps

### 11.1 Create a custom app-like test set

This is the most important next step. A useful custom test set should include:

- phone-camera images
- people near the center / crosshair
- partial body visibility
- occlusion by furniture, walls, and other people
- different distances and lighting
- negative images with no people
- edge-of-frame cases
- extreme partial-visibility cases such as only a hand, leg, or torso

Even 100-300 manually labeled images would make the final conclusions much more convincing.

### 11.2 Validate the two leading public-set candidates

The most important head-to-head comparison for the next stage is:

```text
yolo26n_chv_wp_coco_och_960
vs
yolo26s_chv_960
```

The first is the best balanced mobile-oriented checkpoint by macro-average. The second is the strongest CrowdHuman-specific checkpoint.

### 11.3 Tune confidence threshold

Full hyperparameter search is probably not worth the time budget, but confidence-threshold tuning is cheap and directly relevant to the app.

Useful thresholds to test:

```text
0.10, 0.15, 0.20, 0.25, 0.30, 0.40
```

Because false negatives are costly, a lower threshold with higher recall may be preferable if downstream segmentation / ReID can reject false positives.

### 11.4 Error analysis

For the final report, include qualitative failure categories such as:

- missed tiny / partial person
- loose box
- duplicate detections
- false positive on background object
- person visible but not centered
- severe occlusion missed
- inconsistent localization around mask-derived boxes

### 11.5 Mobile export

Export the best `n` checkpoint to deployment-friendly formats and validate parity:

- ONNX
- TFLite / CoreML / NCNN depending on target
- FP16 or INT8 if supported

The exported model must be compared against the `.pt` checkpoint to confirm that deployment optimization does not damage accuracy too much.

---

## 12. Report narrative

A good report framing is:

> We implemented a person-detection subsystem for a mobile camera-based game where the detector must find visible people, including partially occluded persons. We chose YOLO26 because it provides a strong accuracy / speed tradeoff, supports small mobile-friendly models, and has mature training and export tooling. We converted multiple public person datasets into a unified one-class YOLO format and systematically studied how dataset composition, input size, and checkpoint choice affect cross-dataset detection performance. The best macro-average public-set result came from a YOLO26n model fine-tuned on CrowdHuman visible boxes plus WiderPerson, COCO, and OCHuman at 960 resolution, while a YOLO26s CrowdHuman-visible-box fine-tune produced the strongest CrowdHuman-specific result. Final model selection should still be made using a custom app-like test set.

This narrative highlights:

- clear problem definition
- justified model choice
- nontrivial data conversion
- meaningful HPC engineering work
- systematic ablations
- realistic deployment thinking

---

## 13. Limitations

Current limitations:

- Public datasets still do not perfectly match the final app distribution.
- Bounding-box semantics differ slightly across datasets.
- Dataset balancing was not explicitly controlled.
- The best public-set checkpoint (`yolo26n_chv_wp_coco_och_960`) is still only a proxy for real app performance until it is tested on custom phone-shot data.
- OCHuman-as-training improves OCHuman validation, but then OCHuman is no longer a clean held-out benchmark.
- Inference times measured on the HPC GPU do not directly equal mobile latency.
- The reported inference times are also not fully hardware-controlled across checkpoints, because most evaluations were run on V100 while `yolo26s_stock_640` was run on H100.
- Hyperparameters were mostly kept fixed; there was no extensive search.
- `yolo26x_stock_960` is useful as an upper-bound reference, but not a realistic final deployment candidate.
- The attempted `yolo26x` fine-tuning run was not completed.

These are acceptable limitations for a roughly 50-hour course project, especially because the subproject already includes systematic experiments and substantial engineering work.

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
