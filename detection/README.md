# Detection Subproject

This subproject covers the person-detection part of the larger deep learning project: detecting visible, partially occluded people from a captured frame for a downstream hit-localization pipeline.

The detector target is the **visible person region**, not a hallucinated full-body box behind occluders. The main model family is YOLO26, with experiments centered on `yolo26n` and `yolo26s`.

## Key documents

- `person_detection_plan.md`: initial project plan and experiment rationale.
- `person_detection_project_documentation.md`: implementation notes, completed experiments, consolidated results, and remaining work.

## Directory structure

- `data/`: source papers for the datasets used in this subproject.
- `datasets/`: YOLO-formatted datasets and merged dataset YAML files.
- `scripts/`: dataset conversion, inspection, training, and evaluation scripts.
- `sbatch/`: Slurm job scripts used on the HPC cluster.
- `outputs/`: training runs, evaluation runs, summaries, logs, and archived artifacts.
- `testing.ipynb`: local notebook used for ad hoc inspection and experimentation.

## Outputs layout

- `outputs/training/<run_name>/`: one folder per completed training run.
  Contains Ultralytics artifacts, checkpoints, and the corresponding `slurm_train.out` log.
- `outputs/evaluation/<run_name>/`: one folder per cross-dataset evaluation run.
  Contains:
  - `summary.csv`, `summary.json`, `summary.md`
  - `slurm_eval.out`
  - one subfolder per evaluation dataset: `crowdhuman/`, `widerperson/`, `coco2017/`, `ochuman/`
- `outputs/archive/legacy_evaluations/`: older single-dataset evaluation outputs kept for reference.
- `outputs/archive/orphaned_eval_logs/`: older logs whose matching summary files were missing or overwritten.
- `outputs/archive/incomplete_training/`: partial or abandoned runs.

## Naming conventions

- `stock`: untouched official Ultralytics checkpoint, evaluated without fine-tuning.
- `chv`: CrowdHuman visible-box fine-tuning setup.
- `wp`: WiderPerson.
- `coco`: COCO2017 person subset.
- `och`: OCHuman.

Examples:

- `yolo26n_stock_640`: untouched YOLO26n checkpoint evaluated at `imgsz=640`
- `yolo26n_chv_640`: YOLO26n fine-tuned on CrowdHuman visible boxes at `imgsz=640`
- `yolo26n_chv_wp_coco_och_960`: YOLO26n fine-tuned on the merged CrowdHuman-visible-box + WiderPerson + COCO + OCHuman setup at `imgsz=960`

## Current experimental focus

The current folder structure reflects three main experiment groups:

- `stock` references: untouched YOLO26n / YOLO26s / YOLO26x checkpoints evaluated on all four datasets.
- `chv` fine-tunes: CrowdHuman visible-box fine-tunes used as the main occlusion-focused baseline.
- merged-data fine-tunes: `chv_wp`, `chv_wp_coco`, and `chv_wp_coco_och` runs for dataset-composition analysis.

The best macro-average cross-dataset result currently comes from `yolo26n_chv_wp_coco_och_960`, while `yolo26s_chv_960` is the strongest CrowdHuman-specific checkpoint. The main remaining work is custom app-like evaluation and final model selection for deployment.
