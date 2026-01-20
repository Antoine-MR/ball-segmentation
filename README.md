
# Ball Segmentation Project

This project provides an end-to-end pipeline for object detection and segmentation (balls, persons, trashcans) using YOLO and SAM models. It includes data preparation, automated labeling, training, and inference tools, with modular code and reproducible workflows.

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation
```bash
git clone <repo_url>
cd <repo_folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pretrained Models
Download and place the following in `models/pretrained/`:
- `yolo11n-seg.pt` (YOLO segmentation)
- `yolo11n.pt` (YOLO detection)
- `sam2.1_b.pt` (SAM 2.1)
- `FastSAM-s.pt` (FastSAM)

## Notebooks Overview

**Main Notebooks:**
- `add_to_fixed_val.ipynb`: Add new images to a fixed validation set for consistent evaluation.
- `data_generation_trashcan.ipynb`: Auto-label trashcan images using detection and segmentation models.
- `demo_dino_sam.ipynb`: Demonstration of DINO and SAM models for detection and segmentation.
- `e2e_data_prep.ipynb`: End-to-end data preparation pipeline (cleaning, deduplication, formatting for YOLO).
- `inference.ipynb`: Run inference with trained models on new images.
- `training_v3.ipynb`: Full training pipeline for segmentation models (YOLO, multi-class, augmentation).

**Archived/Reference Notebooks:**
- `archived_notebooks/segmentation_pipeline.ipynb`: Example of the detection→segmentation pipeline.
- `archived_notebooks/segmentation_pipeline_standalone.ipynb`: Standalone segmentation pipeline demo.
- `archived_notebooks/train_trashcan.ipynb`: Training pipeline focused on trashcan class.
- `archived_notebooks/visualize_segmentation.ipynb`: Visualization utilities for segmentation results.
- `archived_notebooks/yolo.ipynb`: Basic YOLO model usage and training.
- `archived_notebooks/yoloseg_finetuning.ipynb`: Fine-tuning YOLO segmentation models.
- `archived_notebooks/yoloseg_finetuning_clean_backup.ipynb`: Backup of YOLO fine-tuning workflow.

## Usage: End-to-End Pipeline

1. **Data Preparation**: Use `e2e_data_prep.ipynb` to clean, deduplicate, and format datasets for YOLO training.
2. **Label Generation**: Run `data_generation_trashcan.ipynb` or scripts to auto-label images using detection and segmentation models.
3. **Training**: Use `training_v3.ipynb` to train segmentation models. For class-specific training, see `train_trashcan.ipynb` (archived).
4. **Inference**: Use `inference.ipynb` to run predictions on new images with trained models.
5. **Validation**: Use `add_to_fixed_val.ipynb` to maintain a fixed validation set for consistent evaluation.

## Modules & Packages

- `src/`
    - `pipeline.py`: Orchestrates detection→segmentation pipeline for images.
    - `detection.py`: Detection utilities (YOLO, DINO, Roboflow, etc.).
    - `segmentation.py`: Segmentation utilities (SAM, FastSAM integration).
    - `data_utils.py`: Data preparation, deduplication, and dataset utilities.
    - `augmentation.py`: Data augmentation for class balancing.
    - `visualization.py`: Visualization and monitoring tools for training and inference.
    - `config.py`: Centralized configuration loader (YAML-based).
- `utils.py`: Miscellaneous utilities (e.g., display helpers for notebooks).
- `configs/config.yaml`: All dataset/model/training configuration.

## Configuration
Edit `configs/config.yaml` to set dataset paths, model locations, and training parameters.

## Scripts (CLI)
Scripts for automation are in the `scripts/` folder (if present) or can be adapted from notebooks:
- Generate labels: detection + segmentation
- Train models
- Run inference
- Validate datasets

## Notes
- All workflows are reproducible via notebooks or scripts.
- For multi-class training (ball + person), combine datasets as shown in `training_v3.ipynb`.
- For new classes, adapt data prep and training notebooks accordingly.

## License
[Your License Here]

## Quick Start

### 1. Generate Labels

Auto-label images using detection + segmentation:

```bash
python scripts/generate_labels.py data/raw/balls \
    --detector yolo \
    --segmenter sam \
    --mode crop
```

### 2. Train Model

Train a ball segmentation model:

```bash
python scripts/train_model.py \
    --prepare-ball-dataset \
    --img-path data/raw/balls \
    --txt-path data/raw/txt_output_folder \
    --output-path data/processed/yolo_dataset \
    --epochs 10 \
    --batch-size 8
```

### 3. Run Inference

```bash
python scripts/inference.py \
    models/trained/ball_person_best.pt \
    path/to/test/images
```

### 4. Validate Dataset

```bash
python scripts/validate_dataset.py \
    data/processed/yolo_dataset \
    --split train \
    --visualize
```

## Workflows

### Ball Detection + Segmentation

1. **Data Collection**: Gather ball images → `data/raw/balls/`
2. **Label Generation**: Use `generate_labels.py` to create YOLO labels
3. **Dataset Preparation**: Split into train/val with `train_model.py --prepare-ball-dataset`
4. **Training**: Fine-tune YOLO11n-seg with head-only training
5. **Evaluation**: Validate mAP metrics and visualize predictions

### Multi-Class Training (Ball + Person)

1. Download COCO person subset (automated in `02_training.ipynb`)
2. Combine ball + person datasets
3. Train with 2 classes to prevent overfitting

### Trashcan Auto-Labeling

Use `01_data_generation_trashcan.ipynb` to:
1. Detect trashcans with pretrained YOLO
2. Segment with SAM
3. Export YOLO polygon labels

## Configuration

Edit `configs/config.yaml` to customize:
- Dataset paths
- Model paths
- Training hyperparameters
- Augmentation settings
- Inference parameters

Example:
```yaml
training:
  ball_person:
    epochs: 10
    batch_size: 8
    learning_rate: 0.001
    augmentation:
      hsv_s: 0.5
      fliplr: 0.5
```

## Key Features

- **Two-Stage Pipeline**: Detection provides ROI → Segmentation refines mask
- **Transfer Learning**: Head-only fine-tuning for fast training
- **Multi-Source Data**: Combines manual collection, web scraping, COCO
- **Automated Labeling**: Use SAM to create training data from detections
- **Flexible Configuration**: Centralized YAML config
- **CLI Tools**: Scriptable workflows for CI/CD

## Notebooks

- **01_data_generation_trashcan.ipynb**: Auto-label trashcan images
- **02_training.ipynb**: Complete training pipeline (ball + person)
- **03_inference.ipynb**: Demo inference on test images
