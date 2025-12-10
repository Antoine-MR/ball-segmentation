# Ball Segmentation Project

Computer vision project for training YOLO segmentation models on multiple object classes (balls, persons, trashcans) using a two-stage detection + segmentation pipeline.

## Project Structure

```
ball_segmentation/
├── configs/               # Configuration files
│   └── config.yaml       # Central configuration
├── src/                  # Core library code
│   ├── detection.py      # Detection classes (YOLO, Roboflow)
│   ├── segmentation.py   # Segmentation classes (SAM, FastSAM)
│   ├── pipeline.py       # Detection→Segmentation pipeline
│   └── config.py         # Configuration loader
├── scripts/              # CLI tools
│   ├── generate_labels.py    # Auto-label images with detection+segmentation
│   ├── train_model.py        # Train YOLO segmentation models
│   ├── inference.py          # Run inference on images
│   └── validate_dataset.py   # Validate dataset quality
├── notebooks/            # Jupyter notebooks for exploration
│   ├── 01_data_generation_trashcan.ipynb
│   ├── 02_training.ipynb
│   ├── 03_inference.ipynb
│   └── archive/          # Old notebook versions
├── data/                 # Datasets
│   ├── raw/              # Original datasets
│   └── processed/        # Prepared YOLO datasets
├── models/               # Model weights
│   ├── pretrained/       # Base models (YOLO, SAM)
│   └── trained/          # Fine-tuned models
├── runs/                 # Training outputs
└── docs/                 # Documentation
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository
2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pretrained models and place in `models/pretrained/`:
   - `yolo11n-seg.pt` - YOLO segmentation base model
   - `yolo11n.pt` - YOLO detection model
   - `sam2.1_b.pt` - SAM 2.1 segmentation model
   - `FastSAM-s.pt` - FastSAM model

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

## Model Performance

Training uses:
- **Backbone Freezing**: Only detection head is trained
- **Data Augmentation**: HSV, flip, scale, mosaic
- **Class Balancing**: Person class prevents ball overfitting
- **Metrics**: mAP50, mAP50-95 for both box and mask

## Notebooks

- **01_data_generation_trashcan.ipynb**: Auto-label trashcan images
- **02_training.ipynb**: Complete training pipeline (ball + person)
- **03_inference.ipynb**: Demo inference on test images

## Architecture

See `docs/ARCHITECTURE.md` for detailed design decisions and technical rationale.

## Contributing

When making changes:
1. Create a test in `cline_temp_tests/` if no test framework exists
2. Follow conventional commits for commit messages
3. Never commit without authorization

## License

[Your License Here]
