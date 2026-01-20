# Ball Segmentation

YOLO11 segmentation model for three classes: red ball (0), human (1) and trashcan (2).

Training data is labeled automatically: GroundingDINO finds the boxes from a text prompt, SAM 2.1 turns each box into a mask, and the mask is saved as a YOLO polygon. COCO persons can be added as pre-labeled humans. The validation set (`datasets/ready/fixed_val`) is reviewed by hand and stays fixed between runs.

## Setup

Python 3.12 and a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # only needed for RoboflowDetector
```

Put the weights in `models/pretrained/`:

- `yolo11n-seg.pt`, `yolo11n.pt`
- `sam2.1_b.pt`, `FastSAM-s.pt`
- `groundingdino_swint_ogc.pth`, from the [GroundingDINO releases](https://github.com/IDEA-Research/GroundingDINO/releases)

## Workflow

1. `e2e_data_prep.ipynb`: converts the images in `SOURCE_DIR` to JPG, removes duplicates by hash, labels them with GroundingDINO + SAM and splits them 80/20 into train/test in `datasets/ready/full_dataset`. The val split is copied from `fixed_val`.
2. `add_to_fixed_val.ipynb`: labels new images the same way, writes previews for review, then adds them to `fixed_val`.
3. `training.ipynb`: augments trashcans and occluded balls, writes `data.yaml`, trains with the backbone frozen and saves per-epoch predictions on a few validation images to `runs/segment/<project>/training_monitor`.
4. `inference.ipynb`: runs a trained model on one image, optionally after a color preprocessing.

Also:

- `demo_dino_sam.ipynb`: GroundingDINO on a single image, to tune prompts and thresholds.
- `data_generation_trashcan.ipynb`: older trashcan labeling with YOLO + SAM.

## Code

| Path | Content |
|---|---|
| `src/detection.py` | `GroundingDINODetector`, `YOLODetector`, `RoboflowDetector` |
| `src/segmentation.py` | `SAMSegmenter`, `FastSAMSegmenter` (box-prompted) |
| `src/pipeline.py` | `img_pipeline`: detect, segment and write labels for one image |
| `src/data_utils.py` | Dataset copy with dedup, class counts, monitoring image selection |
| `src/augmentation.py` | Offline augmentation of one class (keypoint or mask based) |
| `src/visualization.py` | Training callback that plots predictions every epoch |
| `src/config.py` | Reads `configs/config.yaml` |
| `utils.py` | `DisplayPath`, a `Path` that renders as a link in notebooks |

Model paths and the default inference confidence come from `configs/config.yaml`.
