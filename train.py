#!/usr/bin/env python3
"""
YOLO Training Script - Standalone version of training_v2.ipynb
Run in tmux to keep training even after disconnection
"""

from pathlib import Path
import random
import yaml
from ultralytics.models import YOLO
import torch

# Seed for reproducibility
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path (created by e2e_data_prep.ipynb)
YOLO_DATASET = Path("datasets/ready/full_dataset")
RUNS_DIR = Path("runs/segment")

# Training hyperparameters
EPOCHS = 500
BATCH_SIZE = 16
IMG_SIZE = 640
model_type = "yolo11n-seg.pt"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PROJECT_NAME = 'ball_person_model'

# Data augmentation configuration
AUG_CONFIG = {
    'hsv_h': 0.015,  # Hue augmentation
    'hsv_s': 0.7,    # Saturation
    'hsv_v': 0.4,    # Value
    'degrees': 10.0,  # Rotation
    'translate': 0.1, # Translation
    'scale': 0.5,     # Scaling
    'shear': 0.0,     # Shearing
    'perspective': 0.0, # Perspective
    'flipud': 0.0,    # Vertical flip
    'fliplr': 0.5,    # Horizontal flip
    'mosaic': 1.0,    # Mosaic augmentation
    'mixup': 0.0,     # Mixup augmentation
}

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("="*60)
    print("YOLO TRAINING - Starting...")
    print("="*60)
    
    # Verify dataset exists
    if not YOLO_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found at {YOLO_DATASET}. Run e2e_data_prep.ipynb first!")
    
    print(f"Dataset: {YOLO_DATASET}")
    print(f"  Train: {YOLO_DATASET / 'train'}")
    print(f"  Val: {YOLO_DATASET / 'val'}")
    print(f"  Test: {YOLO_DATASET / 'test'}")
    
    # Device info
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    
    # Verify dataset structure
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        img_dir = YOLO_DATASET / split / "images"
        lbl_dir = YOLO_DATASET / split / "labels"
        
        if img_dir.exists() and lbl_dir.exists():
            num_images = len(list(img_dir.glob("*")))
            num_labels = len(list(lbl_dir.glob("*.txt")))
            stats[split] = {'images': num_images, 'labels': num_labels}
            print(f"{split.upper():5s}: {num_images:4d} images, {num_labels:4d} labels")
        else:
            stats[split] = {'images': 0, 'labels': 0}
            print(f"{split.upper():5s}: Missing!")
    
    total_images = sum(s['images'] for s in stats.values())
    total_labels = sum(s['labels'] for s in stats.values())
    
    print(f"{'TOTAL':5s}: {total_images:4d} images, {total_labels:4d} labels")
    print("="*60)
    
    if total_images == 0:
        raise RuntimeError("No dataset found! Run e2e_data_prep.ipynb to create the dataset.")
    
    # Create YOLO configuration file
    config = {
        'path': str(YOLO_DATASET.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': ['red ball', 'human']
    }
    
    config_path = YOLO_DATASET / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ Configuration saved: {config_path}")
    
    # Load pretrained model
    print(f"\n📦 Loading model: {model_type}")
    model = YOLO(model_type)
    
    # Find head layer index for freezing
    head_idx = next((i for i, m in enumerate(model.model.model) 
                     if 'Detect' in m.__class__.__name__ or 'Segment' in m.__class__.__name__), 
                    len(model.model.model) - 1)
    
    print(f"🔒 Freezing layers 0-{head_idx-1} (training head only)")
    
    # Start training
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Project: {RUNS_DIR / PROJECT_NAME}")
    print("="*60 + "\n")
    
    results = model.train(
        data=str(config_path),
        epochs=EPOCHS,
        freeze=list(range(head_idx)),
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=PROJECT_NAME,
        exist_ok=True,
        
        # Checkpointing
        save=True,
        save_period=1,  # Save every epoch
        
        # Validation
        val=True,
        
        # Data augmentation (only applied to train)
        **AUG_CONFIG,
        
        # Optimizer
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Other
        patience=20,  # Early stopping
        workers=8,
        verbose=True
    )
    
    # Validation metrics
    print("\n" + "="*60)
    print("📊 FINAL VALIDATION METRICS")
    print("="*60)
    metrics = model.val()
    
    print(f"Box mAP50: {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    
    # Model paths
    model_dir = RUNS_DIR / PROJECT_NAME
    best_model = model_dir / 'weights' / 'best.pt'
    last_model = model_dir / 'weights' / 'last.pt'
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {best_model}")
    print(f"   Last model: {last_model}")
    print(f"   Results: {model_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
