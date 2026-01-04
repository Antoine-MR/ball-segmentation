"""
Data augmentation utilities for trashcan dataset balancing.

This module provides tools to perform targeted augmentation on under-represented
classes (particularly trashcans) to balance the dataset distribution.
"""

from pathlib import Path
import hashlib
import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations import Compose
except ImportError:
    print("⚠️  albumentations not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'albumentations'])
    import albumentations as A
    from albumentations import Compose


def find_trashcan_training_images(dataset_path):
    """Find all training images that contain trashcans.
    
    Args:
        dataset_path: Path to the YOLO dataset root directory
        
    Returns:
        List of tuples (image_path, label_path) for images containing trashcans
    """
    train_labels_dir = dataset_path / 'train' / 'labels'
    train_images_dir = dataset_path / 'train' / 'images'
    
    trashcan_files = []
    
    for label_file in train_labels_dir.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                has_trashcan = any(line.strip().startswith('2 ') for line in f)
            
            if has_trashcan:
                img_name = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = train_images_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        trashcan_files.append((img_path, label_file))
                        break
        except:
            continue
    
    return trashcan_files


def create_augmentation_pipeline(aug_strength='strong'):
    """Create augmentation pipeline for trashcan images.
    
    Args:
        aug_strength: Either 'strong' or 'light' augmentation
        
    Returns:
        Albumentations Compose pipeline with keypoint support
    """
    if aug_strength == 'strong':
        return Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomScale(scale_limit=0.3, p=0.7),
            A.Affine(
                rotate=(-20, 20),
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.8, 1.2),
                shear=(-10, 10),
                p=0.7
            ),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomRotate90(p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def augment_image_with_labels(img_path, label_path, transform):
    """Apply augmentation to image and adjust YOLO polygon labels accordingly.
    
    Args:
        img_path: Path to input image
        label_path: Path to YOLO format label file
        transform: Albumentations transform pipeline
        
    Returns:
        Tuple of (augmented_image, augmented_labels)
        where augmented_labels is a list of (class_id, normalized_coords)
    """
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    augmented_labels = []
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        
        # Convert normalized polygon to pixel coordinates
        keypoints = []
        for i in range(0, len(coords), 2):
            x = coords[i] * w
            y = coords[i+1] * h
            keypoints.append((x, y))
        
        # Apply augmentation
        try:
            transformed = transform(image=img, keypoints=keypoints)
            aug_img = transformed['image']
            aug_keypoints = transformed['keypoints']
            
            # Convert back to normalized YOLO format
            aug_h, aug_w = aug_img.shape[:2]
            norm_coords = []
            for kp in aug_keypoints:
                norm_coords.append(kp[0] / aug_w)
                norm_coords.append(kp[1] / aug_h)
            
            # Clip coordinates to [0, 1]
            norm_coords = [max(0.0, min(1.0, c)) for c in norm_coords]
            
            augmented_labels.append((class_id, norm_coords))
        except:
            # If augmentation fails, keep original
            augmented_labels.append((class_id, coords))
            aug_img = img
    
    return aug_img, augmented_labels


def augment_trashcan_dataset(dataset_path, num_augmentations=15, aug_strength='strong'):
    """Create multiple augmented copies of trashcan images.
    
    This function finds all training images containing trashcans and creates
    multiple augmented variants to balance class distribution.
    
    Args:
        dataset_path: Path to the YOLO dataset root directory
        num_augmentations: Number of augmented copies to create per image
        aug_strength: Augmentation strength ('strong' or 'light')
        
    Returns:
        Number of successfully created augmented images
    """
    print("="*60)
    print("TRASHCAN DATA AUGMENTATION")
    print("="*60)
    
    # Find trashcan images
    trashcan_files = find_trashcan_training_images(dataset_path)
    print(f"\nFound {len(trashcan_files)} images with trashcans in training set")
    
    if len(trashcan_files) == 0:
        print("⚠️  No trashcan images to augment!")
        return 0
    
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline(aug_strength)
    
    augmented_count = 0
    
    print(f"\nCreating {num_augmentations} augmented copies per image...")
    print(f"Total augmentations to create: {len(trashcan_files) * num_augmentations}")
    
    for img_path, label_path in tqdm(trashcan_files, desc="Augmenting trashcans"):
        base_name = img_path.stem
        
        for aug_idx in range(num_augmentations):
            try:
                # Apply augmentation
                aug_img, aug_labels = augment_image_with_labels(img_path, label_path, transform)
                
                # Generate unique name
                unique_suffix = hashlib.md5(
                    f"{base_name}_{aug_idx}_{np.random.randint(1000000)}".encode()
                ).hexdigest()[:8]
                new_name = f"{base_name}_aug{aug_idx:02d}_{unique_suffix}"
                
                # Save augmented image
                new_img_path = train_images_dir / f"{new_name}.jpg"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(new_img_path), aug_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save augmented labels
                new_label_path = train_labels_dir / f"{new_name}.txt"
                with open(new_label_path, 'w') as f:
                    for class_id, coords in aug_labels:
                        coords_str = " ".join([f"{c:.6f}" for c in coords])
                        f.write(f"{class_id} {coords_str}\n")
                
                augmented_count += 1
                
            except Exception as e:
                print(f"  ⚠️  Failed to augment {img_path.name}: {e}")
                continue
    
    print(f"\n✅ Successfully created {augmented_count} augmented images")
    print(f"   Original trashcan images: {len(trashcan_files)}")
    print(f"   Augmented copies: {augmented_count}")
    print(f"   Total trashcan images now: {len(trashcan_files) + augmented_count}")
    print("="*60)
    
    return augmented_count
