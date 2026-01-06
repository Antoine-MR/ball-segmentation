"""
Data augmentation utilities for dataset balancing.

This module provides tools to perform targeted augmentation on under-represented
classes to balance the dataset distribution.
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


def find_class_training_images(dataset_path, class_id, exclude_augmented=True):
    """Find all training images that contain a specific class.
    
    Args:
        dataset_path: Path to the YOLO dataset root directory
        class_id: The class ID to search for (int)
        exclude_augmented: If True, skip images with '_aug' in filename
        
    Returns:
        List of tuples (image_path, label_path) for images containing the class
    """
    train_labels_dir = dataset_path / 'train' / 'labels'
    train_images_dir = dataset_path / 'train' / 'images'
    
    class_files = []
    class_prefix = f"{class_id} "
    
    for label_file in train_labels_dir.glob("*.txt"):
        # Skip augmented images to avoid re-augmenting them
        if exclude_augmented and '_aug' in label_file.stem:
            continue
            
        try:
            with open(label_file, 'r') as f:
                has_class = any(line.strip().startswith(class_prefix) for line in f)
            
            if has_class:
                img_name = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = train_images_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        class_files.append((img_path, label_file))
                        break
        except:
            continue
    
    return class_files


def get_augmentation_pipeline(config: Compose | str ='strong'):
    """Get augmentation pipeline from config or preset name.
    
    Args:
        config: Either a string ('strong', 'light') or an albumentations.Compose object
        
    Returns:
        Albumentations Compose pipeline with keypoint support
    """
    # If config is already a Compose object, return it
    if isinstance(config, A.Compose):
        return config

    if config == 'strong':
        return Compose([  # type: ignore
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
    elif config == 'light':
        return Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomRotate90(p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        raise ValueError(f"Unknown augmentation config: {config}")


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


def augment_class_dataset(dataset_path, class_id, num_augmentations=15, aug_config: Compose | str ='strong'):
    """Create multiple augmented copies of images containing a specific class.
    
    Args:
        dataset_path: Path to the YOLO dataset root directory
        class_id: The class ID to target
        num_augmentations: Number of augmented copies to create per image
        aug_config: Augmentation config ('strong', 'light', or A.Compose object)
        
    Returns:
        Number of successfully created augmented images
    """
    print(f"\n{'='*60}")
    print(f"AUGMENTATION FOR CLASS {class_id}")
    print(f"{'='*60}")
    
    # Find images containing the class
    class_files = find_class_training_images(dataset_path, class_id, exclude_augmented=True)
    print(f"\nFound {len(class_files)} ORIGINAL images with class {class_id}")
    
    if len(class_files) == 0:
        print(f"⚠️  No images found for class {class_id}!")
        return 0
    
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'
    
    # Create augmentation pipeline
    transform = get_augmentation_pipeline(aug_config)
    
    augmented_count = 0
    
    print(f"Creating {num_augmentations} augmented copies per image...")
    
    for img_path, label_path in tqdm(class_files, desc=f"Augmenting class {class_id}"):
        base_name = img_path.stem
        
        for aug_idx in range(num_augmentations):
            try:
                # Apply augmentation
                aug_img, aug_labels = augment_image_with_labels(img_path, label_path, transform)
                
                # Generate unique name with class ID to avoid collisions
                unique_suffix = hashlib.md5(
                    f"{base_name}_{class_id}_{aug_idx}_{np.random.randint(1000000)}".encode()
                ).hexdigest()[:8]
                
                # Format: original_aug_c{class_id}_{idx}_{hash}
                new_name = f"{base_name}_aug_c{class_id}_{aug_idx:02d}_{unique_suffix}"
                
                # Save augmented image
                new_img_path = train_images_dir / f"{new_name}.jpg"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(new_img_path), aug_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save augmented labels
                new_label_path = train_labels_dir / f"{new_name}.txt"
                with open(new_label_path, 'w') as f:
                    for cid, coords in aug_labels:
                        coords_str = " ".join([f"{c:.6f}" for c in coords])
                        f.write(f"{cid} {coords_str}\n")
                
                augmented_count += 1
                
            except Exception as e:
                print(f"Failed to augment {img_path.name}: {e}")
                continue
    
    print(f"Created {augmented_count} augmented images for class {class_id}")
    return augmented_count


def augment_image_with_masks(img_path, label_path, transform):
    """Apply augmentation using masks to handle occlusion and label updates correctly.
    
    Args:
        img_path: Path to input image
        label_path: Path to YOLO format label file
        transform: Albumentations transform pipeline (must support masks)
        
    Returns:
        Tuple of (augmented_image, augmented_labels)
    """
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    masks = []
    class_ids = []
    
    # Convert polygons to masks
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        
        # Create binary mask for this object
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = []
        for i in range(0, len(coords), 2):
            pts.append([int(coords[i] * w), int(coords[i+1] * h)])
        
        if pts:
            pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, pts, 1)
            masks.append(mask)
            class_ids.append(class_id)
    
    if not masks:
        return img, []

    # Apply augmentation (Image + Masks)
    # This ensures geometric consistency and handles occlusion (CoarseDropout)
    try:
        transformed = transform(image=img, masks=masks)
        aug_img = transformed['image']
        aug_masks = transformed['masks']
        
        aug_h, aug_w = aug_img.shape[:2]
        augmented_labels = []
        
        # Convert masks back to polygons
        for i, mask in enumerate(aug_masks):
            if np.sum(mask) < 10:  # Skip if object is mostly gone
                continue
                
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (main part of the object)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) < 50:  # Filter tiny fragments
                    continue
                    
                # Simplify contour
                epsilon = 0.005 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                
                # Normalize coordinates
                norm_coords = []
                for pt in approx:
                    x, y = pt[0]
                    norm_coords.append(min(1.0, max(0.0, x / aug_w)))
                    norm_coords.append(min(1.0, max(0.0, y / aug_h)))
                
                if len(norm_coords) >= 6:  # At least 3 points
                    augmented_labels.append((class_ids[i], norm_coords))
                    
        return aug_img, augmented_labels
        
    except Exception as e:
        print(f"Augmentation failed: {e}")
        return img, []


def augment_class_with_occlusion(dataset_path, class_id, num_augmentations=5):
    """Augment a class with occlusion (hiding parts) and scaling.
    
    Uses mask-based augmentation to correctly update labels when parts are hidden.
    """
    print(f"\n{'='*60}")
    print(f"OCCLUSION AUGMENTATION FOR CLASS {class_id}")
    print(f"{'='*60}")
    
    # Define pipeline with CoarseDropout (occlusion) and Affine (scaling)
    # mask_fill_value=0 ensures labels are updated where holes are punched
    pipeline = Compose([
        A.CoarseDropout(
            max_holes=3, max_height=50, max_width=50, 
            min_holes=1, min_height=20, min_width=20,
            fill_value=0, mask_fill_value=0, p=0.8
        ),
        A.Affine(scale=(0.5, 0.9), p=0.7),  # Zoom out (make smaller)
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    
    class_files = find_class_training_images(dataset_path, class_id, exclude_augmented=True)
    print(f"Found {len(class_files)} images to augment")
    
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'
    
    count = 0
    for img_path, label_path in tqdm(class_files, desc="Applying occlusion"):
        base_name = img_path.stem
        
        for i in range(num_augmentations):
            aug_img, aug_labels = augment_image_with_masks(img_path, label_path, pipeline)
            
            if not aug_labels:
                continue
                
            # Save
            suffix = hashlib.md5(f"{base_name}_{i}".encode()).hexdigest()[:6]
            new_name = f"{base_name}_occ_{suffix}"
            
            cv2.imwrite(str(train_images_dir / f"{new_name}.jpg"), 
                       cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            
            with open(train_labels_dir / f"{new_name}.txt", 'w') as f:
                for cid, coords in aug_labels:
                    f.write(f"{cid} " + " ".join(map(str, coords)) + "\n")
            
            count += 1
            
    print(f"Created {count} occluded images")
    return count


def clean_augmented_images(dataset_path: Path) -> int:
    """Remove all augmented images (with '_aug' in filename) from training set.
    
    Args:
        dataset_path: Path to the YOLO dataset root directory
        
    Returns:
        Number of removed files (images + labels)
    """
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'
    
    removed_count = 0
    
    print("="*60)
    print("CLEANING AUGMENTED IMAGES")
    print("="*60)
    
    # Remove augmented images
    aug_images = list(train_images_dir.glob("*_aug*"))
    for img_path in aug_images:
        img_path.unlink()
        removed_count += 1
    
    # Remove augmented labels
    aug_labels = list(train_labels_dir.glob("*_aug*"))
    for lbl_path in aug_labels:
        lbl_path.unlink()
        removed_count += 1
    
    print(f"\n✓ Removed {len(aug_images)} augmented images")
    print(f"✓ Removed {len(aug_labels)} augmented labels")
    print(f"Total files removed: {removed_count}")
    print("="*60)
    
    return removed_count
