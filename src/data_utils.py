"""
Data preparation utilities for YOLO segmentation dataset.
"""

from pathlib import Path
import shutil
import hashlib
import random
from tqdm import tqdm


def get_image_hash(file_path: Path) -> str:
    """Calculate MD5 hash of an image file for deduplication."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def copy_coco_persons(
    coco_dir: Path,
    output_images: Path,
    output_labels: Path,
    existing_hashes: set = None,
    max_images: int = None,
    prefix: str = "coco_"
) -> tuple[int, set]:
    """
    Copy COCO person dataset to the project.
    
    Args:
        coco_dir: Path to COCO dataset with images/ and labels/ subdirectories
        output_images: Destination directory for images
        output_labels: Destination directory for labels
        existing_hashes: Set of existing image hashes for deduplication
        max_images: Maximum number of images to copy (None = all)
        prefix: Prefix to add to filenames to avoid collisions
    
    Returns:
        Tuple of (copied_count, updated_hashes_set)
    """
    coco_images = coco_dir / "images"
    coco_labels = coco_dir / "labels"
    
    if not coco_images.exists() or not coco_labels.exists():
        print(f"⚠️  COCO dataset not found at {coco_dir}")
        return 0, existing_hashes or set()
    
    # Initialize hashes set
    hashes = existing_hashes.copy() if existing_hashes else set()
    
    # Get image-label pairs
    images = list(coco_images.glob("*.jpg")) + list(coco_images.glob("*.jpeg")) + list(coco_images.glob("*.png"))
    
    # Shuffle and limit if requested
    if max_images and len(images) > max_images:
        images = random.sample(images, max_images)
    
    print(f"📁 Found {len(images)} COCO images to process")
    
    # Ensure output directories exist
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    for img_path in tqdm(images, desc="Copying COCO"):
        stem = img_path.stem
        txt_path = coco_labels / f"{stem}.txt"
        
        if not txt_path.exists():
            continue
        
        # Check deduplication
        img_hash = get_image_hash(img_path)
        if img_hash in hashes:
            skipped_count += 1
            continue
        
        # Copy image with prefix to avoid collisions
        dest_img = output_images / f"{prefix}{img_path.name}"
        if not dest_img.exists():
            shutil.copy(img_path, dest_img)
        
        # Copy label (already in YOLO format with class ID 1 for humans)
        dest_txt = output_labels / f"{prefix}{stem}.txt"
        if not dest_txt.exists():
            shutil.copy(txt_path, dest_txt)
        
        # Update hashes
        hashes.add(img_hash)
        copied_count += 1
    
    print(f"✅ Copied {copied_count} COCO image-label pairs")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} duplicates")
    
    return copied_count, hashes


def copy_external_dataset(
    source_dir: Path,
    output_images: Path,
    output_labels: Path,
    existing_hashes: set = None,
    max_images: int = None,
    prefix: str = "",
    class_id_mapping: dict = None
) -> tuple[int, set]:
    """
    Copy an external pre-labeled dataset to the project.
    
    Args:
        source_dir: Path to dataset with images/ and labels/ subdirectories
        output_images: Destination directory for images
        output_labels: Destination directory for labels
        existing_hashes: Set of existing image hashes for deduplication
        max_images: Maximum number of images to copy (None = all)
        prefix: Prefix to add to filenames to avoid collisions
        class_id_mapping: Dict mapping source class IDs to target class IDs
                         e.g., {0: 1} to map class 0 -> class 1
    
    Returns:
        Tuple of (copied_count, updated_hashes_set)
    """
    src_images = source_dir / "images"
    src_labels = source_dir / "labels"
    
    if not src_images.exists() or not src_labels.exists():
        print(f"⚠️  Dataset not found at {source_dir}")
        return 0, existing_hashes or set()
    
    # Initialize hashes set
    hashes = existing_hashes.copy() if existing_hashes else set()
    
    # Get images
    images = list(src_images.glob("*.jpg")) + list(src_images.glob("*.jpeg")) + list(src_images.glob("*.png"))
    
    # Shuffle and limit if requested
    if max_images and len(images) > max_images:
        images = random.sample(images, max_images)
    
    print(f"📁 Found {len(images)} images to process")
    
    # Ensure output directories exist
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    for img_path in tqdm(images, desc=f"Copying {source_dir.name}"):
        stem = img_path.stem
        txt_path = src_labels / f"{stem}.txt"
        
        if not txt_path.exists():
            continue
        
        # Check deduplication
        img_hash = get_image_hash(img_path)
        if img_hash in hashes:
            skipped_count += 1
            continue
        
        # Copy image
        dest_img = output_images / f"{prefix}{img_path.name}"
        if not dest_img.exists():
            shutil.copy(img_path, dest_img)
        
        # Process labels (apply class ID mapping if needed)
        dest_txt = output_labels / f"{prefix}{stem}.txt"
        if not dest_txt.exists():
            if class_id_mapping:
                # Read, remap, and write
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                
                remapped_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    src_class = int(parts[0])
                    if src_class in class_id_mapping:
                        parts[0] = str(class_id_mapping[src_class])
                        remapped_lines.append(" ".join(parts))
                
                with open(dest_txt, 'w') as f:
                    f.write("\n".join(remapped_lines))
            else:
                # Direct copy
                shutil.copy(txt_path, dest_txt)
        
        # Update hashes
        hashes.add(img_hash)
        copied_count += 1
    
    print(f"✅ Copied {copied_count} image-label pairs")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} duplicates")
    
    return copied_count, hashes


def count_class_instances(dataset_path: Path, split: str) -> dict:
    """Count instances of each class in a dataset split"""
    label_dir = dataset_path / split / "labels"
    class_counts = {0: 0, 1: 0, 2: 0}  # red ball, human, trashcan
    
    for label_file in label_dir.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(float(parts[0]))
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except:
            continue
    
    return class_counts


def find_images_by_class(val_labels_dir: Path, val_images_dir: Path, target_class: int, max_images: int = 3) -> list:
    """Find validation images that contain a specific class.
    
    Args:
        val_labels_dir: Path to validation labels directory
        val_images_dir: Path to validation images directory
        target_class: Class ID to search for (0=red ball, 1=human, 2=trashcan)
        max_images: Maximum number of images to find
        
    Returns:
        List of image paths
    """
    class_images = []
    
    label_files = list(val_labels_dir.glob("*.txt"))
    
    for label_file in label_files:
        if len(class_images) >= max_images:
            break
            
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # Check if any line starts with target class
            has_class = any(line.strip().startswith(f'{target_class} ') for line in lines)
            
            if has_class:
                # Find corresponding image
                img_name = label_file.stem
                
                # Try different image extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_path = val_images_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        class_images.append(str(img_path))
                        break
        except:
            continue
    
    return class_images


def select_diverse_monitoring_images(val_labels_dir: Path, val_images_dir: Path, 
                                     images_per_class: int = 3, include_mixed: bool = True) -> list:
    """Select a diverse set of validation images covering all classes.
    
    Args:
        val_labels_dir: Path to validation labels directory
        val_images_dir: Path to validation images directory
        images_per_class: Number of images to select per class
        include_mixed: Whether to include images with multiple classes
        
    Returns:
        List of selected image paths
    """
    print("="*60)
    print("SELECTING DIVERSE MONITORING IMAGES")
    print("="*60)
    
    monitoring_images = []
    class_names = {0: 'Red Ball', 1: 'Human', 2: 'Trashcan'}
    
    # Find images for each class
    for class_id, class_name in class_names.items():
        print(f"\nSearching for {class_name} images...")
        class_imgs = find_images_by_class(val_labels_dir, val_images_dir, 
                                          class_id, max_images=images_per_class)
        
        if class_imgs:
            monitoring_images.extend(class_imgs)
            print(f"  ✓ Found {len(class_imgs)} images with {class_name}")
            for img in class_imgs:
                print(f"    - {Path(img).name}")
        else:
            print(f"  ⚠️  No images found with {class_name}")
    
    # Optionally add images with multiple classes
    if include_mixed and len(monitoring_images) < 12:
        print("\nSearching for mixed-class images...")
        label_files = list(val_labels_dir.glob("*.txt"))
        
        for label_file in label_files:
            if len(monitoring_images) >= 12:
                break
                
            try:
                with open(label_file, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                
                # Count unique classes
                classes_in_image = set(int(l.split()[0]) for l in lines if l)
                
                # If image has 2+ classes and we haven't already selected it
                if len(classes_in_image) >= 2:
                    img_name = label_file.stem
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = val_images_dir / f"{img_name}{ext}"
                        if img_path.exists() and str(img_path) not in monitoring_images:
                            monitoring_images.append(str(img_path))
                            class_str = ", ".join([class_names[c] for c in sorted(classes_in_image)])
                            print(f"  ✓ Mixed: {img_path.name} ({class_str})")
                            break
            except:
                continue
    
    print(f"\n{'='*60}")
    print(f"Selected {len(monitoring_images)} images for monitoring")
    print(f"{'='*60}\n")
    
    return monitoring_images
