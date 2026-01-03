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
