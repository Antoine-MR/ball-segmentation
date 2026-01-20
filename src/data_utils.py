from pathlib import Path
import shutil
import hashlib
import random
from tqdm import tqdm


IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
CLASS_NAMES = {0: 'Red Ball', 1: 'Human', 2: 'Trashcan'}


def get_image_hash(file_path: Path) -> str:
    """MD5 of the file content, used for deduplication."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def copy_external_dataset(
    source_dir: Path,
    output_images: Path,
    output_labels: Path,
    existing_hashes: set | None = None,
    max_images: int | None = None,
    prefix: str = "",
    class_id_mapping: dict | None = None
) -> tuple[int, set]:
    """Copy a labeled dataset (source_dir/images, source_dir/labels) into the project.

    Images whose hash is already in existing_hashes are skipped. With
    class_id_mapping (e.g. {0: 1}), only the mapped classes are kept.
    Returns (copied_count, updated_hashes).
    """
    src_images = source_dir / "images"
    src_labels = source_dir / "labels"

    if not src_images.exists() or not src_labels.exists():
        print(f"Dataset not found at {source_dir}")
        return 0, existing_hashes or set()

    hashes = existing_hashes.copy() if existing_hashes else set()

    images = [p for ext in IMAGE_EXTENSIONS for p in src_images.glob(f"*{ext}")]
    if max_images and len(images) > max_images:
        images = random.sample(images, max_images)
    print(f"Found {len(images)} images in {source_dir}")

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0

    for img_path in tqdm(images, desc=f"Copying {source_dir.name}"):
        txt_path = src_labels / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue

        img_hash = get_image_hash(img_path)
        if img_hash in hashes:
            skipped_count += 1
            continue

        dest_img = output_images / f"{prefix}{img_path.name}"
        if not dest_img.exists():
            shutil.copy(img_path, dest_img)

        dest_txt = output_labels / f"{prefix}{img_path.stem}.txt"
        if not dest_txt.exists():
            if class_id_mapping:
                remapped_lines = []
                for line in txt_path.read_text().splitlines():
                    parts = line.split()
                    if parts and int(parts[0]) in class_id_mapping:
                        parts[0] = str(class_id_mapping[int(parts[0])])
                        remapped_lines.append(" ".join(parts))
                dest_txt.write_text("\n".join(remapped_lines))
            else:
                shutil.copy(txt_path, dest_txt)

        hashes.add(img_hash)
        copied_count += 1

    print(f"Copied {copied_count} image-label pairs, skipped {skipped_count} duplicates")
    return copied_count, hashes


def copy_coco_persons(
    coco_dir: Path,
    output_images: Path,
    output_labels: Path,
    existing_hashes: set | None = None,
    max_images: int | None = None,
    prefix: str = "coco_"
) -> tuple[int, set]:
    """Copy the COCO person subset, whose labels already use class 1 (human)."""
    return copy_external_dataset(coco_dir, output_images, output_labels,
                                 existing_hashes, max_images, prefix)


def count_class_instances(dataset_path: Path, split: str) -> dict:
    """Count label instances per class id in a split."""
    class_counts = {class_id: 0 for class_id in CLASS_NAMES}

    for label_file in (dataset_path / split / "labels").glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if parts:
                class_id = int(float(parts[0]))
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    return class_counts


def _find_image(images_dir: Path, stem: str, extensions: list[str]) -> Path | None:
    for ext in extensions:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def find_images_by_class(val_labels_dir: Path, val_images_dir: Path, target_class: int, max_images: int = 3) -> list:
    """Return up to max_images image paths (str) whose labels contain target_class."""
    class_images = []
    extensions = IMAGE_EXTENSIONS + [ext.upper() for ext in IMAGE_EXTENSIONS]

    for label_file in val_labels_dir.glob("*.txt"):
        if len(class_images) >= max_images:
            break

        lines = label_file.read_text().splitlines()
        if not any(line.strip().startswith(f'{target_class} ') for line in lines):
            continue

        img_path = _find_image(val_images_dir, label_file.stem, extensions)
        if img_path is not None:
            class_images.append(str(img_path))

    return class_images


def select_diverse_monitoring_images(val_labels_dir: Path, val_images_dir: Path,
                                     images_per_class: int = 3, include_mixed: bool = True) -> list:
    """Pick images_per_class validation images per class, then fill up to 12 with multi-class images."""
    monitoring_images = []

    for class_id, class_name in CLASS_NAMES.items():
        class_imgs = find_images_by_class(val_labels_dir, val_images_dir,
                                          class_id, max_images=images_per_class)
        monitoring_images.extend(class_imgs)
        print(f"{class_name}: {len(class_imgs)} images")

    if include_mixed:
        for label_file in val_labels_dir.glob("*.txt"):
            if len(monitoring_images) >= 12:
                break

            lines = [l.strip() for l in label_file.read_text().splitlines() if l.strip()]
            if len({int(l.split()[0]) for l in lines}) < 2:
                continue

            img_path = _find_image(val_images_dir, label_file.stem, IMAGE_EXTENSIONS)
            if img_path is not None and str(img_path) not in monitoring_images:
                monitoring_images.append(str(img_path))

    print(f"Selected {len(monitoring_images)} images for monitoring")
    return monitoring_images
