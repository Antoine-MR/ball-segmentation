"""Offline augmentation of under-represented classes in a YOLO segmentation dataset."""

from pathlib import Path
import hashlib
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations import Compose


def find_class_training_images(dataset_path, class_id, exclude_augmented=True):
    """Return (image_path, label_path) pairs of train images containing class_id."""
    train_labels_dir = dataset_path / 'train' / 'labels'
    train_images_dir = dataset_path / 'train' / 'images'

    class_files = []
    class_prefix = f"{class_id} "

    for label_file in train_labels_dir.glob("*.txt"):
        # Augmenting an augmented image compounds the transforms
        if exclude_augmented and '_aug' in label_file.stem:
            continue

        with open(label_file, 'r') as f:
            has_class = any(line.strip().startswith(class_prefix) for line in f)
        if not has_class:
            continue

        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = train_images_dir / f"{label_file.stem}{ext}"
            if img_path.exists():
                class_files.append((img_path, label_file))
                break

    return class_files


def get_augmentation_pipeline(config: Compose | str = 'strong'):
    """Return a Compose as is, or build the 'strong' / 'light' preset."""
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
    if config == 'light':
        return Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomRotate90(p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    raise ValueError(f"Unknown augmentation config: {config}")


def augment_image_with_labels(img_path, label_path, transform):
    """Augment an image, moving its polygon vertices as keypoints.

    Returns (augmented_image, [(class_id, normalized_coords)]).
    """
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    aug_img = img
    augmented_labels = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        keypoints = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]

        try:
            transformed = transform(image=img, keypoints=keypoints)
        except Exception:
            augmented_labels.append((class_id, coords))
            aug_img = img
            continue

        aug_img = transformed['image']
        aug_h, aug_w = aug_img.shape[:2]
        norm_coords = []
        for x, y in transformed['keypoints']:
            norm_coords.append(max(0.0, min(1.0, x / aug_w)))
            norm_coords.append(max(0.0, min(1.0, y / aug_h)))
        augmented_labels.append((class_id, norm_coords))

    return aug_img, augmented_labels


def augment_class_dataset(dataset_path, class_id, num_augmentations=15, aug_config: Compose | str = 'strong'):
    """Write num_augmentations augmented copies of every train image containing class_id.

    Returns the number of images created.
    """
    class_files = find_class_training_images(dataset_path, class_id, exclude_augmented=True)
    print(f"Found {len(class_files)} original images with class {class_id}")

    if len(class_files) == 0:
        return 0

    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'
    transform = get_augmentation_pipeline(aug_config)
    augmented_count = 0

    for img_path, label_path in tqdm(class_files, desc=f"Augmenting class {class_id}"):
        base_name = img_path.stem

        for aug_idx in range(num_augmentations):
            try:
                aug_img, aug_labels = augment_image_with_labels(img_path, label_path, transform)

                unique_suffix = hashlib.md5(
                    f"{base_name}_{class_id}_{aug_idx}_{np.random.randint(1000000)}".encode()
                ).hexdigest()[:8]
                new_name = f"{base_name}_aug_c{class_id}_{aug_idx:02d}_{unique_suffix}"

                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(train_images_dir / f"{new_name}.jpg"), aug_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                with open(train_labels_dir / f"{new_name}.txt", 'w') as f:
                    for cid, coords in aug_labels:
                        f.write(f"{cid} {' '.join(f'{c:.6f}' for c in coords)}\n")

                augmented_count += 1
            except Exception as e:
                print(f"Failed to augment {img_path.name}: {e}")

    print(f"Created {augmented_count} augmented images for class {class_id}")
    return augmented_count


def augment_image_with_masks(img_path, label_path, transform):
    """Augment an image through rasterized masks, so occlusion actually cuts the labels.

    Returns (augmented_image, [(class_id, normalized_coords)]).
    """
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    masks = []
    class_ids = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        coords = [float(p) for p in parts[1:]]
        pts = [[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)]
        if not pts:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([pts], dtype=np.int32), 1)
        masks.append(mask)
        class_ids.append(int(parts[0]))

    if not masks:
        return img, []

    try:
        transformed = transform(image=img, masks=masks)
    except Exception as e:
        print(f"Augmentation failed: {e}")
        return img, []

    aug_img = transformed['image']
    aug_h, aug_w = aug_img.shape[:2]
    augmented_labels = []

    for class_id, mask in zip(class_ids, transformed['masks']):
        if np.sum(mask) < 10:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Occlusion can split an object, keep its largest visible part
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 50:
            continue

        approx = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True)
        norm_coords = []
        for pt in approx:
            x, y = pt[0]
            norm_coords.append(min(1.0, max(0.0, x / aug_w)))
            norm_coords.append(min(1.0, max(0.0, y / aug_h)))

        if len(norm_coords) >= 6:
            augmented_labels.append((class_id, norm_coords))

    return aug_img, augmented_labels


def augment_class_with_occlusion(dataset_path, class_id, num_augmentations=5):
    """Add occluded and zoomed-out copies of train images containing class_id."""
    # mask_fill_value=0 removes the holes from the labels too
    pipeline = Compose([
        A.CoarseDropout(
            max_holes=3, max_height=50, max_width=50,
            min_holes=1, min_height=20, min_width=20,
            fill_value=0, mask_fill_value=0, p=0.8
        ),
        A.Affine(scale=(0.5, 0.9), p=0.7),
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

            suffix = hashlib.md5(f"{base_name}_{i}".encode()).hexdigest()[:6]
            new_name = f"{base_name}_occ_{suffix}"

            cv2.imwrite(str(train_images_dir / f"{new_name}.jpg"), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            with open(train_labels_dir / f"{new_name}.txt", 'w') as f:
                for cid, coords in aug_labels:
                    f.write(f"{cid} " + " ".join(map(str, coords)) + "\n")

            count += 1

    print(f"Created {count} occluded images")
    return count


def clean_augmented_images(dataset_path: Path) -> int:
    """Delete '_aug' images and labels from the train split. Returns the number of files removed."""
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'

    aug_images = list(train_images_dir.glob("*_aug*"))
    aug_labels = list(train_labels_dir.glob("*_aug*"))
    for path in aug_images + aug_labels:
        path.unlink()

    print(f"Removed {len(aug_images)} augmented images and {len(aug_labels)} labels")
    return len(aug_images) + len(aug_labels)
