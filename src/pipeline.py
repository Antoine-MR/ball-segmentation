from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil


COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']


def img_pipeline(
    img_path: Path,
    detect_fn: Callable[[Path], dict[str, list[dict]]],
    segment_fn: Callable[..., Any],
    det_output_dir: Path,
    seg_output_dir: Path,
    empty_dir: Path,
    txt_output_dir: Path | None = None,
    images_output_dir: Path | None = None,
):
    """Detect, segment and write YOLO polygon labels for one image.

    detect_fn returns {label: [detections]}, segment_fn takes (image, bbox).
    Labels go to txt_output_dir/<label>/<stem>.txt, with class id 0 since
    each label has its own folder.
    """
    detections_by_label = detect_fn(img_path)

    if sum(len(dets) for dets in detections_by_label.values()) == 0:
        shutil.copy(img_path, empty_dir / img_path.name)
        if txt_output_dir is not None:
            for label in detections_by_label:
                label_dir = txt_output_dir / label
                label_dir.mkdir(exist_ok=True, parents=True)
                (label_dir / f"{img_path.stem}.txt").write_text("")
        return

    img_corrected = ImageOps.exif_transpose(Image.open(img_path))
    img_array = np.array(img_corrected)

    img_viz = img_corrected.copy()
    draw = ImageDraw.Draw(img_viz)
    label_to_color = {
        label: COLORS[i % len(COLORS)] for i, label in enumerate(detections_by_label)
    }

    font_size = max(20, int(min(img_viz.width, img_viz.height) * 0.03))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for label, detections in detections_by_label.items():
        color = label_to_color[label]
        for det in detections:
            bbox = det['bbox']
            draw.rectangle(bbox, outline=color, width=5)

            text = f"{label} ({det['confidence']:.2f})"
            text_x = bbox[0]
            text_y = max(0, bbox[1] - font_size - 5)  # type: ignore
            draw.rectangle(draw.textbbox((text_x, text_y), text, font=font), fill=color)
            draw.text((text_x, text_y), text, fill="white", font=font)

    det_output_dir.mkdir(exist_ok=True, parents=True)
    img_viz.save(det_output_dir / (img_path.stem + ".png"))

    h, w = img_array.shape[:2]
    combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
    label_colors_rgb = {
        label: tuple(int(c * 255) for c in mcolors.to_rgb(color_name))
        for label, color_name in label_to_color.items()
    }

    all_masks = []
    for label, detections in detections_by_label.items():
        color_rgb = label_colors_rgb[label]
        for det in detections:
            for result in segment_fn(img_corrected, det['bbox']):
                if result.masks is None:
                    continue
                mask_data = result.masks.data[0]
                all_masks.append((label, mask_data))

                mask_binary = mask_data.cpu().numpy() > 0.5
                for c in range(3):
                    combined_mask[:, :, c][mask_binary] = color_rgb[c]

    seg_output_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.imshow(combined_mask, alpha=0.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(seg_output_dir / (img_path.stem + ".png"), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    if txt_output_dir is not None:
        for label, mask_data in all_masks:
            label_dir = txt_output_dir / label
            label_dir.mkdir(exist_ok=True, parents=True)

            ys, xs = (mask_data > 0.5).nonzero(as_tuple=True)
            if len(xs) == 0:
                continue

            mask_h, mask_w = mask_data.shape
            xs_norm = xs.float() / float(mask_w)
            ys_norm = ys.float() / float(mask_h)

            parts: list[str] = ["0"]
            for x_val, y_val in zip(xs_norm.tolist(), ys_norm.tolist()):
                parts.append(f"{x_val:.6f}")
                parts.append(f"{y_val:.6f}")

            # Several instances of the same label end up in the same file
            txt_path = label_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                txt_path.write_text(txt_path.read_text() + "\n" + " ".join(parts))
            else:
                txt_path.write_text(" ".join(parts))

    if images_output_dir is not None:
        images_output_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(img_path, images_output_dir / img_path.name)
