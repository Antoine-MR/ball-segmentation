from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil


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
    """
    Pipeline multi-classes pour détection et segmentation
    
    Args:
        img_path: Chemin vers l'image
        detect_fn: Fonction de détection retournant dict[label, list[detections]]
        segment_fn: Fonction de segmentation acceptant (img, bbox)
        det_output_dir: Dossier de sortie pour visualisation des détections
        seg_output_dir: Dossier de sortie pour visualisation des segmentations
        txt_output_dir: Dossier racine des labels (sous-dossiers par label)
        empty_dir: Dossier pour images sans détections
        images_output_dir: Dossier pour copier les images originales (ready/images)
    """
    detections_by_label = detect_fn(img_path)
    
    # Vérifier si aucune détection
    total_detections = sum(len(dets) for dets in detections_by_label.values())
    if total_detections == 0:
        shutil.copy(img_path, empty_dir / img_path.name)
        
        if txt_output_dir is not None:
            # Créer fichiers vides pour chaque label
            for label in detections_by_label.keys():
                label_dir = txt_output_dir / label
                label_dir.mkdir(exist_ok=True, parents=True)
                txt_path = label_dir / f"{img_path.stem}.txt"
                txt_path.write_text("")
        return

    # Charger l'image une seule fois avec correction EXIF
    img = Image.open(img_path)
    img_corrected = ImageOps.exif_transpose(img)
    img_array = np.array(img_corrected)
    
    # === 1. DETECTION VISUALIZATION ===
    img_viz = img_corrected.copy()
    draw = ImageDraw.Draw(img_viz)
    
    # Palette de couleurs pour les différentes classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    label_to_color = {}
    color_idx = 0
    
    try:
        font_size = max(20, int(min(img_viz.width, img_viz.height) * 0.03))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Dessiner toutes les détections
    for label, detections in detections_by_label.items():
        if label not in label_to_color:
            label_to_color[label] = colors[color_idx % len(colors)]
            color_idx += 1
        
        color = label_to_color[label]
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Bbox
            draw.rectangle(bbox, outline=color, width=5)
            
            # Texte
            text = f"{label} ({confidence:.2f})"
            text_x = bbox[0]
            text_y = max(0, bbox[1] - font_size - 5)  # type: ignore
            
            bbox_text = draw.textbbox((text_x, text_y), text, font=font)
            draw.rectangle(bbox_text, fill=color)
            draw.text((text_x, text_y), text, fill="white", font=font)
    
    det_output_dir.mkdir(exist_ok=True, parents=True)
    img_viz.save(det_output_dir / (img_path.stem + ".png"))
    
    # === 2. SEGMENTATION ===
    # Créer un masque combiné pour visualisation (avec couleurs par classe)
    h, w = img_array.shape[:2]
    combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Mapper les labels vers des couleurs RGB
    label_colors_rgb = {}
    for label, color_name in label_to_color.items():
        rgb = mcolors.to_rgb(color_name)
        label_colors_rgb[label] = tuple(int(c * 255) for c in rgb)
    
    all_masks_data = []  # Pour sauvegarder les txt files
    
    for label, detections in detections_by_label.items():
        color_rgb = label_colors_rgb[label]
        
        for det in detections:
            bbox = det['bbox']
            
            # Segmenter
            results = segment_fn(img_corrected, bbox)
            
            for result in results:
                if result.masks is not None:
                    mask_data = result.masks.data[0]
                    
                    # Stocker pour génération txt
                    all_masks_data.append({
                        'label': label,
                        'mask': mask_data,
                        'bbox': bbox
                    })
                    
                    # Appliquer la couleur au masque combiné
                    mask_binary = (mask_data.cpu().numpy() > 0.5)
                    for c in range(3):
                        combined_mask[:, :, c][mask_binary] = color_rgb[c]
    
    # === 3. SAVE SEGMENTATION VISUALIZATION ===
    seg_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Créer une visualisation avec l'image originale + overlay du masque
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.imshow(combined_mask, alpha=0.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(seg_output_dir / (img_path.stem + ".png"), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    
    # === 4. SAVE LABELS (TXT FILES) ===
    if txt_output_dir is not None:
        for mask_info in all_masks_data:
            label = mask_info['label']
            mask_data = mask_info['mask']
            
            # Créer le sous-dossier pour ce label
            label_dir = txt_output_dir / label
            label_dir.mkdir(exist_ok=True, parents=True)
            
            # Extraire les points du masque
            ys, xs = (mask_data > 0.5).nonzero(as_tuple=True)
            if len(xs) == 0:
                continue

            mask_h, mask_w = mask_data.shape
            xs_norm = xs.float() / float(mask_w)
            ys_norm = ys.float() / float(mask_h)

            parts: list[str] = ["0"]  # Class ID (toujours 0 dans le contexte d'un label spécifique)
            for x_val, y_val in zip(xs_norm.tolist(), ys_norm.tolist()):
                parts.append(f"{x_val:.6f}")
                parts.append(f"{y_val:.6f}")

            # Sauvegarder dans le dossier du label
            txt_path = label_dir / f"{img_path.stem}.txt"
            
            # Si le fichier existe déjà, append (multiple instances de la même classe)
            if txt_path.exists():
                existing = txt_path.read_text()
                txt_path.write_text(existing + "\n" + " ".join(parts))
            else:
                txt_path.write_text(" ".join(parts))
    
    # === 5. COPY ORIGINAL IMAGE ===
    if images_output_dir is not None:
        images_output_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(img_path, images_output_dir / img_path.name)
