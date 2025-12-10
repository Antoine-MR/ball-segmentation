from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps


def img_pipeline(
    img_path: Path,
    detect_fn: Callable[[Path], dict | None],
    segment_fn: Callable[..., Any],
    det_output_dir: Path = Path("detection_output"),
    seg_output_dir: Path = Path("sam_output"),
    txt_output_dir: Path | None = None,
    mode: Literal["crop", "bbox"] = "crop",
):
    detection = detect_fn(img_path)
    if detection is None:
        return
    
    # Support pour ancien format (list) et nouveau format (dict)
    if isinstance(detection, dict):
        bbox = detection['bbox']
        label = detection.get('label', 'object')
        confidence = detection.get('confidence', 0.0)
    else:
        bbox = detection
        label = 'object'
        confidence = 0.0

    img = Image.open(img_path)
    img_corrected = ImageOps.exif_transpose(img)

    img_viz = img_corrected.copy()
    draw = ImageDraw.Draw(img_viz)
    draw.rectangle(bbox, outline="red", width=5)
    
    # Afficher le label et la confiance
    try:
        # Essayer de charger une police avec une taille appropriée
        font_size = max(20, int(min(img_viz.width, img_viz.height) * 0.03))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        # Utiliser la police par défaut si la fonte n'est pas disponible
        font = ImageFont.load_default()
    
    # Texte à afficher
    text = f"{label} ({confidence:.2f})"
    
    # Position du texte (au-dessus de la bbox)
    text_x = bbox[0]
    text_y = max(0, bbox[1] - font_size - 5)
    
    # Dessiner un fond pour le texte
    bbox_text = draw.textbbox((text_x, text_y), text, font=font)
    draw.rectangle(bbox_text, fill="red")
    draw.text((text_x, text_y), text, fill="white", font=font)

    det_output_dir.mkdir(exist_ok=True, parents=True)
    img_viz.save(det_output_dir / (img_path.stem + ".png"))

    if mode == "crop":
        cropped = img_corrected.crop(bbox)
        results_guided = segment_fn(cropped)
    else:
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_corrected.save(tmp, format="PNG")
            tmp_path = Path(tmp.name)
        
        try:
            results_guided = segment_fn(tmp_path, bbox)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    seg_output_dir.mkdir(exist_ok=True, parents=True)

    if txt_output_dir is not None:
        txt_output_dir.mkdir(exist_ok=True, parents=True)

    for result in results_guided:
        res_plotted = result.plot()
        img_sam = Image.fromarray(res_plotted[..., ::-1])
        img_sam.save(seg_output_dir / (img_path.stem + ".png"))

        if txt_output_dir is not None and result.masks is not None:
            mask_data = result.masks.data[0]
            ys, xs = (mask_data > 0.5).nonzero(as_tuple=True)
            if len(xs) == 0:
                continue

            h, w = mask_data.shape
            xs_norm = xs.float() / float(w)
            ys_norm = ys.float() / float(h)

            parts: list[str] = ["0"]
            for x_val, y_val in zip(xs_norm.tolist(), ys_norm.tolist()):
                parts.append(f"{x_val:.6f}")
                parts.append(f"{y_val:.6f}")

            txt_path = txt_output_dir / f"{img_path.stem}.txt"
            txt_path.write_text(" ".join(parts))


def img_pipeline_multi(
    img_path: Path,
    detect_all_fn: Callable[[Path], list[dict]],
    det_output_dir: Path = Path("detection_output"),
):
    """
    Pipeline pour afficher toutes les détections sur l'image
    
    Args:
        img_path: Chemin vers l'image
        detect_all_fn: Fonction de détection retournant une liste de dicts avec 'bbox', 'label', 'confidence'
        det_output_dir: Dossier de sortie pour les images avec détections
    """
    detections = detect_all_fn(img_path)
    if not detections:
        return

    img = Image.open(img_path)
    img_corrected = ImageOps.exif_transpose(img)

    img_viz = img_corrected.copy()
    draw = ImageDraw.Draw(img_viz)
    
    # Charger la police
    try:
        font_size = max(20, int(min(img_viz.width, img_viz.height) * 0.03))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
        font_size = 12
    
    # Dessiner toutes les détections
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"]
    
    for idx, detection in enumerate(detections):
        bbox = detection['bbox']
        label = detection.get('label', 'object')
        confidence = detection.get('confidence', 0.0)
        
        # Couleur différente pour chaque détection
        color = colors[idx % len(colors)]
        
        # Dessiner le rectangle
        draw.rectangle(bbox, outline=color, width=5)
        
        # Texte à afficher
        text = f"{label} ({confidence:.2f})"
        
        # Position du texte (au-dessus de la bbox)
        text_x = bbox[0]
        text_y = max(0, bbox[1] - font_size - 5)
        
        # Dessiner un fond pour le texte
        bbox_text = draw.textbbox((text_x, text_y), text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((text_x, text_y), text, fill="white", font=font)

    det_output_dir.mkdir(exist_ok=True, parents=True)
    img_viz.save(det_output_dir / (img_path.stem + ".jpg"))
