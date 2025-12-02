"""
Pipeline générique de traitement d'images : Détection + Segmentation.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional

from PIL import Image, ImageDraw, ImageOps


def img_pipeline(
    img_path: Path,
    detect_fn: Callable[[Path], Optional[List[float]]],
    segment_fn: Callable[[Path, List[float]], Any],
    det_output_dir: Path = Path("detection_output"),
    seg_output_dir: Path = Path("sam_output"),
):
    """
    Pipeline générique de traitement d'une image.

    Args:
        img_path: Chemin de l'image
        detect_fn: Fonction (Path) -> [x_min, y_min, x_max, y_max] (ou None si rien détecté)
        segment_fn: Fonction (Path, bbox) -> Résultat segmentation (avec méthode .plot())
        det_output_dir: Dossier export détection
        seg_output_dir: Dossier export segmentation
    """

    # 1. Détection (via la fonction fournie)
    bbox = detect_fn(img_path)

    if bbox is None:
        return

    # --- Visualisation de la détection ---
    img_viz = Image.open(img_path)
    img_viz = ImageOps.exif_transpose(img_viz)  # Correction rotation EXIF

    draw = ImageDraw.Draw(img_viz)
    draw.rectangle(bbox, outline="red", width=5)

    # Export détection
    det_output_dir.mkdir(exist_ok=True, parents=True)
    output_path = det_output_dir / (img_path.stem + ".png")
    img_viz.save(output_path)

    # --- 2. Segmentation (via la fonction fournie) ---
    results_guided = segment_fn(img_path, bbox)

    seg_output_dir.mkdir(exist_ok=True, parents=True)

    for result in results_guided:
        res_plotted = result.plot()
        img_sam = Image.fromarray(res_plotted[..., ::-1])  # BGR -> RGB

        output_path_sam = seg_output_dir / (img_path.stem + ".png")
        img_sam.save(output_path_sam)
