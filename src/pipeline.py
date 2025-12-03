from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps


def img_pipeline(
    img_path: Path,
    detect_fn: Callable[[Path], list[float] | None],
    segment_fn: Callable[[Path], Any],
    det_output_dir: Path = Path("detection_output"),
    seg_output_dir: Path = Path("sam_output"),
):
    bbox = detect_fn(img_path)
    if bbox is None:
        return

    img = Image.open(img_path)
    img_corrected = ImageOps.exif_transpose(img)

    img_viz = img_corrected.copy()
    draw = ImageDraw.Draw(img_viz)
    draw.rectangle(bbox, outline="red", width=5)

    det_output_dir.mkdir(exist_ok=True, parents=True)
    img_viz.save(det_output_dir / (img_path.stem + ".png"))

    cropped = img_corrected.crop(bbox)

    results_guided = segment_fn(cropped)

    seg_output_dir.mkdir(exist_ok=True, parents=True)

    for result in results_guided:
        res_plotted = result.plot()
        img_sam = Image.fromarray(res_plotted[..., ::-1])
        img_sam.save(seg_output_dir / (img_path.stem + ".png"))
