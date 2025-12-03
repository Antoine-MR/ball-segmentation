from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps


def img_pipeline(
    img_path: Path,
    detect_fn: Callable[[Path], list[float] | None],
    segment_fn: Callable[[Path, list[float]], Any],
    det_output_dir: Path = Path("detection_output"),
    seg_output_dir: Path = Path("sam_output"),
):
    bbox = detect_fn(img_path)
    if bbox is None:
        return

    img = Image.open(img_path)
    img_corrected = ImageOps.exif_transpose(img)

    draw = ImageDraw.Draw(img_corrected)
    draw.rectangle(bbox, outline="red", width=5)

    det_output_dir.mkdir(exist_ok=True, parents=True)
    img_corrected.save(det_output_dir / (img_path.stem + ".png"))

    # Save EXIF-corrected image for segmentation (bbox coords match corrected orientation)
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_no_draw = ImageOps.exif_transpose(Image.open(img_path))
        img_no_draw.save(tmp.name)
        tmp_path = Path(tmp.name)

    results_guided = segment_fn(tmp_path, bbox)
    tmp_path.unlink()

    seg_output_dir.mkdir(exist_ok=True, parents=True)

    for result in results_guided:
        res_plotted = result.plot()
        img_sam = Image.fromarray(res_plotted[..., ::-1])
        img_sam.save(seg_output_dir / (img_path.stem + ".png"))
