from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Literal

from PIL import Image, ImageDraw, ImageOps


def img_pipeline(
    img_path: Path,
    detect_fn: Callable[[Path], list[float] | None],
    segment_fn: Callable[..., Any],
    det_output_dir: Path = Path("detection_output"),
    seg_output_dir: Path = Path("sam_output"),
    txt_output_dir: Path | None = None,
    mode: Literal["crop", "bbox"] = "crop",
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
