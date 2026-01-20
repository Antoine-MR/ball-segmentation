from pathlib import Path

from PIL import Image
import numpy as np

from ultralytics import SAM, FastSAM  # type: ignore

from .config import config


class _BBoxSegmenter:
    model: SAM | FastSAM

    def segment_bbox(self, img: Path | Image.Image | np.ndarray, bbox: list[float]):
        """Segment the object inside bbox [x1, y1, x2, y2]."""
        img_input = str(img) if isinstance(img, Path) else img
        return self.model(img_input, bboxes=[bbox], verbose=False)


class SAMSegmenter(_BBoxSegmenter):
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = str(config.get_path('models.sam2_1.path'))
        self.model = SAM(model_name)

    def segment(self, img: Image.Image):
        cx, cy = img.width // 2, img.height // 2
        return self.model(img, points=[[cx, cy]], labels=[1], verbose=False)


class FastSAMSegmenter(_BBoxSegmenter):
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = str(config.get_path('models.fastsam.path'))
        self.model = FastSAM(model_name)

    def segment(self, img: Image.Image):
        """Keep only the largest mask."""
        results = self.model(img, verbose=False)
        if not results or results[0].masks is None:
            return results

        masks = results[0].masks.data
        if len(masks) == 0:
            return results

        largest_idx = masks.sum(dim=(1, 2)).argmax().item()
        results[0].masks.data = masks[largest_idx:largest_idx + 1]
        if results[0].boxes is not None:
            results[0].boxes.data = results[0].boxes.data[largest_idx:largest_idx + 1]

        return results
