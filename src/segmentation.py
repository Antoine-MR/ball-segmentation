from pathlib import Path

from PIL import Image

from ultralytics import SAM, FastSAM  # type: ignore

from .config import config


class SAMSegmenter:
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = str(config.get_path('models.sam2_1.path'))
        self.model = SAM(model_name)

    def segment(self, img: Image.Image):
        cx, cy = img.width // 2, img.height // 2
        return self.model(img, points=[[cx, cy]], labels=[1], verbose=False)

    def segment_bbox(self, img_path: Path, bbox: list[float]):
        return self.model(str(img_path), bboxes=[bbox], verbose=False)


class FastSAMSegmenter:
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = str(config.get_path('models.fastsam.path'))
        self.model = FastSAM(model_name)

    def segment(self, img: Image.Image):
        results = self.model(img, verbose=False)
        if not results or results[0].masks is None:
            return results
        
        masks = results[0].masks.data
        if len(masks) == 0:
            return results
        
        areas = masks.sum(dim=(1, 2))
        largest_idx = areas.argmax().item()
        
        results[0].masks.data = masks[largest_idx:largest_idx+1]
        if results[0].boxes is not None:
            results[0].boxes.data = results[0].boxes.data[largest_idx:largest_idx+1]
        
        return results

    def segment_bbox(self, img_path: Path, bbox: list[float]):
        return self.model(str(img_path), bboxes=[bbox], verbose=False)
