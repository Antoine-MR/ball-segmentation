from pathlib import Path

from ultralytics import SAM

MODELS_DIR = Path(__file__).parent.parent / "models"


class SAMSegmenter:
    def __init__(self, model_name: str = "sam2.1_b.pt"):
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = SAM(str(model_path))

    def segment(self, img_path: Path, bbox: list[float]):
        return self.model(str(img_path), bboxes=[bbox], verbose=False)
