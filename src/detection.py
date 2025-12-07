from pathlib import Path

from inference import get_model


class RoboflowDetector:
    def __init__(self, model_id: str, api_key: str):
        self.model = get_model(model_id=model_id, api_key=api_key)

    def detect(self, img_path: Path) -> list[float] | None:
        results = self.model.infer(str(img_path))
        predictions = results[0].predictions

        if not predictions:
            return None

        det = predictions[0]
        return [
            det.x - det.width / 2,
            det.y - det.height / 2,
            det.x + det.width / 2,
            det.y + det.height / 2,
        ]
