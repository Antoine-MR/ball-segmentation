from pathlib import Path
import os
import time
from typing import overload, Literal
from inference import get_model
from ultralytics.models import YOLO
from PIL import Image, ImageOps
import numpy as np
from groundingdino.util.inference import load_model
from groundingdino.util.utils import get_phrases_from_posmap
import groundingdino.datasets.transforms as T
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
import torch

from .config import config


def predict_custom(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
):
    """groundingdino.util.inference.predict, plus remove_combined.

    With several prompts ("red ball . human"), the stock version can return a
    merged phrase like "red ball human". remove_combined keeps only the
    prompt segment with the highest score for each box.
    """
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    if not remove_combined:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
        return boxes, logits.max(dim=1)[0], phrases

    phrases = []
    confidences = []
    dot_token_id = tokenizer(".").input_ids[1]
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    input_ids = tokenized["input_ids"]

    for logit in logits:
        segments = []  # (max_score, token_indices) per prompt
        current_indices = []
        for i, token_id in enumerate(input_ids):
            if token_id == dot_token_id:
                if current_indices:
                    segments.append((logit[current_indices].max().item(), current_indices))
                    current_indices = []
            elif token_id not in special_tokens:
                current_indices.append(i)
        if current_indices:
            segments.append((logit[current_indices].max().item(), current_indices))

        valid_segments = [s for s in segments if s[0] > text_threshold]
        if valid_segments:
            best_score, best_indices = max(valid_segments, key=lambda x: x[0])
            posmap = torch.zeros_like(logit, dtype=torch.bool)
            posmap[best_indices] = True
            phrases.append(get_phrases_from_posmap(posmap, tokenized, tokenizer).replace('.', ''))
            confidences.append(best_score)
        else:
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', ''))
            confidences.append(logit.max().item())

    return boxes, torch.tensor(confidences), phrases


class RoboflowDetector:
    DEFAULT_MODEL_ID = "red-ball-detection-new/1"

    def __init__(self, model_id: str | None = None, api_key: str | None = None):
        if model_id is None:
            model_id = self.DEFAULT_MODEL_ID

        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError(
                    "ROBOFLOW_API_KEY not found. "
                    "Set it as environment variable or pass it explicitly."
                )

        self.model = get_model(model_id=model_id, api_key=api_key)

    def detect(self, img_path: Path) -> dict | None:
        results = self.model.infer(str(img_path))
        predictions = results[0].predictions

        if not predictions:
            return None

        det = predictions[0]
        bbox = [
            det.x - det.width / 2,
            det.y - det.height / 2,
            det.x + det.width / 2,
            det.y + det.height / 2,
        ]

        return {
            'bbox': bbox,
            'label': det.class_name if hasattr(det, 'class_name') else 'ball',
            'confidence': det.confidence if hasattr(det, 'confidence') else 0.0
        }


class YOLODetector:
    def __init__(self, model_path: str | None = None, conf: float | None = None):
        if model_path is None:
            model_path = str(config.get_path('models.yolo_nano.path'))
        if conf is None:
            conf = config.get('inference.confidence', 0.25)

        self.model = YOLO(model_path)
        self.conf: float | None = conf

    def detect(self, img_path: Path) -> dict | None:
        """Return the first detection as {'bbox': [x1, y1, x2, y2], 'label', 'confidence'}, or None."""
        results = self.model(str(img_path), conf=self.conf, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        box = results[0].boxes[0]
        coords = box.xyxy[0].cpu().numpy()

        return {
            'bbox': [float(c) for c in coords[:4]],
            'label': self.model.names[int(box.cls[0])],
            'confidence': float(box.conf[0])
        }


class GroundingDINODetector:
    def __init__(self, model_config_path: str | None = None,
                 model_checkpoint_path: str | None = None,
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.2,
                 device: str = "cuda"):
        if model_config_path is None:
            import groundingdino
            package_dir = Path(groundingdino.__file__).parent
            model_config_path = str(package_dir / "config" / "GroundingDINO_SwinT_OGC.py")

        if model_checkpoint_path is None:
            model_checkpoint_path = "models/pretrained/groundingdino_swint_ogc.pth"
            if not Path(model_checkpoint_path).exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at {model_checkpoint_path}. "
                    "Download from: https://github.com/IDEA-Research/GroundingDINO/releases"
                )

        self.model: GroundingDINO = load_model(model_config_path, model_checkpoint_path, device=device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    @overload
    def detect(self, img_path: Path, text_prompt: str, return_all_by_label: Literal[False] = False, debug: bool = False) -> dict | None: ...

    @overload
    def detect(self, img_path: Path, text_prompt: str, return_all_by_label: Literal[True] = True, debug: bool = False) -> dict[str, list[dict]]: ...

    def detect(self, img_path: Path, text_prompt: str, return_all_by_label: bool = False, debug: bool = False) -> dict | None | dict[str, list[dict]]:
        """Detect objects matching text_prompt.

        Several prompts are separated by ' . ' ("red ball . human"). With
        return_all_by_label, returns {prompt: [detections]}; otherwise the
        best detection or None.
        """
        if return_all_by_label:
            detections = self._detect_single_prompt(img_path, text_prompt, return_all=True, debug=debug)

            prompts = [p.strip().lower() for p in text_prompt.split('.') if p.strip()]
            result: dict[str, list[dict]] = {p: [] for p in prompts}
            for det in detections:
                if det['label'] in result:
                    result[det['label']].append(det)
            return result

        detections = self._detect_single_prompt(img_path, text_prompt, return_all=False, debug=debug)
        return detections[0] if detections else None

    def _detect_single_prompt(self, img_path: Path, text_prompt: str, return_all: bool = False, debug: bool = False) -> list[dict]:
        start_time = time.time()

        # Same EXIF handling as img_pipeline, otherwise boxes don't match the saved image
        pil_image = ImageOps.exif_transpose(Image.open(img_path))
        image_source = np.array(pil_image)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(pil_image, None)

        if debug:
            print(f"  Image loaded: {image_source.shape}, tensor: {image.shape}")

        clean_prompt = text_prompt.strip().lower()
        if not clean_prompt.endswith('.'):
            clean_prompt = clean_prompt + '.'

        if debug:
            print(f"  Prompt: '{clean_prompt}'")
            print(f"  Thresholds: box={self.box_threshold}, text={self.text_threshold}")

        boxes, logits, phrases = predict_custom(
            model=self.model,
            image=image,
            caption=clean_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
            remove_combined=True
        )

        if debug:
            print(f"  Inference time: {time.time() - start_time:.2f}s")
            print(f"  Detections found: {len(boxes)}")
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                print(f"    [{i}] {phrase}: confidence={logit:.3f}, box={box}")

        h, w = image_source.shape[:2]
        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            # Normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
            cx, cy, bw, bh = box
            detections.append({
                'bbox': [float((cx - bw / 2) * w), float((cy - bh / 2) * h),
                         float((cx + bw / 2) * w), float((cy + bh / 2) * h)],
                'label': phrase,
                'confidence': float(logit)
            })

        if not return_all:
            return detections[:1]
        return detections
