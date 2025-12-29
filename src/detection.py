from pathlib import Path
import os
from typing import overload, Literal
from inference import get_model
from ultralytics.models import YOLO
from PIL import Image, ImageOps
import numpy as np
from groundingdino.util.inference import load_model, predict
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
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        phrases = []
        confidences = []
        
        # Get dot token ID (assuming it's the second token in " . ")
        # Or just use the one from tokenizer(".")
        dot_token_id = tokenizer(".").input_ids[1]
        
        input_ids = tokenized["input_ids"]
        
        for logit in logits:
            # Split tokens by '.' and pick the segment with highest max score
            segments = [] # List of (max_score, segment_indices)
            current_indices = []
            
            for i, token_id in enumerate(input_ids):
                if token_id == dot_token_id:
                    if current_indices:
                        # End of segment
                        seg_scores = logit[current_indices]
                        segments.append((seg_scores.max().item(), current_indices))
                        current_indices = []
                elif token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    current_indices.append(i)
            
            # Handle last segment
            if current_indices:
                 seg_scores = logit[current_indices]
                 segments.append((seg_scores.max().item(), current_indices))
            
            # Filter segments > text_threshold
            valid_segments = [s for s in segments if s[0] > text_threshold]
            
            if valid_segments:
                # Pick the best one (highest probability retained)
                best_score, best_indices = max(valid_segments, key=lambda x: x[0])
                
                # Construct phrase
                posmap = torch.zeros_like(logit, dtype=torch.bool)
                posmap[best_indices] = True
                
                phrase = get_phrases_from_posmap(posmap, tokenized, tokenizer).replace('.', '')
                phrases.append(phrase)
                confidences.append(best_score)
            else:
                # Fallback: use original method if no segment passes (shouldn't happen if box passed)
                phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                phrases.append(phrase)
                confidences.append(logit.max().item())
                
        return boxes, torch.tensor(confidences), phrases

    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]
        return boxes, logits.max(dim=1)[0], phrases


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
        """
        Détecte les objets dans l'image
        
        Args:
            img_path: Chemin vers l'image
            prompt: Prompt textuel optionnel pour filtrer les classes (ex: "ball", "person")
        
        Returns:
            Dictionnaire avec 'bbox' [x1, y1, x2, y2], 'label' et 'confidence' ou None
        """
        # Effectuer la détection
        results = self.model(str(img_path), conf=self.conf, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Si un prompt est fourni, filtrer par classe
        # if prompt:
        #     prompt_lower = prompt.lower()
        #     for box in results[0].boxes:
        #         class_id = int(box.cls[0])
        #         class_name = self.model.names[class_id].lower()
                
        #         if prompt_lower in class_name or class_name in prompt_lower:
        #             coords = box.xyxy[0].cpu().numpy()
        #             return [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
        #     return None
        
        # Sinon, retourner la première détection
        box = results[0].boxes[0]
        coords = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = self.model.names[class_id]
        
        return {
            'bbox': [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])],
            'label': label,
            'confidence': confidence
        }


class GroundingDINODetector:
    """Détecteur basé sur Grounding DINO avec prompt textuel"""
    
    def __init__(self, model_config_path: str | None = None, 
                 model_checkpoint_path: str | None = None,
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.2,
                 device: str = "cuda"):

        
        # Chemins par défaut pour Grounding DINO
        if model_config_path is None:
            # Essayer le chemin dans le package groundingdino-py
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
        self.conf = None
    
    @overload
    def detect(self, img_path: Path, text_prompt: str, return_all_by_label: Literal[False] = False, debug: bool = False) -> dict | None: ...
    
    @overload
    def detect(self, img_path: Path, text_prompt: str, return_all_by_label: Literal[True] = True, debug: bool = False) -> dict[str, list[dict]]: ...
    
    def detect(self, img_path: Path, text_prompt: str, return_all_by_label: bool = False, debug: bool = False) -> dict | None | dict[str, list[dict]]:
        """
        Détecte les objets dans l'image en fonction du prompt textuel
        
        Args:
            img_path: Chemin vers l'image
            text_prompt: Prompt textuel (ex: "red ball") ou prompts multiples séparés par ' . ' (ex: "red ball . human")
            return_all_by_label: Si True, retourne dict[label, list[detections]] en parsant les prompts séparés par ' . '
            debug: Si True, affiche des informations de debugging
        
        Returns:
            - Si return_all_by_label=False: dict avec 'bbox', 'label', 'confidence' ou None (meilleure détection)
            - Si return_all_by_label=True: dict[str, list[dict]] groupé par label
        """
        # Mode multi-label: parser les prompts séparés par ' . '
        if return_all_by_label:
            # Optimisation: on lance une seule détection avec tous les prompts
            # Le filtrage des "combined" est géré dans _detect_single_prompt
            detections = self._detect_single_prompt(img_path, text_prompt, return_all=True, debug=debug)
            
            # Grouper par label
            prompts = [p.strip().lower() for p in text_prompt.split('.') if p.strip()]
            result: dict[str, list[dict]] = {p: [] for p in prompts}
            
            for det in detections:
                label = det['label']
                if label in result:
                    result[label].append(det)
            
            return result
        
        # Mode single detection (comportement original)
        detections = self._detect_single_prompt(img_path, text_prompt, return_all=False, debug=debug)
        return detections[0] if detections else None
    
    def _detect_single_prompt(self, img_path: Path, text_prompt: str, return_all: bool = False, debug: bool = False) -> list[dict]:
        """
        Détecte les objets pour un seul prompt
        
        Returns:
            Liste de détections (vide si aucune détection)
        """
        import time
        start_time = time.time()
        
        # Charger l'image avec correction EXIF (comme dans img_pipeline)
        pil_image = Image.open(img_path)
        pil_image = ImageOps.exif_transpose(pil_image)  # Correction rotation EXIF
        image_source = np.array(pil_image)
        
        # Transformer pour Grounding DINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_image, None)
        image = image_transformed
        
        if debug:
            print(f"  Image loaded: {image_source.shape}, tensor: {image.shape}")
        
        # Nettoyer le prompt (ajouter un point final comme dans la démo)
        clean_prompt = text_prompt.strip().lower()
        if not clean_prompt.endswith('.'):
            clean_prompt = clean_prompt + '.'
        
        if debug:
            print(f"  Prompt: '{clean_prompt}'")
            print(f"  Thresholds: box={self.box_threshold}, text={self.text_threshold}")
        
        # Prédiction avec notre fonction custom qui gère remove_combined
        boxes, logits, phrases = predict_custom(
            model=self.model,
            image=image,
            caption=clean_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
            remove_combined=True
        )

        # Plus besoin de filtrage manuel ici car predict_custom le fait mieux
        
        elapsed = time.time() - start_time
        
        if debug:
            print(f"  Inference time: {elapsed:.2f}s")
            print(f"  Detections found: {len(boxes)}")
            if len(boxes) > 0:
                for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                    print(f"    [{i}] {phrase}: confidence={logit:.3f}, box={box}")
        
        if len(boxes) == 0:
            return []
        
        # Convertir toutes les détections
        h, w = image_source.shape[:2]
        detections = []
        
        for box, logit, phrase in zip(boxes, logits, phrases):
            # Convertir de format [cx, cy, w, h] normalisé vers [x1, y1, x2, y2] pixels
            cx, cy, bw, bh = box
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'label': phrase,
                'confidence': float(logit)
            })
        
        # Si return_all=False, retourner seulement la meilleure
        if not return_all and len(detections) > 0:
            return [detections[0]]
        
        return detections
    
    def detect_all(self, img_path: Path, prompt: str|None  = None) -> list[dict]:
        """
        Détecte tous les objets dans l'image
        
        Args:
            img_path: Chemin vers l'image
            prompt: Prompt textuel optionnel pour filtrer les classes
        
        Returns:
            Liste de dictionnaires avec 'bbox', 'label' et 'confidence'
        """
        results = self.model(str(img_path), conf=self.conf, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
        
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            
            # Filtrer par prompt si fourni
            if prompt:
                prompt_lower = prompt.lower()
                
                if not (prompt_lower in class_name.lower() or class_name.lower() in prompt_lower):
                    continue
            
            coords = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            
            detections.append({
                'bbox': [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])],
                'label': class_name,
                'confidence': confidence
            })
        
        return detections
