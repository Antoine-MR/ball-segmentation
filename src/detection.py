from pathlib import Path

from inference import get_model
from ultralytics import YOLO


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


class YOLODetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        """
        Détecteur basé sur YOLO avec support des prompts textuels
        
        Args:
            model_path: Chemin vers le modèle YOLO (.pt)
            conf: Seuil de confiance pour les détections
        """
        self.model = YOLO(model_path)
        self.conf = conf

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
    
    def detect_all(self, img_path: Path, prompt: str = None) -> list[dict]:
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
