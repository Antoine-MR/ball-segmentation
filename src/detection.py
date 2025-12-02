"""
Module de détection d'objets.
Contient les adapters pour différents modèles de détection.
"""

from pathlib import Path
from typing import List, Optional

from inference import get_model


class RoboflowDetector:
    """Wrapper pour les modèles de détection Roboflow."""

    def __init__(self, model_id: str, api_key: str):
        """
        Initialise le détecteur Roboflow.

        Args:
            model_id: ID du modèle sur Roboflow (ex: "raspberrypi_redball/2")
            api_key: Clé API Roboflow
        """
        self.model = get_model(model_id=model_id, api_key=api_key)

    def detect(self, img_path: Path) -> Optional[List[float]]:
        """
        Détecte un objet et retourne sa bounding box.

        Args:
            img_path: Chemin vers l'image

        Returns:
            [x_min, y_min, x_max, y_max] ou None si rien détecté
        """
        results = self.model.infer(str(img_path))
        predictions = results[0].predictions

        if not predictions:
            return None

        # On prend la détection avec la plus haute confiance (première)
        det = predictions[0]

        # Conversion centre (x, y, w, h) -> coins (min, max)
        return [
            det.x - det.width / 2,
            det.y - det.height / 2,
            det.x + det.width / 2,
            det.y + det.height / 2,
        ]
