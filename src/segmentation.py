"""
Module de segmentation d'images.
Contient les adapters pour différents modèles de segmentation.
"""

from pathlib import Path
from typing import List

from ultralytics import SAM

# Chemin vers le dossier des modèles
MODELS_DIR = Path(__file__).parent.parent / "models"


class SAMSegmenter:
    """Wrapper pour les modèles SAM (Segment Anything Model)."""

    def __init__(self, model_name: str = "sam2.1_b.pt"):
        """
        Initialise le segmenteur SAM.

        Args:
            model_name: Nom du fichier modèle (stocké dans models/)
        """
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé : {model_path}\n"
                f"Placez le fichier {model_name} dans le dossier 'models/'"
            )
        self.model = SAM(str(model_path))

    def segment(self, img_path: Path, bbox: List[float]):
        """
        Segmente une région de l'image définie par une bounding box.

        Args:
            img_path: Chemin vers l'image
            bbox: [x_min, y_min, x_max, y_max]

        Returns:
            Résultats de segmentation (objets avec méthode .plot())
        """
        return self.model(str(img_path), bboxes=[bbox], verbose=False)


# Vous pouvez ajouter d'autres segmenteurs ici, par exemple :
# class FastSAMSegmenter:
#     def __init__(self, model_name: str = "FastSAM-s.pt"):
#         ...
