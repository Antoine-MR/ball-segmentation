#!/usr/bin/env python3
"""
Script de conversion pour mettre les datasets irl_balls et irl_persons au nouveau format.

Ancien format:
datasets/preprocessed/{dataset}/
├── detection/
├── segmentation/
└── labels/
    └── *.txt

Nouveau format:
datasets/preprocessed/{dataset}/
├── detection/
├── segmentation/
├── empty/
└── ready/
    ├── images/
    └── labels/
        └── {class_name}/
            └── *.txt

Usage:
    python convert_to_new_format.py
"""

from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm


def convert_dataset(dataset_name: str, class_name: str, raw_images_path: Path):
    """
    Convertit un dataset au nouveau format
    
    Args:
        dataset_name: Nom du dataset (ex: 'irl_balls')
        class_name: Nom de la classe (ex: 'red ball')
        raw_images_path: Chemin vers les images originales
    """
    base = Path("datasets/preprocessed")
    dataset_path = base / dataset_name
    
    if not dataset_path.exists():
        print(f"❌ Dataset {dataset_name} n'existe pas")
        return
    
    print(f"\n🔄 Conversion de {dataset_name} (classe: {class_name})")
    
    # Créer la nouvelle structure
    ready_path = dataset_path / "ready"
    images_path = ready_path / "images"
    labels_path = ready_path / "labels"
    class_labels_path = labels_path / class_name
    empty_path = dataset_path / "empty"
    
    ready_path.mkdir(exist_ok=True)
    images_path.mkdir(exist_ok=True)
    labels_path.mkdir(exist_ok=True)
    class_labels_path.mkdir(exist_ok=True)
    empty_path.mkdir(exist_ok=True)
    
    # Ancien chemin des labels
    old_labels_path = dataset_path / "labels"
    
    # Fallback: vérifier txt_output_folder si le dossier labels est vide ou n'existe pas
    txt_output_base = Path("txt_output_folder") / dataset_name
    
    if not old_labels_path.exists() or not list(old_labels_path.glob("*.txt")):
        if txt_output_base.exists() and list(txt_output_base.glob("*.txt")):
            print(f"⚠️  Labels non trouvés dans {old_labels_path}, utilisation de {txt_output_base}")
            old_labels_path = txt_output_base
        else:
            print(f"⚠️  Pas de dossier labels/ trouvé pour {dataset_name}, skip...")
            return
    
    # Lister tous les fichiers txt
    txt_files = list(old_labels_path.glob("*.txt"))
    print(f"📁 {len(txt_files)} fichiers labels trouvés")
    
    # Copier les labels dans le sous-dossier de classe
    for txt_file in tqdm(txt_files, desc="Copie labels"):
        dest_txt = class_labels_path / txt_file.name
        if not dest_txt.exists():
            shutil.copy(txt_file, dest_txt)
    
    # Copier les images correspondantes
    if raw_images_path.exists():
        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
        copied_count = 0
        
        for txt_file in tqdm(txt_files, desc="Copie images"):
            stem = txt_file.stem
            img_file = None
            
            # Chercher l'image correspondante
            for ext in image_extensions:
                candidate = raw_images_path / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file:
                dest_img = images_path / img_file.name
                if not dest_img.exists():
                    shutil.copy(img_file, dest_img)
                    copied_count += 1
        
        print(f"✅ {copied_count} images copiées")
    else:
        print(f"⚠️  Dossier images {raw_images_path} non trouvé, skip images...")
    
    print(f"✅ Conversion terminée pour {dataset_name}")
    print(f"   - Images: {images_path}")
    print(f"   - Labels: {class_labels_path}")


def main():
    print("=" * 60)
    print("🔧 CONVERSION AU NOUVEAU FORMAT")
    print("=" * 60)
    
    # Définir les datasets à convertir
    conversions = [
        {
            "dataset_name": "irl_balls",
            "class_name": "red ball",
            "raw_images_path": Path("datasets/raw/IRL_validation_pictures")
        },
        {
            "dataset_name": "irl_persons",
            "class_name": "human",
            "raw_images_path": Path("datasets/raw/IRL_validation_pictures")
        }
    ]
    
    for config in conversions:
        convert_dataset(**config)
    
    print("\n" + "=" * 60)
    print("✅ CONVERSION TERMINÉE")
    print("=" * 60)
    print("\nStructure finale:")
    print("datasets/preprocessed/")
    print("├── irl_balls/")
    print("│   └── ready/")
    print("│       ├── images/")
    print("│       └── labels/")
    print("│           └── red ball/")
    print("└── irl_persons/")
    print("    └── ready/")
    print("        ├── images/")
    print("        └── labels/")
    print("            └── human/")


if __name__ == "__main__":
    main()
