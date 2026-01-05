"""
Visualization utilities for YOLO segmentation training monitoring.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import cv2
from ultralytics import YOLO
import numpy as np

def create_monitoring_callback(model, monitor_images, output_dir, project_name):
    """Create a callback to monitor segmentation progress for all classes during training"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch"""
        if len(monitor_images) == 0:
            return
        
        epoch = trainer.epoch
        
        # Skip first few epochs to save time
        if epoch < 5:
            return
        
        # Run predictions on monitor images
        print(f"\n{'='*60}")
        print(f"Generating monitoring visualizations for epoch {epoch}...")
        print(f"{'='*60}")
        
        # Save current training state
        was_training = trainer.model.training
        
        try:
            # Put model in eval mode and disable gradients
            trainer.model.eval()
            
            # Create figure with subplots
            n_images = len(monitor_images)
            n_cols = min(4, n_images)
            n_rows = (n_images + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
            gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)
            
            with torch.no_grad():
                for idx, img_path in enumerate(monitor_images):
                    try:
                        # Use the YOLO wrapper's predict method
                        # Create temporary YOLO instance with current weights
                        temp_model = YOLO(trainer.best if hasattr(trainer, 'best') else trainer.last)
                        results = temp_model.predict(img_path, conf=0.25, verbose=False)
                        
                        # Get the plotted image
                        result = results[0]
                        plotted_img = result.plot()
                        
                        # Convert BGR to RGB
                        plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
                        
                        # Add to subplot
                        row = idx // n_cols
                        col = idx % n_cols
                        ax = fig.add_subplot(gs[row, col])
                        ax.imshow(plotted_img)
                        ax.axis('off')
                        
                        # Count detections by class
                        if result.masks is not None and len(result.boxes) > 0:
                            classes_detected = result.boxes.cls.cpu().numpy()
                            n_balls = int((classes_detected == 0).sum())
                            n_humans = int((classes_detected == 1).sum())
                            n_trashcans = int((classes_detected == 2).sum())
                            
                            title = f"{Path(img_path).name}\n"
                            
                            # Build title without emojis (font compatibility)
                            parts = []
                            if n_balls > 0:
                                parts.append(f"Balls: {n_balls}")
                            if n_humans > 0:
                                parts.append(f"Humans: {n_humans}")
                            if n_trashcans > 0:
                                parts.append(f"Trashcans: {n_trashcans}")
                            
                            if parts:
                                title += " | ".join(parts)
                            else:
                                title += "No detections"
                        else:
                            title = f"{Path(img_path).name}\nNo detections"
                        
                        ax.set_title(title, fontsize=9)
                        
                    except Exception as e:
                        print(f"  ⚠️  Error processing {Path(img_path).name}: {e}")
            
            # Add main title
            fig.suptitle(f"Multi-Class Segmentation Monitoring - Epoch {epoch} - {project_name}", 
                         fontsize=14, fontweight='bold', y=0.998)
            
            # Save figure
            output_file = output_dir / f"epoch_{epoch:03d}.jpg"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✓ Saved monitoring visualization: {output_file.name}")
            print(f"{'='*60}\n")
            
        finally:
            # Restore training mode
            if was_training:
                trainer.model.train()
    
    return on_train_epoch_end
