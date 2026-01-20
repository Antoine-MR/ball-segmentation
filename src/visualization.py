from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import cv2
from ultralytics import YOLO


CLASS_NAMES = {0: "Balls", 1: "Humans", 2: "Trashcans"}


def create_monitoring_callback(model, monitor_images, output_dir, project_name):
    """Return an on_train_epoch_end callback saving a prediction grid of monitor_images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(trainer):
        if len(monitor_images) == 0:
            return

        epoch = trainer.epoch
        # Early epochs predict nothing useful
        if epoch < 5:
            return

        was_training = trainer.model.training

        try:
            trainer.model.eval()

            n_images = len(monitor_images)
            n_cols = min(4, n_images)
            n_rows = (n_images + n_cols - 1) // n_cols

            fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
            gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

            with torch.no_grad():
                for idx, img_path in enumerate(monitor_images):
                    try:
                        temp_model = YOLO(trainer.best if hasattr(trainer, 'best') else trainer.last)
                        result = temp_model.predict(img_path, conf=0.25, verbose=False)[0]
                        plotted_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

                        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
                        ax.imshow(plotted_img)
                        ax.axis('off')

                        parts = []
                        if result.masks is not None and len(result.boxes) > 0:
                            classes_detected = result.boxes.cls.cpu().numpy()
                            for class_id, name in CLASS_NAMES.items():
                                n = int((classes_detected == class_id).sum())
                                if n > 0:
                                    parts.append(f"{name}: {n}")

                        summary = " | ".join(parts) if parts else "No detections"
                        ax.set_title(f"{Path(img_path).name}\n{summary}", fontsize=9)
                    except Exception as e:
                        print(f"Monitoring failed on {Path(img_path).name}: {e}")

            fig.suptitle(f"Epoch {epoch} - {project_name}", fontsize=14, fontweight='bold', y=0.998)

            output_file = output_dir / f"epoch_{epoch:03d}.jpg"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {output_file}")
        finally:
            if was_training:
                trainer.model.train()

    return on_train_epoch_end
