from PIL import Image

from ultralytics import SAM, FastSAM  # type: ignore


class SAMSegmenter:
    def __init__(self, model_name: str = "sam2.1_b.pt"):
        self.model = SAM(model_name)

    def segment(self, img: Image.Image):
        return self.model(img, verbose=False)


class FastSAMSegmenter:
    def __init__(self, model_name: str = "FastSAM-s.pt"):
        self.model = FastSAM(model_name)

    def segment(self, img: Image.Image):
        return self.model(img, verbose=False)
