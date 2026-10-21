import json
from pathlib import Path
import numpy as np
from .frames import read_rgb, rectify
from .tiles import tile_predict


class Detector:
    # wraps a YOLO checkpoint. backend='numpy' needs no PyTorch (about 6 s per straightened photo on a plain CPU).
    # backend='ultralytics' uses the real library if you have it installed. both give the same dots.
    def __init__(self, model_path, backend='numpy', imgsz=512):
        self.imgsz = imgsz
        self.backend = backend
        if backend == 'numpy':
            from .yolo_np import Model
            self.model = Model(str(model_path))
        elif backend == 'ultralytics':
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
        else:
            raise ValueError("backend is 'numpy' or 'ultralytics'")

    def predict_tiles(self, tiles):
        # tiles are BGR (opencv order), same as Ultralytics expects
        if self.backend == 'numpy':
            return self.model.predict(tiles, imgsz=self.imgsz, conf=.001, iou=.5, max_det=2000, batch=4)
        out = []
        for r in self.model.predict(tiles, imgsz=self.imgsz, conf=.001, iou=.5, max_det=2000,
                                    device='cpu', batch=4, verbose=False, save=False):
            if r.boxes is None or not len(r.boxes):
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
            else:
                out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        return out

    def dots(self, rgb, bounds=None):
        # all candidate dots (x, y, radius, confidence) on an already straightened image
        h, w = rgb.shape[:2]
        return tile_predict(self.predict_tiles, rgb, bounds or (0, 0, w, h))


def load_config(path):
    # default.json from training: which checkpoint + the threshold picked on the validation strip (0.37)
    return json.loads(Path(path).read_text())


def count_photo(detector, image_path, corners, threshold):
    # the whole pipeline for one photo: straighten the frame -> tiles -> model -> keep dots above threshold.
    rgb, _ = rectify(read_rgb(image_path), corners)
    d = detector.dots(rgb)
    return rgb, d[d[:, 3] >= threshold]
