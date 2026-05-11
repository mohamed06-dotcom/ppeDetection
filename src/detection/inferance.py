import cv2
import numpy as np
from PIL import Image
from ._load_model  import _load_model


class YOLOInference:
    def __init__(self,confidence:float=0.25):
        self.conf=confidence
        self.model=_load_model()

    def set_confidence(self,confidence:float):
        self.conf=confidence

    @property
    def class_names(self) -> dict:
        return self.model.names

    def run_inference(self,source:Image):
        return self.model(source,conf=self.conf,verbose=False)

    def run_inference_annotated(self,source:Image):
        results = self.run_inference(source)
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return results, annotated_rgb
