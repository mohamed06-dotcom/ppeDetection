import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best-70ep.pt")

def _load_model(model_path:str=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model not found in this current path : {model_path}"
            "Please verify the model path"
        )

    return YOLO(model_path)