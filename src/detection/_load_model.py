import os
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\moham\PycharmProjects\ppeDetection\src\model\best-70ep.pt"

def _load_model(model_path:str=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model not found in this current path : {model_path}"
            "Please verify the model path"
        )

    return YOLO(model_path)