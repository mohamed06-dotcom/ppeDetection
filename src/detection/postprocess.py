import cv2
import numpy as np
from PIL import Image
from .inferance import YOLOInference


class Postprocess:
    def __init__(self, class_names: dict):
        self.class_names = class_names

    @staticmethod
    def _center_inside(person_box, item_box) -> bool:
        px1, py1, px2, py2 = person_box
        ix1, iy1, ix2, iy2 = item_box
        cx = (ix1 + ix2) / 2
        cy = (iy1 + iy2) / 2
        return px1 <= cx <= px2 and py1 <= cy <= py2

    def check_ppe_per_person(self, results) -> list:
        if not results or len(results) == 0:
            return []

        boxes     = results[0].boxes
        persons   = []
        ppe_items = []

        for box in boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, "").lower()
            coords   = box.xyxy[0].tolist()   # [x1, y1, x2, y2]

            if "person" in cls_name:
                persons.append({"box": coords, "cls": cls_name})
            else:
                ppe_items.append({"box": coords, "cls": cls_name})

        if not persons:
            status = self.check_ppe(results)
            return [{
                "person_id":  0,
                "person_box": None,
                "has_helmet": status["has_helmet"],
                "has_vest":   status["has_vest"],
                "has_boots":  status["has_boots"],
                "missing":    status["missing"],
                "compliant":  status["compliant"],
            }]

        report = []
        for idx, person in enumerate(persons):
            has_helmet = False
            has_vest   = False
            has_boots  = False

            for item in ppe_items:
                # Skip if PPE center is NOT inside this person's box
                if not self._center_inside(person["box"], item["box"]):
                    continue

                name = item["cls"]
                if any(k in name for k in ("helmet", "hardhat", "hat")):
                    has_helmet = True
                if any(k in name for k in ("vest", "jacket", "safety-vest")):
                    has_vest = True
                if any(k in name for k in ("boot", "shoe", "footwear")):
                    has_boots = True

            missing = []
            if not has_helmet: missing.append("Helmet")
            if not has_vest:   missing.append("Vest")
            if not has_boots:  missing.append("Boots")

            report.append({
                "person_id":  idx,
                "person_box": person["box"],
                "has_helmet": has_helmet,
                "has_vest":   has_vest,
                "has_boots":  has_boots,
                "missing":    missing,
                "compliant":  len(missing) == 0,
            })

        return report

    def check_ppe(self, results) -> dict:
        has_helmet = False
        has_boots  = False
        has_vest   = False

        if len(results) > 0:
            for box in results[0].boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "").lower()

                if any(k in cls_name for k in ("helmet", "hardhat", "hat")):
                    has_helmet = True
                if any(k in cls_name for k in ("vest", "jacket", "safety-vest")):
                    has_vest = True
                if any(k in cls_name for k in ("boot", "shoe", "footwear")):
                    has_boots = True

        missing = []
        if not has_helmet: missing.append("Helmet")
        if not has_vest:   missing.append("Vest")
        if not has_boots:  missing.append("Boots")

        return {
            "has_helmet": has_helmet,
            "has_vest":   has_vest,
            "has_boots":  has_boots,
            "missing":    missing,
            "compliant":  len(missing) == 0,
        }