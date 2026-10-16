import random
from typing import List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter

from .base import BaseTransform, register_transform, map_box_points


@register_transform
class RandomTranslate(BaseTransform):
    name = "trans"

    def __init__(self, max_dx: int = 20, max_dy: int = 20, rng=None):
        super().__init__(rng)
        self.max_dx = max(0, min(100, max_dx))
        self.max_dy = max(0, min(100, max_dy))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        dx = self.rng.randint(-self.max_dx, self.max_dx)
        dy = self.rng.randint(-self.max_dy, self.max_dy)
        result = QImage(image.width(), image.height(), QImage.Format_ARGB32)
        result.fill(Qt.black)
        painter = QPainter(result)
        painter.drawImage(dx, dy, image)
        painter.end()
        new_boxes = []
        w = image_width
        h = image_height
        for b in boxes:
            nb = map_box_points(b, lambda px, py: (px + dx, py + dy))
            if nb["width"] < 1 or nb["height"] < 1:
                continue
            if b.get("shape_type") == "point" or b.get("points"):
                if not (0 <= nb["x"] <= w and 0 <= nb["y"] <= h
                        and nb["x"] + nb["width"] >= 0
                        and nb["y"] + nb["height"] >= 0):
                    continue
            else:
                x1 = max(0, nb["x"])
                y1 = max(0, nb["y"])
                x2 = min(w, nb["x"] + nb["width"])
                y2 = min(h, nb["y"] + nb["height"])
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                nb["x"], nb["y"] = x1, y1
                nb["width"], nb["height"] = x2 - x1, y2 - y1
            new_boxes.append(nb)
        return result, new_boxes
