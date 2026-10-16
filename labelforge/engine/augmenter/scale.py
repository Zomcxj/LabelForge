import random
from typing import List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter

from .base import BaseTransform, register_transform, map_box_points


@register_transform
class RandomScale(BaseTransform):
    name = "scale"

    def __init__(self, min: float = 0.8, max: float = 1.2, rng=None):
        super().__init__(rng)
        self.min = min
        self.max = max

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        scale = self.rng.uniform(self.min, self.max)
        new_w = max(1, int(image_width * scale))
        new_h = max(1, int(image_height * scale))
        scaled = image.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if scale >= 1.0:
            result = scaled
        else:
            result = QImage(image_width, image_height, QImage.Format_ARGB32)
            result.fill(Qt.black)
            painter = QPainter(result)
            ox = (image_width - new_w) // 2
            oy = (image_height - new_h) // 2
            painter.drawImage(ox, oy, scaled)
            painter.end()
        new_boxes = []
        for b in boxes:
            ox = (image_width - new_w) // 2 if scale < 1.0 else 0
            oy = (image_height - new_h) // 2 if scale < 1.0 else 0
            new_boxes.append(map_box_points(
                b, lambda px, py: (px * scale + ox, py * scale + oy)))
        return result, new_boxes
