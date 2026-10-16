from typing import List, Tuple

from PyQt5.QtGui import QImage

from .base import BaseTransform, register_transform


@register_transform
class HorizontalFlip(BaseTransform):
    name = "fliph"

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        flipped = image.mirrored(True, False)
        new_boxes = []
        for b in boxes:
            nb = _flip_box_points(b, image_width, image_height, flip_x=True)
            new_boxes.append(nb)
        return flipped, new_boxes


@register_transform
class VerticalFlip(BaseTransform):
    name = "flipv"

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        flipped = image.mirrored(False, True)
        new_boxes = []
        for b in boxes:
            nb = _flip_box_points(b, image_width, image_height, flip_x=False)
            new_boxes.append(nb)
        return flipped, new_boxes


def _flip_box_points(b, image_width, image_height, flip_x):
    """翻转框：保留 shape_type/points/group_id（OBB/多边形/关键点）。"""
    new_b = dict(b)
    if b.get("shape_type") == "point" and b.get("points"):
        px, py = b["points"][0]
        new_b["points"] = [[image_width - px, py] if flip_x else [px, image_height - py]]
        new_b["x"], new_b["y"] = new_b["points"][0]
        new_b["width"] = new_b["height"] = 0
        return new_b
    if b.get("shape_type") in ("polygon", "rotation") and b.get("points"):
        pts = [
            [image_width - p[0], p[1]] if flip_x else [p[0], image_height - p[1]]
            for p in b["points"]
        ]
        new_b["points"] = pts
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        new_b["x"], new_b["y"] = min(xs), min(ys)
        new_b["width"], new_b["height"] = max(xs) - min(xs), max(ys) - min(ys)
        return new_b
    new_b["x"] = image_width - b["x"] - b["width"] if flip_x else b["x"]
    new_b["y"] = b["y"] if flip_x else image_height - b["y"] - b["height"]
    return new_b
