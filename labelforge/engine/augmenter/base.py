from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import random

from PyQt5.QtGui import QImage

_registered_transforms: Dict[str, type] = {}


def register_transform(cls):
    if issubclass(cls, BaseTransform) and cls is not BaseTransform:
        name = getattr(cls, 'name', None) or cls.__name__.lower()
        _registered_transforms[name] = cls
    return cls


def get_all_transforms() -> Dict[str, type]:
    return dict(_registered_transforms)


def map_box_points(box: dict, fn) -> dict:
    """对框所有顶点应用 fn(x, y)->(x, y)，同步 bbox，并保留形状元信息。

    rectangle 无 points 时用 4 个角点参与映射（旋转/翻转后取外接框），
    polygon/rotation/point 直接映射原始点。不修改传入 box。
    """
    new_box = dict(box)
    points = box.get("points")
    if not points:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    mapped = [[*fn(p[0], p[1])] for p in points]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    new_box["x"], new_box["y"] = min(xs), min(ys)
    new_box["width"], new_box["height"] = max(xs) - min(xs), max(ys) - min(ys)
    if box.get("points"):
        new_box["points"] = mapped
    return new_box


class BaseTransform(ABC):
    name: str

    def __init__(self, rng=None):
        # rng 由 Augmenter 注入，用于让随机变换可复现；不传时用全局 random
        self.rng = rng if rng is not None else random

    @abstractmethod
    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        pass
