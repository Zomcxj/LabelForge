from typing import List, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .base import (
    BaseTransform,
    qimage_to_rgb_array,
    register_transform,
    rgb_array_to_qimage,
)


def brightness_array(arr: np.ndarray, delta: int) -> np.ndarray:
    """逐像素加 delta 并在 [0, 255] 截断，返回新的 uint8 数组。"""
    return np.clip(arr.astype(np.int16) + int(delta), 0, 255).astype(np.uint8)


def contrast_array(arr: np.ndarray, factor: float) -> np.ndarray:
    """以 128 为中点按 factor 缩放对比度，截断到 [0, 255]。"""
    scaled = (arr.astype(np.float32) - 128.0) * float(factor) + 128.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def hsv_adjust_array(
    arr: np.ndarray, h_delta: float, s_factor: float, v_factor: float
) -> np.ndarray:
    """在 HSV 空间调整色相/饱和度/明度后转回 RGB。

    走 float32 通道（H 0-360、S/V 0-1）而不是 cv2 的 8 位 H 0-179 通路：
    8 位通路的 H 量化误差会让「参数不变」的往返偏差到 ±6，float32 通路
    往返误差为 0。色相偏移按角度计，与标量实现时期的语义一致。
    """
    rgb = arr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + float(h_delta), 360.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(s_factor), 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(v_factor), 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


@register_transform
class Brightness(BaseTransform):
    name = "bright"

    def __init__(self, delta: int = 30, rng=None):
        super().__init__(rng)
        self.delta = max(-100, min(100, delta))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        arr = brightness_array(qimage_to_rgb_array(image), self.delta)
        return rgb_array_to_qimage(arr), boxes


@register_transform
class Contrast(BaseTransform):
    name = "contrast"

    def __init__(self, factor: float = 1.5, rng=None):
        super().__init__(rng)
        self.factor = max(0.5, min(2.0, factor))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        arr = contrast_array(qimage_to_rgb_array(image), self.factor)
        return rgb_array_to_qimage(arr), boxes


class _HSVAdjust(BaseTransform):

    def __init__(self, h_delta=0, s_factor=1.0, v_factor=1.0, rng=None):
        super().__init__(rng)
        self.h_delta = h_delta
        self.s_factor = max(0.0, min(2.0, s_factor))
        self.v_factor = max(0.0, min(2.0, v_factor))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        arr = hsv_adjust_array(qimage_to_rgb_array(image), self.h_delta,
                               self.s_factor, self.v_factor)
        return rgb_array_to_qimage(arr), boxes


@register_transform
class Hue(BaseTransform):
    name = "hue"

    def __init__(self, delta: int = 30, rng=None):
        super().__init__(rng)
        self.delta = max(-180, min(180, delta))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        adj = _HSVAdjust(h_delta=self.delta)
        return adj.apply(image, boxes, image_width, image_height)


@register_transform
class Saturation(BaseTransform):
    name = "saturation"

    def __init__(self, factor: float = 1.5, rng=None):
        super().__init__(rng)
        self.factor = max(0.0, min(2.0, factor))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        adj = _HSVAdjust(s_factor=self.factor)
        return adj.apply(image, boxes, image_width, image_height)


@register_transform
class Value(BaseTransform):
    name = "value"

    def __init__(self, delta: int = 30, rng=None):
        super().__init__(rng)
        self.delta = max(-100, min(100, delta))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        factor = 1.0 + self.delta / 100.0
        adj = _HSVAdjust(v_factor=factor)
        return adj.apply(image, boxes, image_width, image_height)
