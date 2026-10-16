import math
from typing import List, Tuple

from PyQt5.QtGui import QImage

from .base import BaseTransform, register_transform


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
        img = image.convertToFormat(QImage.Format_RGB888)
        bits = img.bits()
        bits.setsize(img.byteCount())
        arr = bytearray(bits.asstring(img.byteCount()))
        stride = img.bytesPerLine()
        for y in range(img.height()):
            offset = y * stride
            for x in range(img.width()):
                idx = offset + x * 3
                for c in range(3):
                    arr[idx + c] = max(0, min(255, arr[idx + c] + self.delta))
        result = QImage(arr, img.width(), img.height(), stride, QImage.Format_RGB888)
        return result, boxes


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
        img = image.convertToFormat(QImage.Format_RGB888)
        bits = img.bits()
        bits.setsize(img.byteCount())
        arr = bytearray(bits.asstring(img.byteCount()))
        stride = img.bytesPerLine()
        for y in range(img.height()):
            offset = y * stride
            for x in range(img.width()):
                idx = offset + x * 3
                for c in range(3):
                    v = int(128 + (arr[idx + c] - 128) * self.factor)
                    arr[idx + c] = max(0, min(255, v))
        result = QImage(arr, img.width(), img.height(), stride, QImage.Format_RGB888)
        return result, boxes


def _rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn
    h = 0.0
    s = 0.0
    v = mx
    if diff > 0:
        if mx == r:
            h = ((g - b) / diff) % 6
        elif mx == g:
            h = (b - r) / diff + 2
        else:
            h = (r - g) / diff + 4
        h *= 60
        if h < 0:
            h += 360
        s = diff / mx
    return h, s, v


def _hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (r + m) * 255, (g + m) * 255, (b + m) * 255


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
        img = image.convertToFormat(QImage.Format_RGB888)
        bits = img.bits()
        bits.setsize(img.byteCount())
        arr = bytearray(bits.asstring(img.byteCount()))
        stride = img.bytesPerLine()
        for y in range(img.height()):
            offset = y * stride
            for x in range(img.width()):
                idx = offset + x * 3
                r, g, b = arr[idx], arr[idx + 1], arr[idx + 2]
                h, s, v = _rgb_to_hsv(r, g, b)
                h = (h + self.h_delta) % 360
                s = max(0.0, min(1.0, s * self.s_factor))
                v = max(0.0, min(1.0, v * self.v_factor))
                nr, ng, nb = _hsv_to_rgb(h, s, v)
                arr[idx] = max(0, min(255, int(nr)))
                arr[idx + 1] = max(0, min(255, int(ng)))
                arr[idx + 2] = max(0, min(255, int(nb)))
        result = QImage(arr, img.width(), img.height(), stride, QImage.Format_RGB888)
        return result, boxes


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
