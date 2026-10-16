import random
from typing import List, Tuple

from PyQt5.QtGui import QImage

from .base import BaseTransform, register_transform


@register_transform
class GaussianNoise(BaseTransform):
    name = "gauss"

    def __init__(self, mean: int = 0, sigma: int = 25, rng=None):
        super().__init__(rng)
        self.mean = max(0, min(255, mean))
        self.sigma = max(1, min(100, sigma))

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
                    noise = int(self.rng.gauss(self.mean, self.sigma))
                    arr[idx + c] = max(0, min(255, arr[idx + c] + noise))
        result = QImage(arr, img.width(), img.height(), stride, QImage.Format_RGB888)
        return result, boxes


@register_transform
class SaltPepper(BaseTransform):
    name = "saltpepper"

    def __init__(self, prob: float = 0.05, rng=None):
        super().__init__(rng)
        self.prob = max(0.0, min(0.5, prob))

    def apply(
        self, image: QImage, boxes: List[dict],
        image_width: int, image_height: int
    ) -> Tuple[QImage, List[dict]]:
        img = image.convertToFormat(QImage.Format_RGB888)
        bits = img.bits()
        bits.setsize(img.byteCount())
        arr = bytearray(bits.asstring(img.byteCount()))
        stride = img.bytesPerLine()
        total = img.width() * img.height()
        n_salt = int(total * self.prob / 2)
        n_pepper = int(total * self.prob / 2)
        for _ in range(n_salt):
            x = self.rng.randrange(img.width())
            y = self.rng.randrange(img.height())
            idx = y * stride + x * 3
            arr[idx] = arr[idx + 1] = arr[idx + 2] = 255
        for _ in range(n_pepper):
            x = self.rng.randrange(img.width())
            y = self.rng.randrange(img.height())
            idx = y * stride + x * 3
            arr[idx] = arr[idx + 1] = arr[idx + 2] = 0
        result = QImage(arr, img.width(), img.height(), stride, QImage.Format_RGB888)
        return result, boxes
