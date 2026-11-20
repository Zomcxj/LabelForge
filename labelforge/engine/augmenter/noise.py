from typing import List, Tuple

import numpy as np
from PySide6.QtGui import QImage

from .base import (
    BaseTransform,
    qimage_to_rgb_array,
    register_transform,
    rgb_array_to_qimage,
)


def _generator(rng) -> np.random.Generator:
    """由 random.Random（或全局 random 模块）派生 numpy Generator。

    传进来的 rng 是 `random.Random`，numpy 用不了；只从它取一个整数作种子，
    同一个 seed 仍然得到同一串噪声（与逐像素时代同样可复现）。
    """
    return np.random.default_rng(rng.randrange(2 ** 32))


def gaussian_noise_array(
    arr: np.ndarray, mean: float, sigma: float, rng
) -> np.ndarray:
    """给每个通道加高斯噪声，截断到 [0, 255]。"""
    gen = _generator(rng)
    noise = gen.normal(mean, sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper_array(arr: np.ndarray, prob: float, rng) -> np.ndarray:
    """按 prob 比例撒椒盐：一半置 255，一半置 0。不修改传入数组。"""
    gen = _generator(rng)
    out = np.array(arr, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    total = h * w
    n_salt = int(total * prob / 2)
    n_pepper = int(total * prob / 2)
    if n_salt:
        flat = gen.integers(0, total, size=n_salt)
        out[flat // w, flat % w] = 255
    if n_pepper:
        flat = gen.integers(0, total, size=n_pepper)
        out[flat // w, flat % w] = 0
    return out


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
        arr = gaussian_noise_array(qimage_to_rgb_array(image),
                                   self.mean, self.sigma, self.rng)
        return rgb_array_to_qimage(arr), boxes


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
        arr = salt_pepper_array(qimage_to_rgb_array(image), self.prob, self.rng)
        return rgb_array_to_qimage(arr), boxes
