"""滤镜类变换：模糊、锐化、边缘、浮雕、形态学与二值化。

所有操作用 cv2 在 RGB 数组上完成，`boxes` 原样返回——本工具只做图像处理，
不改标注。每个类都有一个同名的纯函数 `*_array`，便于直接测像素行为。
"""
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


def _odd(value, minimum: int = 1) -> int:
    """夹成不小于 minimum 的奇数。"""
    v = max(int(minimum), int(value))
    return v if v % 2 == 1 else v + 1


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """单通道灰度 -> 三通道，保证写出格式一致。"""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def gaussian_blur_array(arr: np.ndarray, ksize: int) -> np.ndarray:
    k = _odd(ksize)
    return cv2.GaussianBlur(arr, (k, k), 0)


def box_blur_array(arr: np.ndarray, ksize: int) -> np.ndarray:
    k = _odd(ksize)
    return cv2.blur(arr, (k, k))


def median_array(arr: np.ndarray, ksize: int) -> np.ndarray:
    k = _odd(ksize)
    return cv2.medianBlur(arr, k)


def sharpen_array(arr: np.ndarray, strength: float) -> np.ndarray:
    """非锐化掩模：原图减去模糊图再按强度叠加。strength=0 时回到原图。"""
    s = max(0.0, float(strength))
    if s == 0.0:
        return arr.copy()
    blurred = cv2.GaussianBlur(arr, (0, 0), 3)
    return cv2.addWeighted(arr, 1.0 + s, blurred, -s, 0)


def sobel_array(arr: np.ndarray, strength: float) -> np.ndarray:
    s = max(0.0, float(strength))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5,
                          cv2.convertScaleAbs(gy), 0.5, 0)
    scaled = np.clip(mag.astype(np.float32) * s, 0, 255).astype(np.uint8)
    return _to_rgb(scaled)


def emboss_array(arr: np.ndarray, strength: float) -> np.ndarray:
    """浮雕：固定 3×3 核，强度控制结果与原图的混合比例。"""
    s = max(0.0, float(strength))
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
    relief = cv2.filter2D(arr, -1, kernel)
    if s == 0.0:
        return arr.copy()
    return cv2.addWeighted(arr, 1.0 - min(s, 1.0), relief, min(s, 1.0), 0)


def dilate_array(arr: np.ndarray, ksize: int) -> np.ndarray:
    k = _odd(ksize)
    return cv2.dilate(arr, np.ones((k, k), np.uint8))


def erode_array(arr: np.ndarray, ksize: int) -> np.ndarray:
    k = _odd(ksize)
    return cv2.erode(arr, np.ones((k, k), np.uint8))


def threshold_array(arr: np.ndarray, thresh: int, invert: bool) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, int(thresh), 255, mode)
    return _to_rgb(out)


def adaptive_threshold_array(arr: np.ndarray, block_size: int, c: float) -> np.ndarray:
    block = _odd(block_size, minimum=3)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, block, float(c))
    return _to_rgb(out)


class _FilterBase(BaseTransform):
    """滤镜基类：把 QImage 转数组、调子类指定的纯函数、再转回 QImage。"""

    _array_fn = None

    def _run(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, image: QImage, boxes: List[dict],
              image_width: int, image_height: int) -> Tuple[QImage, List[dict]]:
        arr = qimage_to_rgb_array(image)
        return rgb_array_to_qimage(self._run(arr)), boxes


@register_transform
class GaussianBlur(_FilterBase):
    name = "gauss_blur"

    def __init__(self, ksize: int = 5, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize)

    def _run(self, arr):
        return gaussian_blur_array(arr, self._ksize)


@register_transform
class BoxBlur(_FilterBase):
    name = "box_blur"

    def __init__(self, ksize: int = 5, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize)

    def _run(self, arr):
        return box_blur_array(arr, self._ksize)


@register_transform
class MedianFilter(_FilterBase):
    name = "median"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize)

    def _run(self, arr):
        return median_array(arr, self._ksize)


@register_transform
class Sharpen(_FilterBase):
    name = "sharpen"

    def __init__(self, strength: float = 1.0, rng=None):
        super().__init__(rng)
        self._strength = max(0.0, float(strength))

    def _run(self, arr):
        return sharpen_array(arr, self._strength)


@register_transform
class SobelEdge(_FilterBase):
    name = "sobel"

    def __init__(self, strength: float = 1.0, rng=None):
        super().__init__(rng)
        self._strength = max(0.0, float(strength))

    def _run(self, arr):
        return sobel_array(arr, self._strength)


@register_transform
class Emboss(_FilterBase):
    name = "emboss"

    def __init__(self, strength: float = 1.0, rng=None):
        super().__init__(rng)
        self._strength = max(0.0, float(strength))

    def _run(self, arr):
        return emboss_array(arr, self._strength)


@register_transform
class Dilate(_FilterBase):
    name = "dilate"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize)

    def _run(self, arr):
        return dilate_array(arr, self._ksize)


@register_transform
class Erode(_FilterBase):
    name = "erode"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize)

    def _run(self, arr):
        return erode_array(arr, self._ksize)


@register_transform
class Threshold(_FilterBase):
    name = "threshold"

    def __init__(self, thresh: int = 128, invert: bool = False, rng=None):
        super().__init__(rng)
        self._thresh = int(max(0, min(255, thresh)))
        self._invert = bool(invert)

    def _run(self, arr):
        return threshold_array(arr, self._thresh, self._invert)


@register_transform
class AdaptiveThreshold(_FilterBase):
    name = "adaptive_thresh"

    def __init__(self, block_size: int = 11, c: float = 2, rng=None):
        super().__init__(rng)
        self._block = _odd(block_size, minimum=3)
        self._c = float(c)

    def _run(self, arr):
        return adaptive_threshold_array(arr, self._block, self._c)
