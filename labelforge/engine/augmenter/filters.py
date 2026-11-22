"""滤镜类变换：模糊、锐化、边缘、浮雕、形态学与二值化。

所有操作用 cv2 在 RGB 数组上完成，`boxes` 原样返回——本工具只做图像编辑，
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


# ---------------------------------------------------------------- 形态学

#: 形态学算子：名字 -> OpenCV 常量
_MORPH_CODES = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


def morph_array(arr: np.ndarray, op: str, ksize: int) -> np.ndarray:
    """形态学运算：开/闭/梯度/顶帽/黑帽。核为方形的 ksize×ksize。"""
    code = _MORPH_CODES.get(str(op), cv2.MORPH_OPEN)
    k = _odd(ksize, minimum=3)
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(arr, code, kernel)


@register_transform
class MorphOpen(_FilterBase):
    name = "morph_open"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize, minimum=3)

    def _run(self, arr):
        return morph_array(arr, "open", self._ksize)


@register_transform
class MorphClose(_FilterBase):
    name = "morph_close"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize, minimum=3)

    def _run(self, arr):
        return morph_array(arr, "close", self._ksize)


@register_transform
class MorphGradient(_FilterBase):
    name = "morph_gradient"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize, minimum=3)

    def _run(self, arr):
        return morph_array(arr, "gradient", self._ksize)


@register_transform
class TopHat(_FilterBase):
    name = "tophat"

    def __init__(self, ksize: int = 9, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize, minimum=3)

    def _run(self, arr):
        return morph_array(arr, "tophat", self._ksize)


@register_transform
class BlackHat(_FilterBase):
    name = "blackhat"

    def __init__(self, ksize: int = 9, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize, minimum=3)

    def _run(self, arr):
        return morph_array(arr, "blackhat", self._ksize)


# ------------------------------------------------------- 锐化与降噪

def usm_array(arr: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """USM 锐化：原图 + amount ×（原图 − 高斯模糊）。amount=0 时回到原图。"""
    r = max(0.1, float(radius))
    a = max(0.0, float(amount))
    if a == 0.0:
        return arr.copy()
    blurred = cv2.GaussianBlur(arr, (0, 0), r)
    return cv2.addWeighted(arr, 1.0 + a, blurred, -a, 0)


def laplacian_array(arr: np.ndarray, ksize: int) -> np.ndarray:
    """拉普拉斯锐化：把二阶导叠加回原图，边缘更硬。"""
    k = _odd(ksize, minimum=3)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edge = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=k))
    return cv2.addWeighted(arr, 1.0, _to_rgb(edge), 0.5, 0)


def bilateral_array(arr: np.ndarray, d: int, sigma_color: float,
                    sigma_space: float) -> np.ndarray:
    """双边滤波：保边平滑，皮肤/布料去噪常用。"""
    return cv2.bilateralFilter(arr, max(1, int(d)), float(sigma_color),
                               float(sigma_space))


def nlmeans_array(arr: np.ndarray, h: float) -> np.ndarray:
    """非局部均值降噪：比高斯慢，但细节保留得更好。"""
    strength = max(1.0, float(h))
    return cv2.fastNlMeansDenoisingColored(
        np.ascontiguousarray(arr), None, strength, strength, 7, 21)


def motion_blur_array(arr: np.ndarray, length: int, angle: float) -> np.ndarray:
    """运动模糊：沿给定角度拉一条长度为 length 的核。"""
    size = max(1, int(length))
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5),
                                     float(angle), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = kernel.sum()
    if total <= 0:
        return arr.copy()
    return cv2.filter2D(arr, -1, kernel / total)


def mosaic_array(arr: np.ndarray, block: int) -> np.ndarray:
    """马赛克：按 block×block 取块内均值。用于遮住人脸、车牌等区域。"""
    b = max(1, int(block))
    h, w = arr.shape[:2]
    if b == 1 or h < b or w < b:
        return arr.copy()
    small = cv2.resize(arr, (max(1, w // b), max(1, h // b)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


@register_transform
class USMSharpen(_FilterBase):
    name = "usm"

    def __init__(self, radius: float = 3.0, amount: float = 1.0, rng=None):
        super().__init__(rng)
        self._radius = float(radius)
        self._amount = float(amount)

    def _run(self, arr):
        return usm_array(arr, self._radius, self._amount)


@register_transform
class LaplacianSharpen(_FilterBase):
    name = "laplacian"

    def __init__(self, ksize: int = 3, rng=None):
        super().__init__(rng)
        self._ksize = _odd(ksize, minimum=3)

    def _run(self, arr):
        return laplacian_array(arr, self._ksize)


@register_transform
class Bilateral(_FilterBase):
    name = "bilateral"

    def __init__(self, d: int = 9, sigma_color: float = 75.0,
                 sigma_space: float = 75.0, rng=None):
        super().__init__(rng)
        self._d = max(1, int(d))
        self._sigma_color = float(sigma_color)
        self._sigma_space = float(sigma_space)

    def _run(self, arr):
        return bilateral_array(arr, self._d, self._sigma_color,
                               self._sigma_space)


@register_transform
class NLMeans(_FilterBase):
    name = "nlmeans"

    def __init__(self, h: float = 10.0, rng=None):
        super().__init__(rng)
        self._h = max(1.0, float(h))

    def _run(self, arr):
        return nlmeans_array(arr, self._h)


@register_transform
class MotionBlur(_FilterBase):
    name = "motion_blur"

    def __init__(self, length: int = 15, angle: float = 0.0, rng=None):
        super().__init__(rng)
        self._length = max(1, int(length))
        self._angle = float(angle)

    def _run(self, arr):
        return motion_blur_array(arr, self._length, self._angle)


@register_transform
class Mosaic(_FilterBase):
    name = "mosaic"

    def __init__(self, block: int = 12, rng=None):
        super().__init__(rng)
        self._block = max(1, int(block))

    def _run(self, arr):
        return mosaic_array(arr, self._block)
