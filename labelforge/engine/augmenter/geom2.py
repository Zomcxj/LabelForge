"""几何类变换：90 度旋转、错切、透视、区域裁剪、边框填充。

纯 numpy / cv2 内核。几何变换会改变画布尺寸，`boxes` 仍原样返回——
图像编辑工具不处理标注，由调用方自行决定标注是否失效。
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

_ANGLE_STEPS = (90, 180, 270)


def rotate90_array(arr: np.ndarray, angle: int) -> np.ndarray:
    """顺时针旋转 90/180/270 度；其它角度按 90 度处理。"""
    a = int(angle) if int(angle) in _ANGLE_STEPS else 90
    code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[a]
    return cv2.rotate(arr, code)


def skew_array(arr: np.ndarray, x_angle: float, y_angle: float) -> np.ndarray:
    """错切：x/y 方向各按角度倾斜，画布尺寸保持不变，空出的地方填边。"""
    ax, ay = float(x_angle), float(y_angle)
    if ax == 0.0 and ay == 0.0:
        return arr.copy()
    h, w = arr.shape[:2]
    tx = np.tan(np.deg2rad(ax))
    ty = np.tan(np.deg2rad(ay))
    matrix = np.array([[1.0, tx, 0.0], [ty, 1.0, 0.0]], np.float32)
    return cv2.warpAffine(arr, matrix, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)


def perspective_array(arr: np.ndarray, offset: int) -> np.ndarray:
    """四角等距内缩 offset 像素后拉回原尺寸，产生透视畸变。"""
    off = int(offset)
    if off == 0:
        return arr.copy()
    h, w = arr.shape[:2]
    off = max(0, min(off, min(h, w) // 2 - 1))
    if off == 0:
        return arr.copy()
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array([[off, off], [w - 1 - off, off],
                    [w - 1 - off, h - 1 - off], [off, h - 1 - off]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(arr, matrix, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def crop_region_array(arr: np.ndarray, rect) -> np.ndarray:
    """按像素矩形 (x, y, w, h) 裁剪；矩形越界时钳到图内，空矩形返回原图。"""
    if rect is None:
        return arr.copy()
    x, y, w, h = (int(round(v)) for v in rect)
    ih, iw = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(iw, x + w), min(ih, y + h)
    if x1 <= x0 or y1 <= y0:
        return arr.copy()
    return arr[y0:y1, x0:x1].copy()


def pad_array(arr: np.ndarray, top: int, bottom: int, left: int,
              right: int, color=(114, 114, 114)) -> np.ndarray:
    """四边填充；color 为 RGB 三元组。"""
    t, b = max(0, int(top)), max(0, int(bottom))
    l, r = max(0, int(left)), max(0, int(right))
    if t == b == l == r == 0:
        return arr.copy()
    c = tuple(int(v) for v in color)[:3]
    return cv2.copyMakeBorder(arr, t, b, l, r, cv2.BORDER_CONSTANT, value=c)


class _GeomBase(BaseTransform):
    def _run(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, image: QImage, boxes: List[dict],
              image_width: int, image_height: int) -> Tuple[QImage, List[dict]]:
        arr = qimage_to_rgb_array(image)
        return rgb_array_to_qimage(self._run(arr)), boxes


@register_transform
class Rotate90(_GeomBase):
    name = "rotate90"

    def __init__(self, angle: int = 90, rng=None):
        super().__init__(rng)
        self._angle = int(angle) if int(angle) in _ANGLE_STEPS else 90

    def _run(self, arr):
        return rotate90_array(arr, self._angle)


@register_transform
class Skew(_GeomBase):
    name = "skew"

    def __init__(self, x_angle: float = 0, y_angle: float = 0, rng=None):
        super().__init__(rng)
        self._x_angle = float(x_angle)
        self._y_angle = float(y_angle)

    def _run(self, arr):
        return skew_array(arr, self._x_angle, self._y_angle)


@register_transform
class Perspective(_GeomBase):
    name = "perspective"

    def __init__(self, offset: int = 0, rng=None):
        super().__init__(rng)
        self._offset = max(0, int(offset))

    def _run(self, arr):
        return perspective_array(arr, self._offset)


@register_transform
class CropRegion(_GeomBase):
    name = "crop_region"

    def __init__(self, rect=None, rng=None):
        super().__init__(rng)
        self._rect = None if rect is None else tuple(rect)

    def _run(self, arr):
        return crop_region_array(arr, self._rect)


@register_transform
class EdgePad(_GeomBase):
    name = "pad"

    def __init__(self, top: int = 0, bottom: int = 0, left: int = 0,
                 right: int = 0, color=(114, 114, 114), rng=None):
        super().__init__(rng)
        self._edges = (max(0, int(top)), max(0, int(bottom)),
                       max(0, int(left)), max(0, int(right)))
        self._color = tuple(int(v) for v in color)[:3]

    def _run(self, arr):
        t, b, l, r = self._edges
        return pad_array(arr, t, b, l, r, self._color)


def resize_long_side_array(arr: np.ndarray, long_side: int) -> np.ndarray:
    """按长边等比缩放：只定长边，短边跟着算，不改变宽高比。"""
    h, w = arr.shape[:2]
    target = max(16, int(long_side))
    if w <= 0 or h <= 0:
        return arr.copy()
    scale = target / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        return arr.copy()
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


@register_transform
class ResizeLongSide(_GeomBase):
    name = "resize"

    def __init__(self, long_side: int = 1024, rng=None):
        super().__init__(rng)
        self._long_side = max(16, int(long_side))

    def _run(self, arr):
        return resize_long_side_array(arr, self._long_side)
