"""色彩类变换：反色、Gamma、白平衡、直方图均衡、灰度、色阶、通道提取。

纯 numpy / cv2 内核，`boxes` 原样返回。每个类都有一个同名纯函数 `*_array`。
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

_CHANNEL_INDEX = {"r": 0, "g": 1, "b": 2}


def invert_array(arr: np.ndarray) -> np.ndarray:
    return (255 - arr.astype(np.int16)).astype(np.uint8)


def gamma_array(arr: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 校正：gamma=1 恒等，>1 提亮，<1 压暗。"""
    g = float(gamma)
    if g <= 0:
        g = 1.0
    table = np.array(
        [((i / 255.0) ** (1.0 / g)) * 255.0 for i in range(256)], np.uint8)
    return table[arr]


def white_balance_array(arr: np.ndarray, temperature: int) -> np.ndarray:
    """色温：正数偏暖（加红减蓝），负数偏冷，0 为恒等。"""
    t = max(-100, min(100, int(temperature)))
    if t == 0:
        return arr.copy()
    shift = (t / 100.0) * 0.3
    out = arr.astype(np.float32)
    out[..., 0] *= (1.0 + shift)
    out[..., 2] *= (1.0 - shift)
    return np.clip(out, 0, 255).astype(np.uint8)


def hist_equal_array(arr: np.ndarray) -> np.ndarray:
    """逐通道直方图均衡；纯色图均衡后仍是纯色。"""
    src = np.ascontiguousarray(arr, dtype=np.uint8)
    out = np.empty_like(src)
    for c in range(3):
        out[..., c] = cv2.equalizeHist(np.ascontiguousarray(src[..., c]))
    return out


def grayscale_array(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def levels_array(arr: np.ndarray, black: int, white: int) -> np.ndarray:
    """色阶：把 [black, white] 拉伸到 [0, 255]。white<=black 时原样返回。"""
    b = max(0, min(254, int(black)))
    w = max(1, min(255, int(white)))
    if w <= b:
        return arr.copy()
    out = (arr.astype(np.float32) - b) * (255.0 / (w - b))
    return np.clip(out, 0, 255).astype(np.uint8)


def channel_array(arr: np.ndarray, channel: str) -> np.ndarray:
    """只保留一个通道，其余置 0。"""
    idx = _CHANNEL_INDEX.get(str(channel).lower(), 0)
    out = np.zeros_like(arr)
    out[..., idx] = arr[..., idx]
    return out


class _ColorBase(BaseTransform):
    def _run(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, image: QImage, boxes: List[dict],
              image_width: int, image_height: int) -> Tuple[QImage, List[dict]]:
        arr = qimage_to_rgb_array(image)
        return rgb_array_to_qimage(self._run(arr)), boxes


@register_transform
class Invert(_ColorBase):
    name = "invert"

    def _run(self, arr):
        return invert_array(arr)


@register_transform
class Gamma(_ColorBase):
    name = "gamma"

    def __init__(self, gamma: float = 1.0, rng=None):
        super().__init__(rng)
        self._gamma = max(0.1, min(3.0, float(gamma)))

    def _run(self, arr):
        return gamma_array(arr, self._gamma)


@register_transform
class WhiteBalance(_ColorBase):
    name = "white_balance"

    def __init__(self, temperature: int = 0, rng=None):
        super().__init__(rng)
        self._temperature = max(-100, min(100, int(temperature)))

    def _run(self, arr):
        return white_balance_array(arr, self._temperature)


@register_transform
class HistEqualize(_ColorBase):
    name = "hist_equal"

    def _run(self, arr):
        return hist_equal_array(arr)


@register_transform
class Grayscale(_ColorBase):
    name = "grayscale"

    def _run(self, arr):
        return grayscale_array(arr)


@register_transform
class Levels(_ColorBase):
    name = "levels"

    def __init__(self, black: int = 0, white: int = 255, rng=None):
        super().__init__(rng)
        self._black = max(0, min(254, int(black)))
        self._white = max(1, min(255, int(white)))

    def _run(self, arr):
        return levels_array(arr, self._black, self._white)


@register_transform
class ChannelExtract(_ColorBase):
    name = "channel"

    def __init__(self, channel: str = "r", rng=None):
        super().__init__(rng)
        self._channel = str(channel).lower()
        if self._channel not in _CHANNEL_INDEX:
            self._channel = "r"

    def _run(self, arr):
        return channel_array(arr, self._channel)
