from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import random

import numpy as np
from PySide6.QtGui import QImage

_registered_transforms: Dict[str, type] = {}


def register_transform(cls):
    if issubclass(cls, BaseTransform) and cls is not BaseTransform:
        name = getattr(cls, 'name', None) or cls.__name__.lower()
        _registered_transforms[name] = cls
    return cls


def get_all_transforms() -> Dict[str, type]:
    return dict(_registered_transforms)


def rgb888_buffer(image: QImage) -> bytearray:
    """Format_RGB888 图的像素，拷成可写 bytearray。

    PySide6 的 `bits()` 直接返回带长度的 `memoryview`：不要 `setsize`、没有
    `asstring`，`bytes()` 就是整块像素的拷贝（PyQt5 那套 `setsize` +
    `asstring(byteCount())` 在这里全都不存在）。

    这里的 `bytes()` 拷贝是必须的另一个原因：`bits()` 返回的视图**不持有**
    `image`，不拷贝而直接外泄会在 QImage 回收后指向垃圾内存。
    """
    img = image.convertToFormat(QImage.Format_RGB888)
    return bytearray(bytes(img.bits()))


def qimage_to_rgb_array(image: QImage) -> np.ndarray:
    """QImage → (H, W, 3) uint8 RGB 数组。

    单通道/灰度/带 alpha 的输入统一经 RGB888 转换：alpha 直接被丢弃。
    `bytesPerLine()` 含 4 字节对齐填充，必须按步长切掉，不能按 width*3 读。
    """
    img = image.convertToFormat(QImage.Format_RGB888)
    w, h, stride = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(bytes(img.bits()), np.uint8).reshape(h, stride)
    return buf[:, :w * 3].reshape(h, w, 3).copy()


def rgb_array_to_qimage(arr: np.ndarray) -> QImage:
    """(H, W, 3) uint8 → QImage(Format_RGB888)。

    QImage 不接管传入缓冲区的生命周期，末尾 `.copy()` 让图像自带像素，
    否则缓冲区释放后图像指向垃圾内存。传 `arr.data` 而非 `arr.tobytes()`：
    memoryview 不复制，`.copy()` 已经完成唯一一次必要拷贝。
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 3,
                  QImage.Format_RGB888).copy()


def qimage_gray_to_array(image: QImage) -> np.ndarray:
    """QImage → (H, W) uint8 灰度数组（任意格式先转 Grayscale8）。"""
    img = image.convertToFormat(QImage.Format_Grayscale8)
    w, h, stride = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(bytes(img.bits()), np.uint8).reshape(h, stride)
    return buf[:, :w].copy()


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
