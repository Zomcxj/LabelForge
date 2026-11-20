"""图像处理工具的操作清单与处理链。

对单张图按顺序执行若干变换，可选只在选区内生效。变换本身全部来自
`engine.augmenter` 的注册表，本模块只负责「清单 + 编排 + 蒙版合成」。
"""
import inspect
import random
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from .augmenter import get_all_transforms
from .augmenter.base import (
    qimage_gray_to_array,
    qimage_to_rgb_array,
    rgb_array_to_qimage,
)

# 预览降采样的长边上限；实时预览只在这个尺度上跑处理链
PREVIEW_MAX_SIDE = 1280

# 分组名（界面按此顺序渲染）
GROUP_GEOMETRY = "几何"
GROUP_COLOR = "色彩"
GROUP_FILTER = "滤镜"
GROUP_NOISE = "噪声"

_KSIZE_CHOICES = [3, 5, 7, 9, 15, 21, 31]

# key -> (分组, 显示名, 参数列表)
# 参数为 (name, kind, min, max, default)；choice 为 (name, "choice", 选项, 默认)
# name 必须与变换构造器的形参同名，加载时由 build_transform 校验。
OPERATION_META: dict = {
    # ---- 几何 ----
    "fliph": (GROUP_GEOMETRY, "水平翻转", []),
    "flipv": (GROUP_GEOMETRY, "垂直翻转", []),
    "trans": (GROUP_GEOMETRY, "平移", [
        ("max_dx", "int", 0, 100, 20),
        ("max_dy", "int", 0, 100, 20),
    ]),
    "rotate": (GROUP_GEOMETRY, "旋转", [
        ("max_angle", "float", 0.0, 45.0, 15.0),
    ]),
    "scale": (GROUP_GEOMETRY, "缩放", [
        ("min", "float", 0.1, 1.0, 0.8),
        ("max", "float", 1.0, 3.0, 1.2),
    ]),
    "rotate90": (GROUP_GEOMETRY, "旋转 90 度", [
        ("angle", "choice", [90, 180, 270], 90),
    ]),
    "skew": (GROUP_GEOMETRY, "错切", [
        ("x_angle", "float", -45.0, 45.0, 0.0),
        ("y_angle", "float", -45.0, 45.0, 0.0),
    ]),
    "perspective": (GROUP_GEOMETRY, "透视", [
        ("offset", "int", 0, 200, 0),
    ]),
    "crop_region": (GROUP_GEOMETRY, "按选区裁剪", []),
    "pad": (GROUP_GEOMETRY, "边框填充", [
        ("top", "int", 0, 500, 0),
        ("bottom", "int", 0, 500, 0),
        ("left", "int", 0, 500, 0),
        ("right", "int", 0, 500, 0),
    ]),
    # ---- 色彩 ----
    "bright": (GROUP_COLOR, "亮度", [
        ("delta", "int", -100, 100, 30),
    ]),
    "contrast": (GROUP_COLOR, "对比度", [
        ("factor", "float", 0.5, 2.0, 1.5),
    ]),
    "hue": (GROUP_COLOR, "色相", [
        ("delta", "int", -180, 180, 30),
    ]),
    "saturation": (GROUP_COLOR, "饱和度", [
        ("factor", "float", 0.0, 2.0, 1.5),
    ]),
    "value": (GROUP_COLOR, "明度", [
        ("delta", "int", -100, 100, 30),
    ]),
    "invert": (GROUP_COLOR, "反色", []),
    "gamma": (GROUP_COLOR, "Gamma", [
        ("gamma", "float", 0.1, 3.0, 1.0),
    ]),
    "white_balance": (GROUP_COLOR, "白平衡", [
        ("temperature", "int", -100, 100, 0),
    ]),
    "hist_equal": (GROUP_COLOR, "直方图均衡", []),
    "grayscale": (GROUP_COLOR, "灰度", []),
    "levels": (GROUP_COLOR, "色阶", [
        ("black", "int", 0, 254, 0),
        ("white", "int", 1, 255, 255),
    ]),
    "channel": (GROUP_COLOR, "通道提取", [
        ("channel", "choice", ["r", "g", "b"], "r"),
    ]),
    # ---- 滤镜 ----
    "gauss_blur": (GROUP_FILTER, "高斯模糊", [
        ("ksize", "choice", _KSIZE_CHOICES, 5),
    ]),
    "box_blur": (GROUP_FILTER, "均值模糊", [
        ("ksize", "choice", _KSIZE_CHOICES, 5),
    ]),
    "median": (GROUP_FILTER, "中值滤波", [
        ("ksize", "choice", _KSIZE_CHOICES, 3),
    ]),
    "sharpen": (GROUP_FILTER, "锐化", [
        ("strength", "float", 0.0, 3.0, 1.0),
    ]),
    "sobel": (GROUP_FILTER, "边缘检测", [
        ("strength", "float", 0.0, 3.0, 1.0),
    ]),
    "emboss": (GROUP_FILTER, "浮雕", [
        ("strength", "float", 0.0, 1.0, 1.0),
    ]),
    "dilate": (GROUP_FILTER, "膨胀", [
        ("ksize", "choice", _KSIZE_CHOICES, 3),
    ]),
    "erode": (GROUP_FILTER, "腐蚀", [
        ("ksize", "choice", _KSIZE_CHOICES, 3),
    ]),
    "threshold": (GROUP_FILTER, "二值化", [
        ("thresh", "int", 0, 255, 128),
        ("invert", "bool", False),
    ]),
    "adaptive_thresh": (GROUP_FILTER, "自适应二值化", [
        ("block_size", "choice", [3, 5, 7, 11, 15, 21, 31], 11),
        ("c", "float", 0.0, 20.0, 2.0),
    ]),
    # ---- 噪声 ----
    "gauss": (GROUP_NOISE, "高斯噪声", [
        ("mean", "int", 0, 60, 0),
        ("sigma", "int", 1, 100, 25),
    ]),
    "saltpepper": (GROUP_NOISE, "椒盐噪声", [
        ("prob", "float", 0.0, 0.5, 0.05),
    ]),
}

_GROUP_ORDER = [GROUP_GEOMETRY, GROUP_COLOR, GROUP_FILTER, GROUP_NOISE]

# 全部 key，按分组顺序展开
OPERATION_ORDER: List[str] = [
    key for group in _GROUP_ORDER
    for key in OPERATION_META if OPERATION_META[key][0] == group
]

# 几何类会改变画布尺寸或像素位置，不能做选区合成
GEOMETRIC_KEYS = frozenset(
    key for key, meta in OPERATION_META.items() if meta[0] == GROUP_GEOMETRY)


def group_of(key: str) -> str:
    return OPERATION_META[key][0]


def label_of(key: str) -> str:
    return OPERATION_META[key][1]


def params_of(key: str) -> list:
    return OPERATION_META[key][2]


def default_params(key: str) -> dict:
    """按 meta 取出一个操作的默认参数（各项末位即默认值）。"""
    return {spec[0]: spec[-1] for spec in params_of(key)}


def is_geometric(key: str) -> bool:
    return key in GEOMETRIC_KEYS


def build_transform(name: str, params: Optional[dict] = None,
                    seed: Optional[int] = None):
    """按 key 与参数构造变换实例；未知 key 或未知参数名抛 ValueError。"""
    registry = get_all_transforms()
    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"未知的图像处理操作: {name}")
    kwargs = dict(params or {})
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "rng"}
    unknown = set(kwargs) - accepted
    if unknown:
        raise ValueError(
            f"{name} 不接受参数 {sorted(unknown)}，可接受 {sorted(accepted)}")
    if seed is not None:
        kwargs["rng"] = random.Random(seed)
    return cls(**kwargs)


def apply_chain(image: QImage, specs: List[dict],
                seed: Optional[int] = None) -> QImage:
    """顺序执行处理链。

    每步用 `seed + 序号` 派生随机源：整条链可复现，且各步互不共享序列。
    """
    out = image
    for index, spec in enumerate(specs or []):
        step_seed = None if seed is None else seed + index
        transform = build_transform(spec["name"], spec.get("params"), step_seed)
        out, _ = transform.apply(out, [], out.width(), out.height())
    return out


def apply_chain_with_mask(image: QImage, specs: List[dict], mask: QImage,
                          seed: Optional[int] = None) -> QImage:
    """只在蒙版白色区域应用处理链，其余保留原图。

    处理链改变了画布尺寸（几何类）时无法逐像素合成，直接返回处理结果。
    """
    processed = apply_chain(image, specs, seed)
    if not specs:
        return processed
    original = qimage_to_rgb_array(image)
    result = qimage_to_rgb_array(processed)
    if result.shape != original.shape:
        return processed
    gray = qimage_gray_to_array(_scaled_mask(mask, image))
    inside = gray > 127
    return rgb_array_to_qimage(np.where(inside[:, :, None], result, original))


def _scaled_mask(mask: QImage, image: QImage) -> QImage:
    if mask.width() == image.width() and mask.height() == image.height():
        return mask
    return mask.scaled(image.width(), image.height(),
                       Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


def downsample_for_preview(image: QImage,
                           max_side: int = PREVIEW_MAX_SIDE) -> QImage:
    """长边超过 max_side 时等比缩小，预览用。"""
    if image.isNull():
        return image
    if max(image.width(), image.height()) <= max_side:
        return image
    return image.scaled(max_side, max_side,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)


def selection_mask_from_points(points: List[Tuple[float, float]],
                               width: int, height: int,
                               mode: str) -> QImage:
    """归一化选区点 → 灰度蒙版，白色为选区内。

    points 少于 2 个（单击未拖动）时返回全黑，处理链退化为不生效。
    """
    mask = QImage(width, height, QImage.Format_Grayscale8)
    mask.fill(0)
    if not points or len(points) < 2:
        return mask
    pts = [(_clamp01(p[0]) * width, _clamp01(p[1]) * height) for p in points]
    painter = QPainter(mask)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.setPen(QPen(QColor(255, 255, 255), 1))
    painter.setBrush(QColor(255, 255, 255))
    if mode in ("rect", "ellipse"):
        x0, x1 = sorted((pts[0][0], pts[1][0]))
        y0, y1 = sorted((pts[0][1], pts[1][1]))
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)
        if mode == "rect":
            painter.drawRect(rect)
        else:
            painter.drawEllipse(rect)
    else:
        painter.drawPolygon(QPolygonF([QPointF(p[0], p[1]) for p in pts]))
    painter.end()
    return mask


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))
