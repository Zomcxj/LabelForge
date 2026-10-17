"""视频画面上的标注叠加绘制（④A/④B 共用）。

为什么要单独一个模块：时间轴、清单、编码这些逻辑不该依赖 Qt，而这个
模块必须用 Qt——`cv2.putText` 画不出中文，只能 `QImage`/`QPainter` 离屏。

两条硬约束：
- **不在模块级导入 Qt**，让不需要绘制的路径（清单校验、纯逻辑测试）
  能在无 Qt 环境下导入。
- **颜色必须由调用方预先取好传进来**：`editor.get_label_color()` 会写
  缓存和配置，后台线程不能碰它，主线程先算好 `label_colors` 再交给 worker。

坐标约定：所有 `points`/`x,y,width,height` 都是**原图像素坐标**，与画布
和 labelme 一致，不做缩放。框可能来自推理（越界）或人工（退化尺寸），
都必须能画，不能抛异常。
"""
from dataclasses import dataclass, field

import numpy as np

# 线宽与字号按画面高度缩放：480p 线宽 1、1080p 约 2、2160p 约 5
PEN_HEIGHT_DIVISOR = 480
FONT_HEIGHT_DIVISOR = 40
MIN_PEN_WIDTH = 1
MAX_PEN_WIDTH = 6
MIN_FONT_PIXELS = 10
MAX_FONT_PIXELS = 72

# 类别没配颜色时的兜底（与画布默认色系一致）
DEFAULT_COLOR = (0, 255, 128)
# 坐标钳制范围：够大不会切掉有效几何，又不至于让 Qt 光栅化溢出
COORD_LIMIT = 32000.0


@dataclass
class RenderSpec:
    """绘制规格。

    `labels` 为 None 表示画全部类别，否则是勾选的类别名集合。
    `show_labels` / `show_scores` 决定框上文字是否带类别名与置信度。
    `label_colors` 由主线程预先算好，键是类别名，值是 `#RRGGBB`。
    """
    labels: object = None
    show_labels: bool = False
    show_scores: bool = False
    label_colors: dict = field(default_factory=dict)
    pen_width: int = 0          # 0 = 按画面高度自动
    font_pixels: int = 0        # 0 = 按画面高度自动


def filter_boxes(boxes, labels):
    """按勾选的类别筛选；`labels` 为 None 时全画。"""
    if labels is None:
        return list(boxes)
    allowed = {str(name) for name in labels}
    return [box for box in boxes
            if str(box.get("label") or "") in allowed]


def _parse_hex(value):
    text = str(value or "").strip()
    if len(text) != 7 or not text.startswith("#"):
        return None
    try:
        return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
    except ValueError:
        return None


def color_for_label(label_colors, label):
    """类别色；没配或颜色串损坏时用兜底色。"""
    found = _parse_hex((label_colors or {}).get(str(label or "")))
    return found if found is not None else DEFAULT_COLOR


def label_text(box, spec):
    """框上文字；不开「类别名」时返回空串（一个字都不画）。"""
    if not spec.show_labels:
        return ""
    from .shape_io import box_display_label
    return box_display_label(box, show_score=bool(spec.show_scores))


def _auto_pen_width(height):
    value = int(round(height / PEN_HEIGHT_DIVISOR))
    return max(MIN_PEN_WIDTH, min(MAX_PEN_WIDTH, value))


def _auto_font_pixels(height):
    value = int(round(height / FONT_HEIGHT_DIVISOR))
    return max(MIN_FONT_PIXELS, min(MAX_FONT_PIXELS, value))


# 字号滑块的量程。**0 表示自动**（按画面高度算），1..MAX 是手动像素值。
# 上限取 MAX_FONT_PIXELS：自动档的天花板已经是 72，手动再往上放只会糊。
FONT_SLIDER_MIN = 0
FONT_SLIDER_MAX = MAX_FONT_PIXELS


def label_font_pixels(height):
    """视频里的字号：**跟设置里的「类别名字号」走**，与画布观感对齐。

    设置窗口给的是 5..15 的抽象字号（`DETECTION_BOX_CONFIG['label_font_size']`，
    默认 9）。画布那边把它当**点**用（`QFont.setPointSize`），视频这边为了
    分辨率无关用的是像素（`setPixelSize`），所以不能把这个数直接搬过来——
    1080p 上「9 px」小得看不见。

    换算口径：1 点 ≈ 1.333 像素（96 DPI / 72）。于是设置 9 → 12 px，正好等于
    原来的自动值（1080p 高度 /40 = 27；这里按点换算更保守，避免字幕过粗）。
    再乘上「画面高度 / 1080」让 480p 和 4K 的观感一致，最后夹进引擎的上下限。
    """
    from ..core.config import DETECTION_BOX_CONFIG

    points = max(5, min(15, DETECTION_BOX_CONFIG.get('label_font_size', 9)))
    pixels = points * 4 / 3                      # 点 → 像素
    scaled = pixels * (max(1, int(height)) / 1080.0)
    return int(max(MIN_FONT_PIXELS, min(MAX_FONT_PIXELS, round(scaled))))


def resolve_font_pixels(spec, height):
    """这份 spec 实际会用的字号（像素）。

    `spec.font_pixels` 非 0 时用它（留给调用方显式指定）；为 0 时跟设置里的
    「类别名字号」走，见 `label_font_pixels()`。成片与预览共用这一条路径。
    """
    return int(spec.font_pixels) or label_font_pixels(height)


def _clamp_points(points):
    return [(max(-COORD_LIMIT, min(COORD_LIMIT, float(p[0]))),
             max(-COORD_LIMIT, min(COORD_LIMIT, float(p[1]))))
            for p in points]


def _rect_points(box):
    x = float(box.get("x") or 0.0)
    y = float(box.get("y") or 0.0)
    w = float(box.get("width") or 0.0)
    h = float(box.get("height") or 0.0)
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _buffer(qimage, writable):
    """QImage 像素缓冲的 numpy 视图。

    PySide6 的 `bits()` / `constBits()` 直接返回带长度的 `memoryview`，
    `np.frombuffer` 可以直接用；PyQt5 那套「先 `setsize` 再 frombuffer」
    在这里不存在（`memoryview` 没有 `setsize`）。

    **返回的视图不持有 `qimage`**（实测 `memoryview.obj is None`）：调用方必须
    在视图存活期间自己保活那张 QImage，否则内存回收后读到随机像素、且不报错。
    本文件的使用者都在同一函数内用完，别再让视图外泄。
    """
    pointer = qimage.bits() if writable else qimage.constBits()
    return np.frombuffer(pointer, np.uint8)


def _copy_into_qimage(qimage, canvas):
    """把 HxWx3 连续 RGB 拷贝进 QImage 自己的内存。

    `QImage(canvas.data, ...)` 是**零拷贝**：QImage 只是包住那块外部缓冲，
    像素所有权不在 QImage 手上。这里显式拷进去、画完再读回结果，让像素
    归属 QImage，不依赖调用方是否保活那块数组。

    行步长 `bytesPerLine` 可能有尾部补齐，逐行只拷贝有效像素。
    """
    height, width = canvas.shape[:2]
    target = _buffer(qimage, writable=True).reshape(
        height, qimage.bytesPerLine())
    target[:, :width * 3] = canvas.reshape(height, width * 3)


def _read_back(qimage, width, height):
    """从 QImage 读回连续 HxWx3 RGB 数组。

    必须**真的拷贝**：`ascontiguousarray` 碰到本就连续的切片会原样返回视图，
    那样返回的数组指着 QImage 的内存，QImage 一析构（函数返回时）就悬空，
    调用方读到的就是被复用的垃圾内存——表现为随机花屏、随机丢框。
    """
    raw = _buffer(qimage, writable=False).reshape(
        height, qimage.bytesPerLine())
    return raw[:, :width * 3].copy().reshape(height, width, 3)


def _box_anchor(box):
    """文字锚点：矩形左上，其它形状用外接盒左上。"""
    points = _clamp_points(box.get("points") or [])
    if points:
        x = min(p[0] for p in points)
        y = min(p[1] for p in points)
        return x, y
    return float(box.get("x") or 0.0), float(box.get("y") or 0.0)


def _draw_label(painter, anchor_x, anchor_y, text, bg_color, font):
    """在框上方画标签条；顶到画面外就改画在框内侧。

    必须 `painter.setFont(font)`：条子尺寸是拿 `QFontMetrics(font)` 量的，
    不设字体的话 `drawText` 会退回 painter 当前字体（应用字体），
    字比条小一大截——1080p 下条子 27px 高、字只有十几像素。
    画布那条路（`_draw_label_above_rect`）设了字体，只有这里漏了。
    """
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFontMetrics

    metrics = QFontMetrics(font)
    text_width = metrics.horizontalAdvance(text)
    text_height = metrics.height()
    pad = max(2, int(font.pixelSize() * 0.2))
    top = anchor_y - text_height
    if top < 0:
        top = anchor_y
    rect = QRectF(anchor_x, top, text_width + pad * 2, text_height)
    painter.fillRect(rect, QColor(*bg_color))
    painter.setPen(QColor(0, 0, 0))
    painter.setFont(font)
    painter.drawText(rect.adjusted(pad, 0, -pad, 0),
                     Qt.AlignLeft | Qt.AlignVCenter, text)


def draw_boxes(image_rgb, boxes, spec=None):
    """在 RGB 数组上叠加标注，返回**新**数组（输入不被修改）。

    通道顺序必须是 RGB：调用方从 cv2 拿到的是 BGR，要先 `cvtColor`
    再传进来，否则颜色会红蓝颠倒。
    """
    spec = spec or RenderSpec()
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (QColor, QFont, QImage, QPainter, QPen,
                             QPolygonF)

    # 输入必须转成连续 uint8 副本：非连续数组（切片视图）行步长对不上，
    # 而且绘制会改像素，不能动调用方的原数组。
    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"只支持 HxWx3 的 RGB 图：{array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"只支持 uint8：{array.dtype}")

    canvas = np.ascontiguousarray(array, dtype=np.uint8)
    height, width = canvas.shape[:2]

    selected = filter_boxes(boxes or [], spec.labels)
    if not selected:
        return canvas.copy()

    pen_width = int(spec.pen_width) or _auto_pen_width(height)
    font_pixels = resolve_font_pixels(spec, height)
    font = QFont()
    font.setPixelSize(font_pixels)

    qimage = QImage(width, height, QImage.Format_RGB888)
    _copy_into_qimage(qimage, canvas)
    painter = QPainter(qimage)
    try:
        # 关抗锯齿：边线是精确的类别色（不被背景混色），几千帧也更快
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setBrush(Qt.NoBrush)
        # 奇数线宽靠半像素平移对齐到整像素，避免边线糊成两行
        if pen_width % 2 == 1:
            painter.translate(0.5, 0.5)

        for box in selected:
            color = color_for_label(spec.label_colors,
                                    box.get("label") if isinstance(box, dict) else "")
            pen = QPen(QColor(*color), pen_width)
            painter.setPen(pen)

            shape_type = (box.get("shape_type") or "rectangle") \
                if isinstance(box, dict) else "rectangle"
            if shape_type == "point":
                points = _clamp_points(box.get("points")
                                       or [(box.get("x") or 0.0, box.get("y") or 0.0)])
                size = max(3, pen_width * 3)
                painter.setBrush(QColor(*color))
                half = size / 2.0
                painter.drawRect(QRectF(points[0][0] - half,
                                        points[0][1] - half, size, size))
                painter.setBrush(Qt.NoBrush)
                anchor = (points[0][0], points[0][1] - half)
            elif shape_type in ("polygon", "rotation", "linestrip"):
                points = _clamp_points(box.get("points") or [])
                if len(points) < 2:
                    continue
                polygon = QPolygonF([QPointF(x, y) for x, y in points])
                if shape_type == "linestrip":
                    painter.drawPolyline(polygon)
                else:
                    painter.drawPolygon(polygon)
                anchor = (min(p[0] for p in points), min(p[1] for p in points))
            else:
                points = _clamp_points(_rect_points(box))
                if points[2][0] <= points[0][0] and points[2][1] <= points[1][1]:
                    continue
                polygon = QPolygonF([QPointF(x, y) for x, y in points])
                painter.drawPolygon(polygon)
                anchor = (points[0][0], points[0][1])

            text = label_text(box, spec) if isinstance(box, dict) else ""
            if text:
                _draw_label(painter, anchor[0], anchor[1], text, color, font)
    finally:
        painter.end()

    return _read_back(qimage, width, height)
