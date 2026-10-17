"""QImage 与推理内核之间的桥接：取图与 letterbox 绘制。

单独成模块是为了让 onnx_inference 保持无 Qt 依赖，便于纯计算测试；
同时把「显示缓冲 → 模型输入」的通道与步长处理集中在一处。
"""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage


def qimage_to_rgb_array(image):
    """QImage → (H, W, 3) uint8 RGB 数组。

    显示缓冲是 ARGB32（小端内存为 BGRA），必须显式转换并处理 bytesPerLine：
    Qt 每行有 4 字节对齐填充，直接按 width*4 读会逐行错位。
    """
    if image is None or image.isNull():
        return None
    return _rgb888_rows(image, image.width(), image.height())


def letterbox_to_array(rgb, geometry, fill):
    """把 RGB 数组等比缩放并居中补边到模型输入尺寸，返回 (dst_h, dst_w, 3) uint8。

    与 letterbox_geometry 共用同一组整数（new_w/new_h/pad_x/pad_y），
    避免绘制与逆变换各自取整造成坐标偏移。
    """
    src_h, src_w = rgb.shape[0], rgb.shape[1]
    if (src_w, src_h) == (geometry.new_w, geometry.new_h):
        scaled_rgb = np.ascontiguousarray(rgb)
    else:
        # QImage 按 src_w*3 的步长读内存，非连续数组必须先规整
        src = np.ascontiguousarray(rgb)
        img = QImage(src.data, src_w, src_h, src_w * 3, QImage.Format_RGB888)
        scaled = img.scaled(geometry.new_w, geometry.new_h,
                            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        scaled_rgb = _rgb888_rows(scaled, geometry.new_w, geometry.new_h)

    canvas = np.full((geometry.dst_h, geometry.dst_w, 3), int(fill), dtype=np.uint8)
    canvas[geometry.pad_y:geometry.pad_y + geometry.new_h,
           geometry.pad_x:geometry.pad_x + geometry.new_w] = scaled_rgb
    return canvas


def _rgb888_rows(image, width, height):
    """取 Format_RGB888 图的像素，剥掉每行 4 字节对齐的填充。"""
    rgb = image.convertToFormat(QImage.Format_RGB888)
    stride = rgb.bytesPerLine()
    buf = rgb.constBits()
    rows = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(height, stride)
    return np.ascontiguousarray(rows[:, : width * 3].reshape(height, width, 3))


def resize_for_model(rgb, geometry, fill):
    """OnnxDetector.detect 的 resize_fn 实现。"""
    return letterbox_to_array(rgb, geometry, fill)
