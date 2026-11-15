"""贴图落盘用的位图转换：纯 numpy，不依赖 Qt 之外的东西。

贴图按 (H, W, 4) 的 RGBA uint8 理解，背景按 (H, W, 3) 的 BGR uint8 理解。
"""

import numpy as np


def pixmap_to_rgba(pixmap, width, height):
    """QPixmap -> (H, W, 4) uint8 RGBA，按目标尺寸缩放。"""
    from PySide6.QtGui import QImage
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    if width > 0 and height > 0 and (image.width() != width or image.height() != height):
        image = image.scaled(width, height)
    arr = np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 4]
    return arr.reshape(image.height(), image.width(), 4).copy()


def qimage_to_bgr(image):
    """QImage -> (H, W, 3) uint8 BGR。"""
    from PySide6.QtGui import QImage
    image = image.convertToFormat(QImage.Format_RGB888)
    arr = np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 3]
    return arr.reshape(image.height(), image.width(), 3)[:, :, ::-1].copy()


def bgr_to_qimage(bgr):
    """(H, W, 3) uint8 BGR -> QImage（Format_RGB888）。"""
    from PySide6.QtGui import QImage
    rgb = np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()


def rgba_to_qimage(rgba):
    """(H, W, 4) uint8 RGBA -> QImage。"""
    from PySide6.QtGui import QImage
    h, w = rgba.shape[:2]
    buf = np.ascontiguousarray(rgba).tobytes()
    return QImage(buf, w, h, w * 4, QImage.Format_RGBA8888).copy()


def rgba_to_pixmap(rgba):
    """(H, W, 4) uint8 RGBA -> QPixmap。"""
    from PySide6.QtGui import QPixmap
    return QPixmap.fromImage(rgba_to_qimage(rgba))
