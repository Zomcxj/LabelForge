"""原图缩略 + 滑窗网格叠加预览。

与 `crop_dialog` 的分离小图预览**不同**：推理侧要看的是「窗口落在原图哪里」，
分离小图在大图/多窗时无法辨认。窗口起点直接复用 `window_starts()`，
与引擎枚举同源，保证画面上的网格就是要跑的窗口。

非法参数的口径与引擎一致：能不能跑由 `iter_windows()` 说了算，不在预览里
自造规则。特别地，**重叠 >= 窗口尺寸不算错**——引擎的 `normalized_window()`
把它钳到 `min(cw,ch)-1`；真正会失败的是尺寸越界与窗口数超上限
（`MAX_WINDOWS`）。两者都表现为「参数不可用」，预览给文字提示而不是抛异常。

大图保护：窗口数可能上万（20000² 配小窗口），逐窗全量绘制会卡死，
所以每个窗口先换算成画布矩形、与可见区求交，完全在外的一律跳过。
"""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

from ...engine.augmenter.crop import window_starts
from ...engine.inference_scope import ScopeError, iter_windows
from ..i18n import t as tr
from ..theme import ThemeManager

_MARGIN = 8


class WindowOverlayPreview(QWidget):
    """原图缩略上叠加滑窗网格；参数非法或无图时给文字提示。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_size = None
        self._spec = None
        self._pixmap = None
        self.setMinimumHeight(160)

    # ---------- 对外接口 ----------

    def set_image(self, size):
        """参考图尺寸 (w, h)；None 表示没有图。"""
        self._image_size = tuple(size) if size else None
        self.update()

    def set_pixmap(self, pixmap):
        """当前背景图；会在网格底下画出来，让用户看清窗口落在画面哪里。

        传 None 或空 pixmap 时退回纯色占位（仍画网格）——预览是调参参照，
        缺图不该让面板报错或不显示。
        """
        if pixmap is not None and hasattr(pixmap, 'isNull') and pixmap.isNull():
            pixmap = None
        self._pixmap = pixmap
        if pixmap is not None:
            # 图本身比尺寸更权威：用它同步尺寸，避免「图是 A、网格按 B 算」
            self._image_size = (pixmap.width(), pixmap.height())
        self.update()

    def set_spec(self, spec):
        """窗口参数 {w, h, overlap}；None 表示未配置。"""
        self._spec = dict(spec) if spec else None
        self.update()

    def is_valid(self):
        return self.window_count() is not None

    def window_count(self):
        """实际窗口数；没图/参数非法时 None。"""
        windows = self._window_rects()
        return None if windows is None else len(windows)

    # ---------- 内部 ----------

    def _window_rects(self):
        """原图像素坐标下的窗口矩形列表；不可用时 None。

        用 `iter_windows()` 取窗口起点而不是自己排布：它与引擎、与「窗口数
        预估」文字同源，三处不会漂移。`iter_windows()` 内部已把重叠钳到
        合法范围、并在窗口数超上限时抛错——所以这里只需把异常收成 None。
        """
        if not self._image_size or not self._spec:
            return None
        try:
            iw = int(self._image_size[0])
            ih = int(self._image_size[1])
        except (TypeError, ValueError, IndexError):
            return None
        if iw <= 0 or ih <= 0:
            return None
        try:
            windows = iter_windows(iw, ih, self._spec)
        except ScopeError:
            return None
        return [(wx, wy, ww, wh) for wx, wy, ww, wh in windows]

    def _image_scale(self):
        """原图 → 控件的等比缩放与居中偏移；不可用时 None。"""
        if not self._image_size:
            return None
        try:
            iw = int(self._image_size[0])
            ih = int(self._image_size[1])
        except (TypeError, ValueError, IndexError):
            return None
        if iw <= 0 or ih <= 0:
            return None
        avail_w = max(1, self.width() - _MARGIN * 2)
        avail_h = max(1, self.height() - _MARGIN * 2)
        scale = min(avail_w / iw, avail_h / ih)
        draw_w, draw_h = iw * scale, ih * scale
        return scale, (self.width() - draw_w) / 2, (self.height() - draw_h) / 2

    def paintEvent(self, event):
        theme = ThemeManager.get_theme()
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor(theme['canvas_bg']))
            mapping = self._image_scale()
            if mapping is None:
                self._draw_center_text(painter, tr("未加载图片，无法预估窗口数"))
                return
            scale, ox, oy = mapping
            iw, ih = int(self._image_size[0]), int(self._image_size[1])
            target = QRectF(ox, oy, iw * scale, ih * scale)
            if self._pixmap is not None:
                # 先画原图再叠网格：用户要看的正是「窗口落在原图的哪里」，
                # 只有格子没有画面的话，落在人脸还是天空上完全看不出来。
                painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
            else:
                painter.fillRect(target, QColor(theme['panel_bg']))
            windows = self._window_rects()
            if windows is None:
                self._draw_center_text(painter, tr("窗口参数非法，请检查尺寸与重叠"))
                return
            self._draw_grid(painter, windows, scale, ox, oy, theme)
        finally:
            if painter.isActive():
                painter.end()

    def _draw_grid(self, painter, windows, scale, ox, oy, theme):
        """只画与控件可见区相交的窗口；全量绘制在万级窗口下会卡死。"""
        color = QColor(theme['accent'])
        pen = QPen(color, 1, Qt.DashLine)
        painter.setPen(pen)
        visible = QRectF(self.rect())
        for (x, y, w, h) in windows:
            rect = QRectF(ox + x * scale, oy + y * scale, w * scale, h * scale)
            if not rect.intersects(visible):
                continue
            painter.drawRect(rect)

    def _draw_center_text(self, painter, text):
        painter.setPen(QPen(QColor(ThemeManager.get_theme()['text_secondary'])))
        painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, text)
