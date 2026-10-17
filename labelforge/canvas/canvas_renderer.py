"""
Canvas 绘制混入 - 负责所有绘制逻辑（背景、贴图、检测框、临时框、网格）
"""
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QFont, QFontMetrics
from PySide6.QtCore import Qt, QPointF, QRectF
import numpy as np

from ..core.config import (
    DETECTION_BOX_CONFIG, PASTE_ITEM_CONFIG, GRID_CONFIG, MAGNIFIER_CONFIG,
    CROSSHAIR_CONFIG, REGION_HANDLE_EXTRA,
)
from ..engine.inference_scope import ScopeError, iter_windows_with_index
from ..ui.theme import ThemeManager

class CanvasRendererMixin:
    """Canvas 绘制混入类 - paintEvent 及所有 _draw_* 方法"""

    # 画布上给窗口标 `r1c1` 的可见格子数上限。超过就只画框不标号：逐格
    # drawText 比 drawRect 贵得多，大图+小窗（上万格）时会把重绘拖慢。
    _SCOPE_GRID_LABEL_LIMIT = 300

    def paintEvent(self, event):
        """绘制事件

        paintEvent 内未捕获的异常会让 Qt 直接终止进程（0xC0000409，无
        traceback），所以整段绘制都包在兜底里：失败只降级为一条日志和
        一个错误提示，应用继续运行。
        """
        try:
            self._paint_scene(event)
        except Exception as exc:
            self._report_paint_failure(exc)

    def _paint_scene(self, event):
        """实际绘制流程；异常由 paintEvent 兜底。

        每个 QPainter 都必须用 finally 收尾：异常时 traceback 会持有本帧，
        未 end() 的 painter 会一直存活到 except 块结束，届时 Qt 在
        render() 上下文里析构它会直接段错误（exit 139），而不是抛异常。
        """
        # 所有切图路径最终都会重绘，亮度/对比度在此单点重新套用
        self.apply_display_adjustments()

        scene = QPixmap(self.size())
        scene.fill(Qt.transparent)
        sp = QPainter(scene)
        background_rect = None
        try:
            sp.setRenderHint(QPainter.Antialiasing)

            t = ThemeManager.get_theme()
            bg_color = t['canvas_bg']
            r = int(bg_color[1:3], 16)
            g = int(bg_color[3:5], 16)
            b = int(bg_color[5:7], 16)
            sp.fillRect(self.rect(), QColor(r, g, b))

            background_rect = self.get_background_rect()

            if self._editor.current_background is not None and background_rect:
                self._draw_background(sp, background_rect)

            if background_rect:
                self._draw_grid(sp, background_rect)

            # 滑窗网格直接叠在主画布上（没有单独的小预览窗）。画在区域之前、
            # 缩小框之前：网格是「范围参照」，不该盖住标注框。
            if background_rect and self._show_scope_grid():
                self._draw_scope_grid(sp, background_rect)

            sp.setOpacity(self.shape_opacity)

            # 区域分**两趟**画，检测框夹在中间（用户要求：未锁区域在「最上层」、
            # 锁定的在「最底层」）：
            #   锁定的区域  → 检测框 → 贴图项 → 未锁定的区域
            # 这样未锁区域压住框、锁定的被框压住，与点击优先级一一对应
            # （见 `canvas_menu._region_at()` 跳过锁定的区域）。
            if background_rect and self._show_regions():
                self._draw_regions(sp, background_rect, locked_only=True)

            if background_rect:
                self._draw_paste_items(sp, background_rect)

            if (self._editor.show_labels_checkbox.isChecked() and
                background_rect and self._editor.detection_boxes):
                self._draw_detection_boxes(sp, background_rect)

            if background_rect and self._show_regions():
                self._draw_regions(sp, background_rect, locked_only=False)

            if self.is_drawing_box:
                self._draw_temp_box(sp)
            if getattr(self, 'is_drawing_region', False) and self._region_draw_visible():
                self._draw_temp_region_box(sp)
            if getattr(self, 'is_drawing_polygon', False):
                self._draw_temp_polygon(sp, background_rect)
            if getattr(self, 'is_drawing_region_polygon', False) and self._region_draw_visible():
                self._draw_temp_region_polygon(sp, background_rect)

            sp.setOpacity(1.0)

            if (getattr(self._editor, 'edit_mode', 'paste') == 'annotate' and
                    self.mouse_inside and self._editor.current_background is not None and
                    background_rect is not None):
                self._draw_crosshair(sp)
        finally:
            if sp.isActive():
                sp.end()

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawPixmap(0, 0, scene)

            if background_rect:
                self._draw_magnifier(painter, background_rect, scene)
        finally:
            if painter.isActive():
                painter.end()

    def _report_paint_failure(self, exc):
        """绘制失败时的降级处理：记日志，绝不再抛异常。

        这里刻意不新建 QPainter：本方法可能在 render() 的上下文里被调用，
        此时再往 widget 上开 painter 会段错误。失败就只留日志。
        """
        try:
            from ..core.exception_hook import _write_log
            _write_log(f"画布绘制失败，已跳过本帧: {type(exc).__name__}: {exc}")
        except Exception:
            pass

    def _draw_background(self, painter, background_rect):
        """绘制背景图"""
        painter.drawPixmap(
            int(background_rect.left()),
            int(background_rect.top()),
            int(background_rect.width()),
            int(background_rect.height()),
            self._editor.current_background
        )

    def _draw_grid(self, painter, background_rect):
        """绘制网格参考线"""
        if not self._editor.show_grid_checkbox.isChecked():
            return

        t = ThemeManager.get_theme()
        mode = ThemeManager.get_mode().value
        color_map = {'light': GRID_CONFIG['color_light'],
                     'dark': GRID_CONFIG['color_dark'],
                     'ink': GRID_CONFIG['color_ink']}
        grid_color = color_map.get(mode, GRID_CONFIG['color_light'])

        r = int(grid_color[1:3], 16)
        g = int(grid_color[3:5], 16)
        b = int(grid_color[5:7], 16)

        pen = QPen(QColor(r, g, b, GRID_CONFIG.get('alpha', 120)))
        pen.setWidth(GRID_CONFIG.get('line_width', 1))
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)

        spacing = GRID_CONFIG['spacing']
        scaled_spacing = spacing * self.background_scale

        if scaled_spacing < 5:
            return

        x = background_rect.left()
        while x <= background_rect.right():
            painter.drawLine(int(x), int(background_rect.top()),
                           int(x), int(background_rect.bottom()))
            x += scaled_spacing

        y = background_rect.top()
        while y <= background_rect.bottom():
            painter.drawLine(int(background_rect.left()), int(y),
                           int(background_rect.right()), int(y))
            y += scaled_spacing

    def _show_scope_grid(self):
        """是否在主画布上画滑窗网格。

        条件：范围面板开着**且**当前选的是「滑窗」。用户要求「滑窗和区域各自
        的内容直接显示在画布上」——所以切到区域时网格就该消失，只剩区域框；
        关掉面板则两者都不显示（它只是调参时的参照）。
        """
        if not getattr(self._editor, '_scope_panel_open', False):
            return False
        from ..engine.inference_scope import MODE_WINDOW
        return getattr(self._editor, '_scope_mode', None) == MODE_WINDOW

    def _draw_scope_grid(self, painter, background_rect):
        """把滑窗的窗口网格画在主画布上（换算到背景图坐标）。

        每个窗口的**左下角**标出行列号 `r1c1`，与数据增强预览里标的号
        同源同格式（用户要求「滑窗在画布上左下显示 r1c1 同数据增强中的」），
        所以这里用 `iter_windows_with_index()` 而不是自己排行列。

        窗口起点复用引擎的窗口数学，与「每张图 X × Y = N 个窗口」的文字
        同源，画的格子就是要跑的窗口。参数非法时什么都不画（不是画一半），
        否则用户会以为「这些窗会跑」。
        """
        spec = getattr(self._editor, '_scope_window_spec', None)
        if not spec:
            return
        iw, ih = self._scope_grid_reference_size()
        if iw <= 0 or ih <= 0:
            return
        try:
            windows = iter_windows_with_index(iw, ih, spec)
        except ScopeError:
            return

        from ..core.config import REGION_CONFIG
        color = QColor(REGION_CONFIG['border_color'])
        color.setAlpha(200)
        pen = QPen(color, 1, Qt.DashLine)
        pen.setDashPattern(REGION_CONFIG['dash_pattern'])
        painter.save()
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        # 窗口上万时逐格标字会把重绘拖垮，所以先按视口过滤出要画的格子，
        # 再决定字要不要标（格子太小/太多时只画框，不标号）。
        visible = QRectF(self.rect())
        to_draw = []
        for (x, y, w, h, r, c) in windows:
            rect = QRectF(
                background_rect.left() + x * self.background_scale,
                background_rect.top() + y * self.background_scale,
                w * self.background_scale,
                h * self.background_scale)
            # 大图 + 小窗口会有上万格，逐格全画会卡死；视口外的直接跳过
            if not rect.intersects(visible):
                continue
            to_draw.append((rect, r, c))
            painter.drawRect(rect)

        # 标号门槛与数据增强预览一致（那边是 44x30 才标）：格子太小标了也
        # 看不清，只会糊成一片。另外总的可见格子太多时也不标，避免重绘变慢。
        if len(to_draw) <= self._SCOPE_GRID_LABEL_LIMIT:
            font = QFont(painter.font())
            font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
            painter.setFont(font)
            painter.setPen(QColor(REGION_CONFIG['border_color']))
            for rect, r, c in to_draw:
                if rect.width() < 44 or rect.height() < 30:
                    continue
                painter.drawText(rect.adjusted(2, 0, -2, -2),
                                 Qt.AlignBottom | Qt.AlignLeft, f"r{r}c{c}")
        painter.restore()

    def _scope_grid_reference_size(self):
        """窗口网格按哪个尺寸切：当前背景图的**原始像素**尺寸。

        用原来的图像尺寸而不是画布尺寸：窗口是原图上的裁剪范围，按画布
        （已缩放）算会让格子数目与真实要跑的窗口数对不上。
        """
        pixmap = getattr(self._editor, 'current_background', None)
        if pixmap is not None and not pixmap.isNull():
            return pixmap.width(), pixmap.height()
        size = getattr(self._editor, '_scope_reference_size', None)
        if size:
            return int(size[0]), int(size[1])
        return 0, 0

    def _is_paste_mode(self):
        """区域仅在贴图模式显示/编辑。"""
        return (getattr(self._editor, 'edit_mode', 'paste') == 'paste'
                and not getattr(self._editor, '_is_delete_view', False))

    def _region_draw_visible(self):
        """临时区域框是否该画：贴图模式，或**区域绘制态**中（含面板开着时）。

        面板开着时在标注模式画区域也要看得见，否则用户看不到自己在画什么。
        只影响「画不画临时框」，不改交互。
        """
        return self._is_paste_mode() or bool(
            getattr(self, 'is_drawing_region', False)
            or getattr(self, 'is_drawing_region_polygon', False))

    def _show_regions(self):
        """是否画区域轮廓：贴图模式，或推理范围选了区域。

        推理用区域时要在标注模式也看得见范围，否则用户不知道框为什么只出现在
        某块地方。编辑仍然只在贴图模式（`_is_paste_mode()` 不动，免得影响
        贴图交互）。
        """
        return self._is_paste_mode() or bool(
            getattr(self._editor, '_show_scope_regions', False))

    def _draw_regions(self, painter, background_rect, locked_only=None):
        """绘制语义区域，支持矩形与多边形。

        `locked_only`：`True` 只画**已锁定**的、`False` 只画**未锁定**的、
        `None` 两趟都画（老调用方/测试用）。分两趟是为了让锁定的区域落在
        检测框之下、未锁定的压在其上——层级要和点击优先级一致，否则「看着
        在上面、点下去却是别的框」。
        """
        regions = self._active_regions()
        if not regions:
            return
        from ..core.config import REGION_CONFIG
        color = QColor(REGION_CONFIG['border_color'])
        fixed = self._active_region_fixed()

        painter.save()
        for i, region in enumerate(regions):
            if locked_only is not None and bool(region.get('locked')) != bool(locked_only):
                continue
            is_selected = (i == self.selected_region and not fixed)
            pen = self._get_region_border_pen(color, is_selected)
            pen.setStyle(Qt.DashLine)
            pen.setDashPattern(REGION_CONFIG['dash_pattern'])
            painter.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(REGION_CONFIG['fill_alpha'])
            painter.setBrush(fill if is_selected else Qt.NoBrush)

            if (region.get('shape_type') or 'rectangle') == 'polygon' and region.get('points'):
                self._draw_region_polygon(painter, region, background_rect, color, is_selected)
                continue

            x = region['x'] * self.background_scale + background_rect.left()
            y = region['y'] * self.background_scale + background_rect.top()
            w = region['width'] * self.background_scale
            h = region['height'] * self.background_scale
            painter.drawRect(QRectF(x, y, w, h))

            if is_selected:
                self._draw_region_handles(painter, x, y, w, h, color)
        painter.restore()

    def _draw_region_polygon(self, painter, region, background_rect, color, is_selected):
        """绘制多边形区域；选中时显示顶点圆点手柄。"""
        from ..core.config import REGION_CONFIG
        from PySide6.QtGui import QPainterPath
        pts = self._image_to_canvas_points(region['points'], background_rect)
        if len(pts) < 2:
            return
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        painter.drawPath(path)

        if is_selected:
            size = REGION_CONFIG['handle_size']
            radius = size / 2
            painter.setPen(QPen(color, 1))
            painter.setBrush(QColor(255, 255, 255))
            for p in pts:
                painter.drawEllipse(p, radius, radius)

    def _draw_region_handles(self, painter, x, y, w, h, color):
        """区域缩放手柄：**只画四角**，与检测框完全同一套画法。

        用户第 11 轮：「推理范围内的区域编辑要和检测框的编辑一致，检测框选中
        句柄会变成正方形」——所以这里不再自己画一套白圆圈，改成与
        `_draw_box_handles()` 一模一样的两态：

        - **悬停**：白色正方形（`hover_region_handle` 命中的那一个）
        - **常态**：白色底圆减去区域内部，再叠一个区域色的实心圆

        **只画四个角**：检测框 `_draw_box_handles()` 的 `corners` 就是四个角，
        边中点一个都不画——这是逐像素核对过的，不是推断。四边中点虽然仍能
        抓（`_region_handle_at()` 认八处），但**不画**，与检测框保持一致。

        （第 11 轮曾错画成八个，用户当场纠正：「检测框也只有四个点」。）

        半径用 `_region_handle_draw_size()`：与命中判定同一套随缩放放大的
        口径，保证「看到的圆」就是「抓得到的圆」。
        """
        from PySide6.QtGui import QPainterPath
        size = self._region_handle_draw_size()
        radius = size / 2
        hovered = getattr(self, 'hover_region_handle', None)
        hovered_index = getattr(self, 'hover_region_index', None)
        is_hover_region = (hovered_index is not None
                           and hovered_index == getattr(self, 'selected_region', None))

        corners = (
            ('tl', QPointF(x, y)),
            ('tr', QPointF(x + w, y)),
            ('bl', QPointF(x, y + h)),
            ('br', QPointF(x + w, y + h)),
        )

        painter.save()
        for name, pos in corners:
            if hovered and hovered == name and is_hover_region:
                painter.setPen(QPen(color, 2))
                painter.setBrush(QColor(255, 255, 255))
                painter.drawRect(QRectF(pos.x() - radius, pos.y() - radius,
                                        size, size))
            else:
                # 1. 白色底层——减去区域内部，只留区域外那半圈
                circle = QPainterPath()
                circle.addEllipse(pos, radius * 1.3, radius * 1.3)
                region_path = QPainterPath()
                region_path.addRect(QRectF(x, y, w, h))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255))
                painter.drawPath(circle.subtracted(region_path))
                # 2. 常态覆盖（区域色实心圆）
                painter.setPen(QPen(color, 1))
                painter.setBrush(color)
                painter.drawEllipse(pos, radius, radius)
        painter.restore()

    def _region_handle_draw_size(self):
        """手柄的绘制边长（画布像素）。

        真源在 `canvas_menu._region_handle_size()`（与命中判定同一个类，避免
        两处各算一遍）。这里转调；万一那个 mixin 不在（早期测试用的精简
        Canvas），退回「检测框 + REGION_HANDLE_EXTRA」的同一口径。
        """
        shared = getattr(self, '_region_handle_size', None)
        if callable(shared):
            return shared()
        from ..core.config import DETECTION_BOX_CONFIG, REGION_HANDLE_EXTRA
        return DETECTION_BOX_CONFIG['resize_handle_size'] + REGION_HANDLE_EXTRA

    def _draw_paste_items(self, painter, background_rect):
        """绘制所有贴图"""
        for i, (pixmap, rect, label) in enumerate(self._editor.canvas_items):
            item_x = (rect.x() * self.background_scale) + background_rect.left()
            item_y = (rect.y() * self.background_scale) + background_rect.top()
            item_width = rect.width() * self.background_scale
            item_height = rect.height() * self.background_scale

            item_rect = QRectF(item_x, item_y, item_width, item_height)
            is_selected = (i == self._editor.selected_item)

            is_pressed_label = self._is_pressed_label({"label": label})
            self._draw_single_paste_item(
                painter, pixmap, item_rect, label, is_selected, is_pressed_label, i
            )

    def _get_box_border_pen(self, border_color, is_selected):
        from ..core.config import BOX_BORDER_CONFIG
        w = max(1, min(4, float(BOX_BORDER_CONFIG['width'])))
        pen_width = w * 1.5 if is_selected else w
        pen_width = max(1, min(4, pen_width))
        return QPen(border_color, pen_width)

    def _get_region_border_pen(self, color, is_selected):
        """区域边框笔：**复用检测框那把尺子，整体 +1**。

        用户要求「区域画图的框句柄大小和边框粗细只比检测框参数 +1」，所以
        这里不自己定粗细，而是拿 `BOX_BORDER_CONFIG['width']` 加 1，再套上
        与 `_get_box_border_pen()` **同一套**「选中 ×1.5、夹在 1..4」的规则。
        两边算法一致，以后检测框改粗细，区域自动跟着走。
        """
        from ..core.config import BOX_BORDER_CONFIG, REGION_BORDER_EXTRA
        base = (max(1, min(4, float(BOX_BORDER_CONFIG['width'])))
                + REGION_BORDER_EXTRA)
        pen_width = base * 1.5 if is_selected else base
        pen_width = max(1, min(4, pen_width))
        return QPen(color, pen_width)

    def _draw_single_paste_item(self, painter, pixmap, item_rect, label, is_selected, is_pressed_label, item_index=0):
        """绘制单个贴图"""
        item_x = item_rect.left()
        item_y = item_rect.top()
        item_width = item_rect.width()
        item_height = item_rect.height()

        label_color_hex = self._editor.get_label_color(label)
        lr = int(label_color_hex[1:3], 16)
        lg = int(label_color_hex[3:5], 16)
        lb = int(label_color_hex[5:7], 16)

        if is_selected or is_pressed_label:
            border_color = QColor(255, 255, 255)
        else:
            border_color = QColor(lr, lg, lb)
        pen = self._get_box_border_pen(border_color, is_selected or is_pressed_label)
        painter.setPen(pen)
        rx, ry = int(item_x), int(item_y)
        rw = int(item_x + item_width) - rx
        rh = int(item_y + item_height) - ry
        painter.drawRect(rx, ry, rw, rh)

        if is_selected or is_pressed_label:
            self._draw_paste_with_overlay(painter, pixmap, item_rect, label, 155)
        else:
            self._draw_paste_with_overlay(painter, pixmap, item_rect, label, 60)

        if is_selected:
            is_handle_hovered = (
                self.hover_resize_target == 'item' and
                self.hover_resize_handle == 'br'
            )
            self._draw_resize_handle(painter, item_rect, QColor(255, 255, 255), QColor(lr, lg, lb), is_handle_hovered)

        self._draw_paste_label(painter, item_x, item_y, label, is_selected, item_index)

    def _draw_paste_with_overlay(self, painter, pixmap, item_rect, label, fill_alpha):
        """绘制带标签色透明覆盖层的贴图。"""
        temp_pixmap = QPixmap(int(item_rect.width()), int(item_rect.height()))
        temp_pixmap.fill(Qt.transparent)
        temp_painter = QPainter(temp_pixmap)

        temp_painter.drawPixmap(
            0, 0, int(item_rect.width()), int(item_rect.height()), pixmap
        )

        color = QColor(self._editor.get_label_color(label))
        overlay_color = QColor(color.red(), color.green(), color.blue(), fill_alpha)
        temp_painter.fillRect(
            0, 0, int(item_rect.width()), int(item_rect.height()), overlay_color
        )
        temp_painter.end()

        painter.drawPixmap(
            int(item_rect.left()), int(item_rect.top()),
            temp_pixmap
        )

    def _draw_resize_handle(self, painter, item_rect, stroke_color, fill_color, is_hovered=False):
        """绘制右下角缩放手柄；悬停时显示白色正方形命中范围。"""
        size = PASTE_ITEM_CONFIG['handle_size']
        radius = size / 2
        br_handle = item_rect.bottomRight()

        painter.save()
        if is_hovered:
            painter.setPen(QPen(stroke_color, 2))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawRect(QRectF(br_handle.x() - radius, br_handle.y() - radius, size, size))
        else:
            # 1. 白色底层——减去贴图区域，只留贴图外部分
            from PySide6.QtGui import QPainterPath
            circle = QPainterPath()
            circle.addEllipse(br_handle, radius * 1.3, radius * 1.3)
            item = QPainterPath()
            item.addRect(item_rect)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawPath(circle.subtracted(item))
            # 2. 常态句柄覆盖（标签色）
            painter.setPen(QPen(fill_color, 1))
            painter.setBrush(fill_color)
            painter.drawEllipse(br_handle, radius, radius)
        painter.restore()

    @staticmethod
    def _draw_label_above_rect(painter, x, y, label, bg_color, font_size=None, position='outside'):
        """在矩形上方或内侧绘制标签（背景 + 文字）"""
        painter.save()
        font = painter.font()
        if font_size is not None:
            font.setPointSize(font_size)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(label)
        text_height = metrics.height()
        padding_x = 2

        label_y = int(y) if position == 'inside' else int(y) - text_height
        label_rect = QRectF(int(x), label_y, text_width + padding_x * 2, text_height)
        painter.fillRect(label_rect, bg_color)
        painter.setPen(QColor(0, 0, 0))
        text_rect = label_rect.adjusted(padding_x, 0, -padding_x, 0)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label)
        painter.restore()

    def _draw_paste_label(self, painter, x, y, label, is_selected, item_index=0):
        """绘制贴图标签"""
        if not self._editor.show_paste_names_checkbox.isChecked():
            return
        label_color_hex = self._editor.get_label_color(label)
        lr = int(label_color_hex[1:3], 16)
        lg = int(label_color_hex[3:5], 16)
        lb = int(label_color_hex[5:7], 16)
        bg_color = QColor(lr, lg, lb)
        self._draw_label_above_rect(painter, x, y, label, bg_color)

    def _box_visible(self, box):
        """框是否通过任务/分组筛选（未启用筛选则全部可见）。"""
        from ..core.utils import box_visible
        return box_visible(self._editor, box)

    def _draw_detection_boxes(self, painter, background_rect):
        """绘制所有检测框（关键点最后画，保证叠在同组框之上）。

        关键点分两遍：先画所有圆点，再画所有标签。否则后一个点的标签色块
        会盖住前一个点的圆点（标签块是不透明填充）。
        """
        points = []
        for i, box in enumerate(self._editor.detection_boxes):
            if not self._box_visible(box):
                continue
            if box.get("shape_type") == "point" and box.get("points"):
                points.append((i, box))
                continue
            self._draw_one_box(painter, background_rect, i, box)
        for i, box in points:
            self._draw_one_box(painter, background_rect, i, box, draw_dot=False)
        # 圆点最后画，且选中/按下的点排在最后，保证正在编辑的点不被相邻点盖住
        selected = set(getattr(self, 'selected_boxes', []) or [])
        if self.selected_box is not None:
            selected.add(self.selected_box)
        points.sort(key=lambda item: item[0] in selected)
        for i, box in points:
            self._draw_one_box(painter, background_rect, i, box, draw_label=False)

    def _draw_one_box(self, painter, background_rect, i, box, draw_dot=True, draw_label=True):
        is_selected = (i == self.selected_box or i in getattr(self, 'selected_boxes', []))
        pressed_box = getattr(self._editor, 'pressed_box_index', None)
        is_pressed_label = (
            (isinstance(pressed_box, int) and pressed_box == i)
            or self._is_pressed_label(box)
        )

        # 点/多边形用 points 定位，宽高为 0 属正常，不能套用矩形的退化判断
        if box.get("shape_type") == "point" and box.get("points"):
            if draw_dot:
                self._draw_point_shape(
                    painter, box, background_rect, is_selected, is_pressed_label,
                    draw_label=False,
                )
            if draw_label:
                self._draw_point_label(
                    painter, box, background_rect, is_selected, is_pressed_label
                )
            return

        if box.get("shape_type") == "polygon" and box.get("points"):
            self._draw_polygon_shape(
                painter, box, background_rect, is_selected, is_pressed_label
            )
            return

        if box.get("shape_type") == "rotation" and len(box.get("points") or []) == 4:
            self._draw_polygon_shape(
                painter, box, background_rect, is_selected, is_pressed_label
            )
            if is_selected:
                self._draw_rotation_handle(painter, box, background_rect)
            return

        if box["width"] <= 0 or box["height"] <= 0:
            return

        box_x = box["x"] * self.background_scale + background_rect.left()
        box_y = box["y"] * self.background_scale + background_rect.top()
        box_width = box["width"] * self.background_scale
        box_height = box["height"] * self.background_scale

        self._draw_single_detection_box(
            painter, box_x, box_y, box_width, box_height,
            box.get("label", ""), is_selected, is_pressed_label,
            group_id=box.get("group_id"), score=box.get("score"),
        )

    def _is_pressed_label(self, box):
        """检查检测框是否是当前按下的标签名（统计模式按住）。"""
        if not hasattr(self._editor, 'pressed_label') or self._editor.pressed_label is None:
            return False
        return box.get('label') == self._editor.pressed_label

    def _draw_single_detection_box(self, painter, x, y, width, height, label,
                                    is_selected, is_pressed_label, group_id=None,
                                    score=None):
        """绘制单个检测框"""
        label_color_hex = self._editor.get_label_color(label)
        lr = int(label_color_hex[1:3], 16)
        lg = int(label_color_hex[3:5], 16)
        lb = int(label_color_hex[5:7], 16)

        fill_alpha = 155 if is_selected or is_pressed_label else 60
        fill_color = QColor(lr, lg, lb, fill_alpha)
        if is_selected or is_pressed_label:
            border_color = QColor(255, 255, 255)
        else:
            border_color = QColor(lr, lg, lb)
        pen = self._get_box_border_pen(border_color, is_selected or is_pressed_label)
        painter.save()
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        rx, ry = int(x), int(y)
        rw = int(x + width) - rx
        rh = int(y + height) - ry
        painter.drawRect(rx, ry, rw, rh)
        painter.restore()
        painter.fillRect(rx, ry, rw, rh, fill_color)

        if label and getattr(self._editor, 'show_label_names_checkbox', None) and self._editor.show_label_names_checkbox.isChecked():
            from ..engine.shape_io import box_display_label
            display = box_display_label(
                {"label": label, "group_id": group_id, "score": score},
                show_score=self._show_scores())
            label_bg = QColor(lr, lg, lb)
            self._draw_box_label(painter, x, y, display, label_bg)

        if is_selected:
            handle_stroke = QColor(255, 255, 255)
            handle_fill = QColor(lr, lg, lb)
            self._draw_box_handles(painter, x, y, width, height, handle_stroke, handle_fill)

    def _show_scores(self):
        """「显示置信度」开关状态；没有该复选框时按不显示处理。"""
        cb = getattr(self._editor, 'show_scores_checkbox', None)
        return bool(cb is not None and cb.isChecked())

    def _draw_box_label(self, painter, x, y, label, bg_color):
        """绘制检测框标签"""
        font_size = max(5, min(15, DETECTION_BOX_CONFIG.get('label_font_size', 9)))
        position = DETECTION_BOX_CONFIG.get('label_position', 'outside')
        if position not in ('outside', 'inside'):
            position = 'outside'
        self._draw_label_above_rect(painter, x, y, label, bg_color, font_size, position)

    def _draw_box_handles(self, painter, x, y, width, height, stroke_color, fill_color):
        """绘制检测框四个角的调整手柄；悬停时显示白色正方形命中范围。"""
        size = DETECTION_BOX_CONFIG['resize_handle_size']
        radius = size / 2
        corners = (
            ('tl', QPointF(x, y)),
            ('tr', QPointF(x + width, y)),
            ('bl', QPointF(x, y + height)),
            ('br', QPointF(x + width, y + height)),
        )

        painter.save()
        for handle_name, corner in corners:
            is_hovered = (
                self.hover_resize_target == 'box' and
                self.hover_resize_handle == handle_name
            )
            if is_hovered:
                painter.setPen(QPen(stroke_color, 2))
                painter.setBrush(QColor(255, 255, 255))
                painter.drawRect(QRectF(corner.x() - radius, corner.y() - radius, size, size))
            else:
                # 1. 白色底层——减去框区域，只留框外部分
                from PySide6.QtGui import QPainterPath
                circle = QPainterPath()
                circle.addEllipse(corner, radius * 1.3, radius * 1.3)
                box = QPainterPath()
                box.addRect(QRectF(x, y, width, height))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255))
                painter.drawPath(circle.subtracted(box))
                # 2. 常态句柄覆盖（标签色）
                painter.setPen(QPen(fill_color, 1))
                painter.setBrush(fill_color)
                painter.drawEllipse(corner, radius, radius)
        painter.restore()

    def _draw_temp_box(self, painter):
        """绘制临时检测框（正在绘制中）"""
        if self.temp_draw_box is not None and self.draw_start_pos is not None:
            crosshair = CROSSHAIR_CONFIG.get('color', '#00FF80')
            pen = QPen(QColor(crosshair), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.temp_draw_box)

    def _image_to_canvas_points(self, points, background_rect):
        scale = self.background_scale
        left, top = background_rect.left(), background_rect.top()
        return [
            QPointF(p[0] * scale + left, p[1] * scale + top)
            for p in points
        ]

    def _draw_polygon_shape(self, painter, box, background_rect, is_selected, is_pressed_label):
        from PySide6.QtGui import QPainterPath
        pts = self._image_to_canvas_points(box.get("points") or [], background_rect)
        if len(pts) < 2:
            return
        label = box.get("label", "")
        label_color_hex = self._editor.get_label_color(label)
        lr = int(label_color_hex[1:3], 16)
        lg = int(label_color_hex[3:5], 16)
        lb = int(label_color_hex[5:7], 16)
        fill_alpha = 155 if is_selected or is_pressed_label else 60
        fill_color = QColor(lr, lg, lb, fill_alpha)
        border_color = QColor(255, 255, 255) if is_selected or is_pressed_label else QColor(lr, lg, lb)
        painter.setPen(self._get_box_border_pen(border_color, is_selected or is_pressed_label))
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        painter.fillPath(path, fill_color)
        painter.save()
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.restore()
        if label and getattr(self._editor, 'show_label_names_checkbox', None) and self._editor.show_label_names_checkbox.isChecked():
            from ..engine.shape_io import box_display_label
            text = box_display_label(box, show_score=self._show_scores())
            self._draw_box_label(painter, pts[0].x(), pts[0].y(), text, QColor(lr, lg, lb))
        if is_selected:
            size = DETECTION_BOX_CONFIG['resize_handle_size']
            painter.save()
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(lr, lg, lb))
            for p in pts:
                painter.drawEllipse(p, size / 2, size / 2)
            painter.restore()

    def _draw_rotation_handle(self, painter, box, background_rect):
        """旋转手柄：从 p0-p1 边中点向外的黑圈白心圆点。"""
        from ..engine.shape_io import rotation_handle_point
        points = box.get("points") or []
        if len(points) != 4:
            return
        handle = rotation_handle_point(points, 20 / max(self.background_scale, 1e-6))
        if handle is None:
            return
        hx = handle[0] * self.background_scale + background_rect.left()
        hy = handle[1] * self.background_scale + background_rect.top()
        mid_x = (points[0][0] + points[1][0]) / 2 * self.background_scale + background_rect.left()
        mid_y = (points[0][1] + points[1][1]) / 2 * self.background_scale + background_rect.top()
        radius = 6
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawLine(QPointF(mid_x, mid_y), QPointF(hx, hy))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(QPointF(hx, hy), radius, radius)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPointF(hx, hy), radius - 2, radius - 2)
        painter.restore()

    def _draw_point_shape(self, painter, box, background_rect, is_selected, is_pressed_label,
                          draw_label=True):
        label = box.get("label", "")
        label_color_hex = self._editor.get_label_color(label)
        lr = int(label_color_hex[1:3], 16)
        lg = int(label_color_hex[3:5], 16)
        lb = int(label_color_hex[5:7], 16)
        p = box["points"][0]
        cx = p[0] * self.background_scale + background_rect.left()
        cy = p[1] * self.background_scale + background_rect.top()
        center = QPointF(cx, cy)
        size = DETECTION_BOX_CONFIG['resize_handle_size']
        active = is_selected or is_pressed_label
        from ..engine.shape_io import point_warning
        warned = point_warning(box, getattr(self._editor, 'detection_boxes', None)) is not None
        painter.save()
        if warned:
            # 异常关键点（无同组框 / 不在框内）：方形 + 白色描边，与正常圆点区分
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QColor(lr, lg, lb))
            half = size / 2 + (3 if active else 2)
            painter.drawRect(QRectF(cx - half, cy - half, half * 2, half * 2))
        elif active:
            # 编辑态：白色外圈 + 放大实心点，和未选中的单色小圆点明显区分
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(center, size / 2 + 3, size / 2 + 3)
            painter.setPen(QPen(QColor(lr, lg, lb), 2))
            painter.setBrush(QColor(lr, lg, lb))
            painter.drawEllipse(center, size / 2 + 1, size / 2 + 1)
        else:
            painter.setPen(QPen(QColor(lr, lg, lb), 1))
            painter.setBrush(QColor(lr, lg, lb))
            painter.drawEllipse(center, size / 2, size / 2)
        painter.restore()
        if draw_label:
            self._draw_point_label(painter, box, background_rect, is_selected, is_pressed_label)

    def _draw_point_label(self, painter, box, background_rect, is_selected, is_pressed_label):
        label = box.get("label", "")
        if not (label and getattr(self._editor, 'show_label_names_checkbox', None)
                and self._editor.show_label_names_checkbox.isChecked()):
            return
        label_color_hex = self._editor.get_label_color(label)
        lr = int(label_color_hex[1:3], 16)
        lg = int(label_color_hex[3:5], 16)
        lb = int(label_color_hex[5:7], 16)
        p = box["points"][0]
        cx = p[0] * self.background_scale + background_rect.left()
        cy = p[1] * self.background_scale + background_rect.top()
        from ..engine.shape_io import box_display_label
        text = box_display_label(box, show_score=self._show_scores())
        self._draw_box_label(painter, cx, cy, text, QColor(lr, lg, lb))

    def _draw_temp_polygon(self, painter, background_rect):
        if not background_rect or not getattr(self, 'temp_polygon_points', None):
            return
        from PySide6.QtGui import QPainterPath
        pts = self._image_to_canvas_points(self.temp_polygon_points, background_rect)
        crosshair = CROSSHAIR_CONFIG.get('color', '#00FF80')
        painter.setPen(QPen(QColor(crosshair), 2, Qt.DashLine))
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        if self.mouse_inside:
            path.lineTo(QPointF(self.mouse_pos.x(), self.mouse_pos.y()))
        painter.drawPath(path)
        painter.setBrush(QColor(crosshair))
        painter.setPen(Qt.NoPen)
        for p in pts:
            painter.drawEllipse(p, 3, 3)


    def _draw_temp_region_box(self, painter):
        """绘制中的矩形区域预览（青色虚线，与区域颜色一致）。"""
        if self.temp_draw_box is None or self.draw_start_pos is None:
            return
        from ..core.config import REGION_CONFIG
        color = QColor(REGION_CONFIG['border_color'])
        # 绘制中的预览不是「选中」态，走与区域边框同一把尺子（检测框 +1）
        pen = self._get_region_border_pen(color, False)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern(REGION_CONFIG['dash_pattern'])
        painter.setPen(pen)
        painter.drawRect(self.temp_draw_box)

    def _draw_temp_region_polygon(self, painter, background_rect):
        """绘制中的多边形区域预览（青色虚线）。"""
        if not background_rect or not getattr(self, 'temp_region_points', None):
            return
        from PySide6.QtGui import QPainterPath
        from ..core.config import REGION_CONFIG
        color = QColor(REGION_CONFIG['border_color'])
        pts = self._image_to_canvas_points(self.temp_region_points, background_rect)
        pen = self._get_region_border_pen(color, False)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        if self.mouse_inside:
            path.lineTo(QPointF(self.mouse_pos.x(), self.mouse_pos.y()))
        painter.drawPath(path)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(color, 1))
        for p in pts:
            painter.drawEllipse(p, 3, 3)

    def _draw_crosshair(self, painter):
        """绘制标注模式的鼠标十字虚线。"""
        color = QColor(CROSSHAIR_CONFIG.get('color', '#00FF80'))
        color.setAlpha(CROSSHAIR_CONFIG.get('alpha', 160))
        cross_pen = QPen(color, CROSSHAIR_CONFIG.get('width', 1), Qt.DashLine)
        painter.setPen(cross_pen)
        painter.drawLine(QPointF(0, self.mouse_pos.y()), QPointF(self.width(), self.mouse_pos.y()))
        painter.drawLine(QPointF(self.mouse_pos.x(), 0), QPointF(self.mouse_pos.x(), self.height()))

    def _draw_magnifier(self, painter, background_rect, scene):
        if not getattr(self._editor, '_magnifier_enabled', False):
            return
        if not self.mouse_inside or self.mouse_pos is None:
            return
        pixmap = self._editor.current_background
        if pixmap is None or pixmap.isNull() or not background_rect.contains(self.mouse_pos):
            return

        size = int(MAGNIFIER_CONFIG.get('size', 160))
        zoom = float(MAGNIFIER_CONFIG.get('zoom', 2.0))
        if size <= 0 or zoom <= 0:
            return

        src_size = max(1, int(size / zoom))
        # 取样窗口始终以光标为中心；越界不夹紧，否则放大内容会偏离光标
        src_x = int(self.mouse_pos.x()) - src_size // 2
        src_y = int(self.mouse_pos.y()) - src_size // 2

        t_bg = ThemeManager.get_theme()['canvas_bg']
        src_pix = QPixmap(src_size, src_size)
        src_pix.fill(QColor(
            int(t_bg[1:3], 16), int(t_bg[3:5], 16), int(t_bg[5:7], 16)
        ))
        ox = max(src_x, 0)
        oy = max(src_y, 0)
        ox2 = min(src_x + src_size, self.width())
        oy2 = min(src_y + src_size, self.height())
        if ox2 > ox and oy2 > oy:
            sp2 = QPainter(src_pix)
            sp2.drawPixmap(ox - src_x, oy - src_y, scene, ox, oy, ox2 - ox, oy2 - oy)
            sp2.end()

        mag_pos = MAGNIFIER_CONFIG.get('position', 'side')
        if mag_pos == 'center':
            dst_x = self.mouse_pos.x() - size // 2
            dst_y = self.mouse_pos.y() - size // 2
        else:
            margin = 18
            dst_x = self.mouse_pos.x() + margin
            dst_y = self.mouse_pos.y() + margin
            if dst_x + size > self.width():
                dst_x = self.mouse_pos.x() - margin - size
            if dst_y + size > self.height():
                dst_y = self.mouse_pos.y() - margin - size
            dst_x = max(0, min(self.width() - size, int(dst_x)))
            dst_y = max(0, min(self.height() - size, int(dst_y)))
        dst = QRectF(dst_x, dst_y, size, size)

        painter.save()
        painter.fillRect(dst, QColor(255, 255, 255, 210))
        painter.drawPixmap(dst, src_pix, QRectF(0, 0, src_size, src_size))
        painter.setPen(QPen(QColor(60, 60, 60, 180), 1))
        painter.drawRect(dst)
        cross_color = QColor(CROSSHAIR_CONFIG.get('color', '#00FF80'))
        cross_color.setAlpha(CROSSHAIR_CONFIG.get('alpha', 160))
        painter.setPen(QPen(cross_color, 1, Qt.DashLine))
        center_x = dst.left() + size / 2
        center_y = dst.top() + size / 2
        painter.drawLine(QPointF(center_x, dst.top()), QPointF(center_x, dst.bottom()))
        painter.drawLine(QPointF(dst.left(), center_y), QPointF(dst.right(), center_y))
        painter.restore()

    # ---------- 画布显示（亮度/对比度）----------

    def apply_brightness_contrast(self, brightness, contrast):
        """滑块入口：更新参数并立刻套用到当前图。"""
        self._brightness = brightness
        self._contrast = contrast
        self.apply_display_adjustments()
        self.update()

    def _resolve_adjust_source(self):
        """找出本轮调整的源图（未套用亮度/对比度的那张）。

        切图时 current_background 会被直接换成新图，这里用身份比较识别：
        当前图只要不是本方法上一次产出的结果，就说明换图了。
        """
        current = getattr(self._editor, 'current_background', None)
        if current is None:
            return None
        if current is getattr(self, '_adjusted_background', None):
            return getattr(self, '_adjusted_source', None)
        return current

    def _render_adjusted_background(self, source, brightness, contrast):
        """按亮度/对比度生成新图；中性参数直接返回源图。"""
        b_factor = brightness / 50.0
        c_factor = contrast / 50.0
        if b_factor == 1.0 and c_factor == 1.0:
            return source
        from PySide6.QtGui import QImage
        img = source.toImage().convertToFormat(QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        ptr = img.bits()
        pixels = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        rgb = (pixels[:, :, :3].astype(np.float32) - 128.0) * \
            (b_factor * c_factor) + 128.0
        pixels[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        return QPixmap.fromImage(img)

    def apply_display_adjustments(self):
        """让当前图片带上当前的亮度/对比度。

        由 paintEvent 每次调用，所以切图后会自动重新套用，滑块值本身不重置。
        按「源图身份 + 参数」跳过重复计算，避免鼠标移动重绘时反复跑像素处理。
        调整失败时退化为不调整，绝不让绘制路径抛异常。
        """
        if getattr(self, '_editor', None) is None:
            return
        brightness = getattr(self, '_brightness', 50)
        contrast = getattr(self, '_contrast', 50)
        current = getattr(self._editor, 'current_background', None)

        if current is None:
            self._adjusted_source = None
            self._adjusted_background = None
            return

        source = self._resolve_adjust_source()
        if source is None:
            self._adjusted_source = None
            self._adjusted_background = None
            return

        unchanged = (
            current is getattr(self, '_adjusted_background', None)
            and getattr(self, '_adjusted_brightness', None) == brightness
            and getattr(self, '_adjusted_contrast', None) == contrast
        )
        if unchanged:
            return

        try:
            result = self._render_adjusted_background(
                source, brightness, contrast)
        except Exception as exc:
            from ..core.exception_hook import _write_log
            _write_log(f"画布显示调整失败，已跳过: {exc}")
            self._adjusted_source = source
            self._adjusted_background = current
            return

        self._adjusted_source = source
        self._adjusted_background = result
        self._adjusted_brightness = brightness
        self._adjusted_contrast = contrast
        if result is not source:
            self._editor.current_background = result
