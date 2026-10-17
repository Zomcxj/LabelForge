"""
Canvas 控件 - 由 CanvasRendererMixin + CanvasInteractionMixin 组合而成
"""
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint, QRectF

from ..core.config import BACKGROUND_SCALE_CONFIG, WINDOW_CONFIG
from ..core.utils import format_size_ratio
from .canvas_renderer import CanvasRendererMixin
from .canvas_interaction import CanvasInteractionMixin


class Canvas(CanvasRendererMixin, CanvasInteractionMixin, QWidget):
    """画布控件 - 用于显示和编辑背景图、贴图、检测框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editor = parent
        self.setMinimumSize(
            WINDOW_CONFIG.get('min_width', 1024),
            WINDOW_CONFIG.get('min_height', 768)
        )

        # 拖动相关
        self.drag_start = QPoint()
        self.resize_handle = None
        self.resize_start = QPoint()
        self.hover_resize_target = None
        self.hover_resize_handle = None
        # 区域悬停态：与检测框的 `hover_resize_target/handle` 对应。
        # `hover_region_index` 是悬停那条区域的序号，`hover_region_handle` 是
        # 手柄名（tl/tr/bl/br/t/b/l/r）。渲染端据此把该手柄画成白色正方形。
        self.hover_region_index = None
        self.hover_region_handle = None

        # 背景图缩放和偏移
        self.background_scale = BACKGROUND_SCALE_CONFIG['default_scale']
        self.background_offset = QPoint(0, 0)
        self.is_dragging_background = False
        self.is_manual_scale = False

        # 贴图相关
        self.is_dragging_item = False
        self.selected_item_size = None

        # 检测框相关
        self.selected_box = None
        self.selected_boxes = []
        self.is_dragging_box = False
        self.box_drag_start = QPoint()
        self.is_resizing_box = False
        self.box_resize_start = QPoint()
        # OBB 旋转手柄拖拽
        self.is_rotating_box = False
        self.rotation_center = None
        self.rotation_prev_angle = None

        # 绘制模式相关
        self.is_drawing_box = False
        self.is_drawing_polygon = False
        self.is_drawing_point = False
        self.is_drawing_obb = False
        self.current_draw_mode = None
        self.draw_start_pos = None
        self.temp_draw_box = None
        self.temp_polygon_points = []

        # 语义区域相关（仅运行时，贴图模式右键菜单操作）
        self.is_drawing_region = False
        self.is_drawing_region_polygon = False
        self.temp_region_points = []
        self.selected_region = None
        self.selected_region_vertex = None
        self.is_dragging_region = False
        self.is_resizing_region = False
        self.is_dragging_region_vertex = False
        self.region_vertex_drag_index = None
        self.region_resize_handle = None
        self.region_drag_start = QPoint()
        self.region_resize_start = QPoint()

        # 画布显示参数（亮度/对比度在本次运行内跨图常驻）
        self.shape_opacity = 1.0
        self._brightness = 50
        self._contrast = 50
        # 上一次调整所用的源图与产出图，用于识别切图
        self._adjusted_source = None
        self._adjusted_background = None
        self._adjusted_brightness = None
        self._adjusted_contrast = None

        # 鼠标状态跟踪
        self.mouse_inside = False
        self.mouse_pos = QPoint(0, 0)
        self._drag_out_pending = False
        # 按下时记下的起点，用来判断「拖出复制」的位移是否够阈值
        self._drag_out_start = None

        # 设置焦点策略和鼠标追踪
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def get_background_rect(self):
        """获取背景图在画布上的实际绘制矩形"""
        if self._editor.current_background is None:
            return None

        bg_w = self._editor.current_background.width()
        bg_h = self._editor.current_background.height()
        if bg_w <= 0 or bg_h <= 0:
            return None

        if not self.is_manual_scale:
            scale_x = self.width() / bg_w
            scale_y = self.height() / bg_h
            self.background_scale = min(scale_x, scale_y)
            self.background_offset = QPoint(0, 0)

        scaled_width = bg_w * self.background_scale
        scaled_height = bg_h * self.background_scale

        x = (self.width() - scaled_width) // 2 + self.background_offset.x()
        y = (self.height() - scaled_height) // 2 + self.background_offset.y()

        return QRectF(x, y, scaled_width, scaled_height)

    def reset_view(self):
        """重置画布视图为适应窗口"""
        if self._editor.current_background is None:
            return
        bg_w = self._editor.current_background.width()
        bg_h = self._editor.current_background.height()
        if bg_w > 0 and bg_h > 0:
            scale_x = self.width() / bg_w
            scale_y = self.height() / bg_h
            self.background_scale = min(scale_x, scale_y)
        else:
            self.background_scale = 1.0
        self.background_offset = QPoint(0, 0)
        self.is_manual_scale = False

    def find_item_at_position(self, pos):
        """查找指定位置的贴图索引（含右下角缩放手柄区域）"""
        if self._editor.current_background is None:
            return None

        background_rect = self.get_background_rect()
        if background_rect is None:
            return None

        from labelforge.core.config import PASTE_ITEM_CONFIG
        handle_size = PASTE_ITEM_CONFIG['handle_size']

        for i in range(len(self._editor.canvas_items) - 1, -1, -1):
            pixmap, rect, label = self._editor.canvas_items[i]

            item_x = (rect.x() * self.background_scale) + background_rect.left()
            item_y = (rect.y() * self.background_scale) + background_rect.top()
            item_width = rect.width() * self.background_scale
            item_height = rect.height() * self.background_scale

            if (item_x <= pos.x() <= item_x + item_width and
                item_y <= pos.y() <= item_y + item_height):
                return i

            # 右下角缩放手柄区域（延伸到贴图框外，与检测框手柄一致）
            br_x = item_x + item_width
            br_y = item_y + item_height
            if (br_x - handle_size / 2 <= pos.x() <= br_x + handle_size / 2 and
                br_y - handle_size / 2 <= pos.y() <= br_y + handle_size / 2):
                return i

        return None

    def update_status_label(self):
        """更新状态栏显示"""
        if not self._editor.current_background:
            return

        info = self._editor.get_image_info()
        stats = self._editor.get_label_stats()
        stats_parts = [f"{k}:{v}" for k, v in list(stats.items())[
            :self._editor._max_labels if hasattr(self._editor, '_max_labels') else 3
        ]]

        if self.mouse_inside:
            background_rect = self.get_background_rect()
            if background_rect:
                rel_x = self.mouse_pos.x() - background_rect.left()
                rel_y = self.mouse_pos.y() - background_rect.top()
                orig_x = rel_x / self.background_scale
                orig_y = rel_y / self.background_scale

                parts = []
                if info:
                    parts.append(f"Box:{info['box_count']} Paste:{info['paste_count']}")
                parts.append(f"X:{int(orig_x)} Y:{int(orig_y)}")

                size = None
                if self.selected_item_size:
                    size = self.selected_item_size
                elif (self.selected_box is not None and
                      0 <= self.selected_box < len(self._editor.detection_boxes)):
                    box = self._editor.detection_boxes[self.selected_box]
                    size = (box['width'], box['height'])
                elif self.is_drawing_box and self.temp_draw_box:
                    size = (self.temp_draw_box.width() / self.background_scale,
                            self.temp_draw_box.height() / self.background_scale)
                elif getattr(self, 'is_drawing_polygon', False) and self.temp_polygon_points:
                    xs = [p[0] for p in self.temp_polygon_points]
                    ys = [p[1] for p in self.temp_polygon_points]
                    size = (max(xs) - min(xs) if xs else 0, max(ys) - min(ys) if ys else 0)
                if size:
                    image_w = info['width'] if info else None
                    image_h = info['height'] if info else None
                    parts.append(format_size_ratio(size[0], size[1],
                                                   image_w, image_h))

                angle_text = _rotation_angle_status_text(
                    self._editor.detection_boxes, self.selected_box)
                if angle_text:
                    parts.append(angle_text)

                if stats_parts:
                    parts.append(" ".join(stats_parts))

                prefix = "[移除路径] " if self._editor._is_delete_view else ""
                self._editor.status_label.setText(prefix + " | ".join(parts))
                return

        # 鼠标不在canvas上：只显示 Paste/Box + 类别
        parts = []
        if info:
            parts.append(f"Box:{info['box_count']} Paste:{info['paste_count']}")
        angle_text = _rotation_angle_status_text(
            getattr(self._editor, 'detection_boxes', None), self.selected_box)
        if angle_text:
            parts.append(angle_text)
        if stats_parts:
            parts.append(" ".join(stats_parts))
        if parts:
            prefix = "[移除路径] " if self._editor._is_delete_view else ""
            self._editor.status_label.setText(prefix + " | ".join(parts))


def _rotation_angle_status_text(detection_boxes, selected_box):
    """选中的旋转框在状态栏显示角度，如 'A:30°'；非旋转框返回空串。"""
    if (selected_box is None or not detection_boxes or
            not (0 <= selected_box < len(detection_boxes))):
        return ""
    box = detection_boxes[selected_box]
    if not isinstance(box, dict) or box.get("shape_type") != "rotation":
        return ""
    points = box.get("points") or []
    if len(points) != 4:
        return ""
    from ..engine.shape_io import rotation_angle_degrees
    angle = rotation_angle_degrees(points)
    if angle >= 359.5:
        angle = 0.0
    return f"A:{angle:.0f}°"
