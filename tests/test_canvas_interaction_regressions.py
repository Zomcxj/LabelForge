"""画布悬停状态回归测试。"""
from PySide6.QtCore import Qt

from labelforge.canvas import canvas_interaction
from labelforge.canvas import canvas as canvas_module
from labelforge.canvas.canvas import Canvas as RealCanvas
from labelforge.canvas.canvas_interaction import CanvasInteractionMixin
from labelforge.ui.main_window import ImageEditor


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self): return self._x
    def y(self): return self._y

    def __sub__(self, other):
        return Point(self._x - other.x(), self._y - other.y())

    def __add__(self, other):
        return Point(self._x + other.x(), self._y + other.y())


class Rect:
    def __init__(self, x=0, y=0, w=10, h=10):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def x(self): return self._x
    def y(self): return self._y
    def width(self): return self._w
    def height(self): return self._h
    def left(self): return self._x
    def top(self): return self._y
    def center(self): return Point(self._x + self._w / 2, self._y + self._h / 2)
    def bottomRight(self): return Point(self._x + self._w, self._y + self._h)
    def contains(self, point):
        return self._x <= point.x() <= self._x + self._w and self._y <= point.y() <= self._y + self._h


class Background:
    def __init__(self, w=100, h=100):
        self._w = w
        self._h = h

    def width(self): return self._w
    def height(self): return self._h


class WheelDelta:
    def __init__(self, y):
        self._y = y

    def y(self): return self._y


class WheelEvent:
    def __init__(self, delta_y, modifiers=0):
        self._delta = WheelDelta(delta_y)
        self._modifiers = modifiers

    def angleDelta(self): return self._delta
    def modifiers(self): return self._modifiers


class Editor:
    def __init__(self, mode):
        self.edit_mode = mode
        self._is_delete_view = False
        self.background_images = ["work.png"]
        self.current_background = Background()
        self.current_background_index = 0
        self.canvas_items = [(None, Rect(0, 0, 10, 10), "paste")]
        self.detection_boxes = [{"label": "cat", "x": 0, "y": 0, "width": 10, "height": 10}]
        self.detection_boxes_dict = {0: [self.detection_boxes[0].copy()]}
        self.selected_item = None
        self.pressed_label = None
        self.show_labels_checkbox = type("Check", (), {"isChecked": lambda self: True})()

    def save_undo_state(self):
        pass


class Canvas(CanvasInteractionMixin):
    def __init__(self, mode):
        self._editor = Editor(mode)
        self.background_scale = 1
        self.mouse_pos = Point(5, 5)
        self.selected_box = None
        self.selected_boxes = []
        self.selected_item_size = None
        self.updated = 0
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.synced_indexes = []
        self.saved_detection_boxes = 0
        self.is_drawing_box = False
        self.draw_start_pos = None
        self.temp_draw_box = None
        self.box_drag_start = Point(0, 0)

    def setCursor(self, *a):
        pass

    def update_status_label(self):
        pass

    def get_background_rect(self):
        return Rect(0, 0, 100, 100)

    def update(self):
        self.updated += 1

    def _sync_detection_box_to_dict(self, index):
        self.synced_indexes.append(index)

    def _can_edit_canvas(self):
        return not getattr(self._editor, '_is_delete_view', False)

    def _save_current_detection_boxes(self):
        self.saved_detection_boxes += 1

    def _current_modifiers(self):
        return 0

    def save_undo_state(self):
        pass

    def find_item_at_position(self, point):
        for index, (_, rect, _) in enumerate(self._editor.canvas_items):
            if rect.contains(point):
                return index
        return None


def test_reset_view_keeps_background_adaptive_to_canvas_resize(monkeypatch):
    monkeypatch.setattr(canvas_module, "QPoint", Point)
    canvas = RealCanvas.__new__(RealCanvas)
    canvas._editor = type("Editor", (), {"current_background": Background(100, 50)})()
    canvas.background_scale = 1.0
    canvas.background_offset = Point(0, 0)
    canvas.is_manual_scale = False
    canvas.width = lambda: 200
    canvas.height = lambda: 100

    canvas.reset_view()
    canvas.width = lambda: 100
    canvas.height = lambda: 200

    canvas.get_background_rect()

    assert canvas.background_scale == 1.0


def test_paste_hover_selects_paste_without_selecting_detection_box(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("paste")

    canvas._check_hover()

    assert canvas._editor.selected_item == 0
    assert canvas.selected_box is None
    assert canvas.selected_boxes == []


def test_annotate_hover_clears_paste_item_edit_state(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("annotate")
    canvas._editor.selected_item = 0
    canvas.selected_item_size = (10, 10)

    canvas._check_hover()

    # annotate mode 下 hover 到检测框，自动切换到框编辑
    assert canvas.selected_box == 0
    assert canvas._editor.selected_item is None
    assert canvas.selected_item_size is None


def test_paste_hover_does_not_select_detection_box_without_click(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("paste")
    canvas._editor.canvas_items = []

    canvas._check_hover()

    assert canvas.selected_box is None
    assert canvas.selected_boxes == []


def test_paste_click_still_selects_detection_box(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("paste")
    canvas._editor.canvas_items = []

    canvas._handle_left_click(Point(5, 5))

    assert canvas.selected_box == 0
    assert canvas.selected_boxes == [0]


def test_click_prefers_keypoint_over_enclosing_group_box(monkeypatch):
    """关键点画在框上层，点击时应优先选中关键点而不是框。"""
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("annotate")
    canvas._editor.canvas_items = []
    canvas._editor.detection_boxes = [
        {"label": "BCar", "x": 0, "y": 0, "width": 40, "height": 40,
         "shape_type": "rectangle", "group_id": 1},
        {"label": "p1", "shape_type": "point", "points": [[5, 5]],
         "x": 5, "y": 5, "width": 0, "height": 0, "group_id": 1},
    ]

    canvas._handle_left_click(Point(5, 5))

    assert canvas.selected_box == 1


def test_click_prefers_topmost_overlapping_point(monkeypatch):
    """两个重叠关键点：后画的在上层，点击应选中它。"""
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("annotate")
    canvas._editor.canvas_items = []
    canvas._editor.detection_boxes = [
        {"label": "p1", "shape_type": "point", "points": [[5, 5]],
         "x": 5, "y": 5, "width": 0, "height": 0, "group_id": 1},
        {"label": "p2", "shape_type": "point", "points": [[5, 5]],
         "x": 5, "y": 5, "width": 0, "height": 0, "group_id": 1},
    ]

    canvas._handle_left_click(Point(5, 5))

    assert canvas.selected_box == 1


def test_mode_switch_clears_paste_and_detection_box_selection():
    class StatusLabel:
        def setText(self, text):
            self.text = text

    editor = type("Editor", (), {})()
    editor.edit_mode = "annotate"
    editor.selected_item = 0
    editor.status_label = StatusLabel()
    editor.canvas = type("Canvas", (), {
        "selected_item_size": (10, 10),
        "selected_box": 0,
        "selected_boxes": [0],
        "hover_resize_target": "item",
        "hover_resize_handle": "br",
        "update": lambda self: None,
    })()
    editor._apply_mode_visibility_defaults = lambda: None
    editor._update_mode_seg_style = lambda animated=False: None
    editor._apply_paste_label_visibility = lambda: None

    ImageEditor._set_edit_mode(editor, "paste")

    assert editor.selected_item is None
    assert editor.canvas.selected_item_size is None
    assert editor.canvas.selected_box is None
    assert editor.canvas.selected_boxes == []


def test_ctrl_hover_adds_detection_box_to_selected_boxes_without_click(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("annotate")
    canvas.selected_box = 0
    canvas.selected_boxes = [0]
    canvas.mouse_pos = Point(25, 5)
    canvas._editor.detection_boxes.append({"label": "dog", "x": 20, "y": 0, "width": 10, "height": 10})
    canvas._current_modifiers = lambda: Qt.ControlModifier

    canvas._check_hover()

    assert canvas.selected_box == 1
    assert canvas.selected_boxes == [0, 1]


def test_scale_selected_box_uses_detection_box_scale_step(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'detection_box_scale_step', 0.1)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'paste_item_scale_step', 0.2)
    canvas = Canvas("annotate")
    canvas.selected_box = 0

    canvas._scale_selected_box(WheelEvent(120))

    box = canvas._editor.detection_boxes[0]
    assert round(box['x'], 2) == 0.0
    assert round(box['y'], 2) == 0.0
    assert round(box['width'], 2) == 11.0
    assert round(box['height'], 2) == 11.0
    assert canvas.synced_indexes == [0]


def test_scale_selected_item_uses_paste_item_scale_step(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'detection_box_scale_step', 0.1)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'paste_item_scale_step', 0.2)
    canvas = Canvas("paste")
    canvas._editor.selected_item = 0

    canvas._scale_selected_item(WheelEvent(120))

    _, rect, _ = canvas._editor.canvas_items[0]
    assert round(rect.width(), 2) == 12.0
    assert round(rect.height(), 2) == 12.0


def test_adjust_selected_box_edge_moves_only_nearest_edge(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'edge_step', 5)
    canvas = Canvas("annotate")
    canvas.selected_box = 0
    canvas._editor.detection_boxes[0].update({"x": 20, "y": 10, "width": 10, "height": 10})
    canvas.mouse_pos = Point(17, 15)

    canvas._adjust_selected_box_edge(WheelEvent(120))

    box = canvas._editor.detection_boxes[0]
    assert box['x'] == 15
    assert box['y'] == 10
    assert box['width'] == 15
    assert box['height'] == 10
    assert canvas.synced_indexes == [0]


def test_wheel_event_uses_edge_adjustment_when_mouse_is_outside_selected_box(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'edge_step', 4)
    canvas = Canvas("annotate")
    canvas.selected_box = 0
    canvas._editor.detection_boxes[0].update({"x": 20, "y": 10, "width": 20, "height": 10})
    canvas.mouse_pos = Point(41, 15)

    canvas.wheelEvent(WheelEvent(-120))

    box = canvas._editor.detection_boxes[0]
    assert box['x'] == 20
    assert box['y'] == 10
    assert box['width'] == 16
    assert box['height'] == 10


def test_wheel_edge_adjustment_keeps_opposite_edge_fixed_across_scrolls(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'edge_step', 5)
    canvas = Canvas("annotate")
    canvas.selected_box = 0
    canvas._editor.detection_boxes[0].update({"x": 20, "y": 10, "width": 10, "height": 10})
    canvas.mouse_pos = Point(17, 15)

    canvas.wheelEvent(WheelEvent(120))
    canvas.wheelEvent(WheelEvent(120))

    box = canvas._editor.detection_boxes[0]
    assert box['x'] == 10
    assert box['y'] == 10
    assert box['width'] == 20
    assert box['height'] == 10


def test_wheel_event_uses_image_coords_for_right_edge_adjustment_when_scaled(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'edge_step', 4)
    canvas = Canvas("annotate")
    canvas.selected_box = 0
    canvas.background_scale = 0.5
    canvas._editor.detection_boxes[0].update({"x": 20, "y": 10, "width": 20, "height": 10})
    canvas.mouse_pos = Point(21, 8)

    canvas.wheelEvent(WheelEvent(-120))

    box = canvas._editor.detection_boxes[0]
    assert box['x'] == 20
    assert box['y'] == 10
    assert box['width'] == 16
    assert box['height'] == 10


def test_delete_view_ignores_detection_box_edits_and_preserves_work_cache(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'detection_box_scale_step', 0.1)
    canvas = Canvas("annotate")
    canvas._editor._is_delete_view = True
    original = canvas._editor.detection_boxes[0].copy()
    canvas.selected_box = 0

    canvas._scale_selected_box(WheelEvent(120))
    canvas._nudge_selected(1, 0)

    assert canvas._editor.detection_boxes == [original]
    assert canvas._editor.detection_boxes_dict[0] == [original]
    assert canvas.synced_indexes == []


def test_delete_view_ignores_box_drawing(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("annotate")
    canvas._editor._is_delete_view = True

    assert canvas._handle_drawing_press(Point(5, 5)) is True
    assert canvas.draw_start_pos is None
    assert canvas.temp_draw_box is None


def test_delete_view_release_does_not_save_removed_image_labels_to_work_path(monkeypatch):
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    canvas = Canvas("annotate")
    canvas._editor._is_delete_view = True
    work_boxes = [canvas._editor.detection_boxes[0].copy()]
    canvas.is_dragging_box = True
    canvas._needs_save = True

    canvas.mouseReleaseEvent(None)

    assert canvas.saved_detection_boxes == 0
    assert canvas._editor.detection_boxes_dict[0] == work_boxes


# ---------- 画布图片复制：画布上按住拖动应该能拖出文件 ----------

class _CopyEditor(Editor):
    """开了「画布图片复制」的 editor。"""

    def __init__(self, mode="paste"):
        super().__init__(mode)
        self._canvas_image_copy_enabled = True
        self.current_background_path = "work.png"


class _CopyCanvas(Canvas):
    """点数够用的假画布：只要事件路径，不要真 Qt 控件。"""

    def __init__(self, mode="paste", scale=1.0):
        super().__init__(mode)
        self._editor = _CopyEditor(mode)
        self._drag_out_pending = False
        self.is_dragging_background = False
        self.is_dragging_item = False
        self.is_manual_scale = False
        self.background_scale = scale
        self.background_offset = Point(0, 0)
        self.drag_start = Point(0, 0)
        self._drag_out_calls = 0
        self._drag_out_mime = None
        # `mousePressEvent` 尾部会读的拖动状态
        self.is_dragging_box = False
        self.is_resizing_box = False
        self.is_rotating_box = False
        self.is_dragging_region = False
        self.is_resizing_region = False
        self.is_dragging_region_vertex = False
        self.is_drawing_box = False
        self.is_drawing_polygon = False
        self.is_drawing_region = False
        self.is_drawing_region_polygon = False
        self.is_drawing_point = False
        self.selected_item_size = None
        self.draw_start_pos = None
        self.temp_draw_box = None
        self._editor.detection_boxes = []
        self._editor.canvas_items = []

    def width(self):
        return 800

    def height(self):
        return 800

    def setFocus(self):
        pass

    def update(self):
        pass

    def update_status_label(self):
        pass

    def setCursor(self, *a):
        pass

    def get_background_rect(self):
        # 图铺满整块画布（图比画布小 → 不进入背景拖动，只能靠「拖出复制」）
        return Rect(0, 0, 100, 100)

    def find_item_at_position(self, point):
        return None

    def _region_at(self, point, include_locked=False):
        return None

    def _do_canvas_drag_out(self):
        self._drag_out_calls += 1
        self._drag_out_pending = False


class _PanningCopyCanvas(_CopyCanvas):
    """放大到超出画布的那种画布：按下会进入平移。"""

    def width(self):
        return 100

    def height(self):
        return 100

    def get_background_rect(self):
        return Rect(0, 0, 100, 100)


class _PressEvent:
    def __init__(self, x, y, button=Qt.LeftButton):
        self._pos = Point(x, y)
        self._button = button

    def pos(self): return self._pos
    def button(self): return self._button
    def modifiers(self): return 0


class _MoveEvent:
    """只带位置与按住左键的移动事件（不带 modifiers，走不到就退）。"""

    def __init__(self, x, y, buttons=Qt.LeftButton):
        self._pos = Point(x, y)
        self._buttons = buttons

    def pos(self): return self._pos
    def buttons(self): return self._buttons


def _enable_copy_drag_out(monkeypatch, canvas):
    """把拖出复制里那几样真 Qt 依赖换成假的，只观察有没有触发。"""
    import types
    monkeypatch.setitem(
        __import__("sys").modules, "PySide6.QtGui",
        types.SimpleNamespace(QCursor=types.SimpleNamespace(
            pos=lambda: Point(10, 10))))
    monkeypatch.setattr(canvas_interaction, "QDrag",
                        lambda *a, **k: type("D", (), {
                            "setMimeData": lambda s, m: None,
                            "exec_": lambda s, a: None})())
    monkeypatch.setattr(canvas_interaction, "QMimeData",
                        lambda: type("M", (), {"setUrls": lambda s, u: None})())
    monkeypatch.setattr(canvas_interaction, "QUrl", type(
        "U", (), {"fromLocalFile": staticmethod(lambda p: p)}))
    monkeypatch.setattr(canvas_interaction.os.path, "isfile", lambda p: True)


def test_canvas_copy_drags_out_without_leaving_the_window(monkeypatch):
    """在图**外**的空白画布上按住拖动一段就该拖出文件——不必把光标拖出主窗口。

    原实现只在**光标离开整个主窗口**时才触发拖出，主窗口最大化时光标出不去，
    于是拖多远都没反应。旁边的背景图列表同款功能是「移动超过 20 像素就触发」，
    两边行为应当一致。

    注意落点：按在**图上**是拖动画布（见 `test_canvas_pans_even_when_the_image_fits`），
    按在图外的空白处才是「把这张图拖出去」。
    """
    canvas = _CopyCanvas()
    _enable_copy_drag_out(monkeypatch, canvas)
    # 主窗口很大，光标整段拖动都在窗口内
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    # 图占 (0,0,100,100)，这里点在图外
    canvas.mousePressEvent(_PressEvent(400, 400))
    canvas.mouseMoveEvent(_MoveEvent(400 + 40, 400 + 40))

    assert canvas._drag_out_calls == 1, (
        "在窗口内拖了 40 像素却没有触发拖出复制，用户会感觉「拖不动」")


def test_canvas_copy_small_jitter_does_not_drag_out(monkeypatch):
    """手抖几个像素不该触发拖出——否则单击/微调都会误拖出文件。"""
    canvas = _CopyCanvas()
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(52, 51))

    assert canvas._drag_out_calls == 0, "轻微抖动就拖出文件，太容易误触"


def test_canvas_copy_drag_out_needs_the_copy_feature_enabled(monkeypatch):
    """没开这个功能时，拖动不该拖出文件。"""
    canvas = _CopyCanvas()
    canvas._editor._canvas_image_copy_enabled = False
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 0


def test_canvas_copy_drag_out_wins_over_panning_on_the_image(monkeypatch):
    """复制开着时，按住**图内**拖动是拖出文件，不是平移画布。

    用户报的「开启时拖动到桌面松开没有复制一份」就撞在这里：按在图上会先被判
    成平移（`is_dragging_background`），拖出永远挂不起来，只有点在图外空白才
    拖得出去。所以复制开着时拖出让位平移——平移在复制关掉时照旧。
    """
    canvas = _CopyCanvas(scale=3.0)
    canvas.width = lambda: 100
    canvas.height = lambda: 100
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 1, (
        "复制开着、按在图上拖动没把文件拖出去")


def test_canvas_pans_on_the_image_when_copy_drag_out_is_off(monkeypatch):
    """复制**关掉**时，按住图内拖动仍是平移画布（原行为不被破坏）。"""
    canvas = _CopyCanvas(scale=1.0)
    # `_CopyCanvas` 的 editor 默认是「复制开着」，这里显式关掉
    canvas._editor._canvas_image_copy_enabled = False
    canvas.mousePressEvent(_PressEvent(50, 50))

    assert canvas.is_dragging_background is True, (
        "不放大时按下图内就该能拖动画布，现在没进入拖动状态")
    canvas.mouseMoveEvent(_MoveEvent(70, 60))
    assert (canvas.background_offset.x(), canvas.background_offset.y()) != (0, 0), (
        "拖动后画布偏移没变，用户会感觉「拖不动」")


def test_canvas_panning_fitted_image_does_not_also_drag_the_file_out(monkeypatch):
    """复制**关掉**时拖动图内只平移，不会拖出文件。"""
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)
    canvas._editor._canvas_image_copy_enabled = False

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 0, "复制关掉时不该触发拖出"
    assert canvas.is_dragging_background is True


def test_copy_drag_out_still_works_from_empty_area_outside_the_image(monkeypatch):
    """落在图**外**的空白画布上按下再拖，仍然能拖出文件（保留原功能）。"""
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    # 图占的是 (0,0,100,100)，这里点在图外
    canvas.mousePressEvent(_PressEvent(400, 400))
    canvas.mouseMoveEvent(_MoveEvent(460, 460))

    assert canvas._drag_out_calls == 1, "图外空白处拖动应当还能拖出文件"


def test_copy_drag_out_wins_over_panning_when_press_is_on_the_image(monkeypatch):
    """① 复制开着、按在**图上**拖动 → 拖出文件（用户报的那个 bug）。

    用户原话「开启时拖动到桌面松开没有复制一份」：按在图上会先被判成平移
    （`is_dragging_background`），拖出永远挂不起来。复制开着时让拖出优先，
    平移在复制关掉时照旧（见上一条）。
    """
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    # 图占 (0,0,100,100)：点在图内
    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 1, (
        "复制开着、按在图上拖动没把文件拖出去")


def test_copy_drag_out_press_outside_the_image_still_works(monkeypatch):
    """① 的边界：点在图**外**的空白处拖动也照旧拖得出去。"""
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    canvas.mousePressEvent(_PressEvent(400, 400))
    canvas.mouseMoveEvent(_MoveEvent(440, 440))

    assert canvas._drag_out_calls == 1, (
        "图外空白处拖动应当还能拖出文件")


def test_copy_drag_out_does_not_hijack_a_press_on_a_label_box(monkeypatch):
    """① 的边界：按在检测框上是编辑标注，不能变成拖出文件。

    判据不能只看「按在图上」——框也在图上。抓到东西（框/贴图/区域）时
    必须让位给编辑，否则用户一点框就把图拖出去了。
    """
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)
    canvas._editor.detection_boxes = [
        {"label": "cat", "x": 20, "y": 20, "width": 60, "height": 60}]

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    # 关键断言：按在东西上不能变成拖出文件。框本身是否进入拖动由别的用例钉
    # （这里用的是贴图模式的替身，框点击不走那条路），所以只断言不拖出。
    assert canvas._drag_out_calls == 0, (
        "按在检测框上不该拖出文件（那是编辑标注）")


def test_copy_drag_out_from_blank_canvas_outside_the_image(monkeypatch):
    """① 的边界：点在图**外**的空白画布上拖动，也能拖出文件。

    老行为允许这条（图外空白不算「图内」，平移在那也没意义）。新判据把
    「拖出优先」限定在按到图上，所以图外空白走的是原来那条兜底分支——这条
    用例就是钉住那条分支还在。
    """
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    # 图占 (0,0,100,100)，点在图外
    canvas.mousePressEvent(_PressEvent(400, 400))
    canvas.mouseMoveEvent(_MoveEvent(440, 440))

    assert canvas._drag_out_calls == 1, "图外空白处拖动应当还能拖出文件"


def test_press_outside_the_image_does_not_count_as_on_the_image(monkeypatch):
    """① 的边界：`_press_is_on_background()` 必须是**真的几何判断**。

    它区分「按在图上」与「按在图外空白」：前者才算「用户要拖这张图」，后者走
    原来那条兜底分支。两种情况下 `_drag_out_pending` 都为真，所以只断言拖出
    发生抓不到差别——这里直接问这个判据本身，`return True` 的退化会被打回。
    """
    canvas = _CopyCanvas(scale=1.0)

    # 图占 (0,0,100,100)
    assert canvas._press_is_on_background(_PressEvent(50, 50)._pos) is True, (
        "图内按下应判为「在图上」")
    assert canvas._press_is_on_background(_PressEvent(400, 400)._pos) is False, (
        "图外空白按下不该判为「在图上」——否则「按在图上才让位给拖出」这条"
        "规则会退化成「按哪都算图上」")


def test_copy_drag_out_wins_when_the_image_is_not_zoomed(monkeypatch):
    """① 复制开着、图**没缩放**：按在图上拖动 = 拖出文件。

    用户要的手感：「直接左键拖出图片，在图片没有缩放时」。
    """
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)
    canvas.is_manual_scale = False

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 1, (
        "未缩放时按在图上拖动没把文件拖出去")


def test_zoomed_image_pans_instead_of_dragging_out(monkeypatch):
    """① 复制开着但图**已缩放**：拖动 = 平移画布，不拖出。

    用户原话「如果缩放了只能平移」——放大看细节时不该弹出文件。
    """
    canvas = _CopyCanvas(scale=3.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)
    canvas.is_manual_scale = True

    canvas.mousePressEvent(_PressEvent(50, 50))

    assert canvas.is_dragging_background is True, (
        "已缩放时按下应当进入平移（用户要的正是这个）")
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 0, (
        "已缩放时不该拖出文件")
    assert (canvas.background_offset.x(), canvas.background_offset.y()) != (0, 0), (
        "已缩放时拖动没平移画布，用户会感觉「拖不动」")
