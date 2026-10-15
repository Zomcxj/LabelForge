"""画布悬停状态回归测试。"""
from PyQt5.QtCore import Qt

from pastelabel.canvas import canvas_interaction
from pastelabel.canvas import canvas as canvas_module
from pastelabel.canvas.canvas import Canvas as RealCanvas
from pastelabel.canvas.canvas_interaction import CanvasInteractionMixin
from pastelabel.ui.main_window import ImageEditor


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

    def _region_at(self, point):
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
        __import__("sys").modules, "PyQt5.QtGui",
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


def test_canvas_copy_drag_out_does_not_fire_while_panning(monkeypatch):
    """图放大到超出画布时，按住拖是**平移画布**，不能顺手拖出文件。"""
    pan = _CopyCanvas(scale=3.0)
    pan.width = lambda: 100
    pan.height = lambda: 100
    canvas = pan
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    canvas.mousePressEvent(_PressEvent(50, 50))
    assert canvas.is_dragging_background is True, "放大后按下应进入平移"
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 0, "平移画布时不该把文件拖出去"


def test_canvas_pans_even_when_the_image_fits(monkeypatch):
    """不放大（图完整显示）时，按住图拖动也应该能挪动画布位置。

    用户报的「在不放大画布下拖动失效」：`_handle_background_click` 原来只在
    「图比画布大」时才进入背景拖动，图装得下就直接 return——于是按住拖
    什么都不会动。但 `background_offset` 在非手动缩放下**照样参与渲染**，
    也就是说这个平移本来就有意义，只是被那个门槛挡住了。
    """
    canvas = _CopyCanvas(scale=1.0)      # 图 400x300 装进 800x800
    canvas.mousePressEvent(_PressEvent(50, 50))

    assert canvas.is_dragging_background is True, (
        "不放大时按下图内就该能拖动画布，现在没进入拖动状态")
    canvas.mouseMoveEvent(_MoveEvent(70, 60))
    assert (canvas.background_offset.x(), canvas.background_offset.y()) != (0, 0), (
        "拖动后画布偏移没变，用户会感觉「拖不动」")


def test_canvas_panning_fitted_image_does_not_also_drag_the_file_out(monkeypatch):
    """拖动装得下的图是平移画布，不能同时把它当文件拖出去。"""
    canvas = _CopyCanvas(scale=1.0)
    _enable_copy_drag_out(monkeypatch, canvas)
    canvas._editor.geometry = lambda: Rect(0, 0, 4000, 4000)
    canvas._editor.mapFromGlobal = lambda p: Point(1, 1)

    canvas.mousePressEvent(_PressEvent(50, 50))
    canvas.mouseMoveEvent(_MoveEvent(90, 90))

    assert canvas._drag_out_calls == 0, "平移时不该触发拖出复制"
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
