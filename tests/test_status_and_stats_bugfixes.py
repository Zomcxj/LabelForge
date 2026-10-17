"""三个状态栏/统计 bug 的回归测试。

1. 滚轮缩放框、调整单边框时状态栏宽高不刷新（无鼠标移动事件）。
2. 旋转框选中时状态栏不显示角度。
3. 刚加载数据集时统计弹窗「框类型」为空，切图后才有（扫描缓存缺 tasks）。
"""
import json

from PySide6.QtCore import Qt

from labelforge.canvas import canvas_interaction
from labelforge.canvas.canvas import Canvas, _rotation_angle_status_text
from labelforge.engine.image_loader import ImageLoaderMixin, scan_dataset_full


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self): return self._x
    def y(self): return self._y


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
    def __init__(self):
        self.edit_mode = "annotate"
        self._is_delete_view = False
        self.background_images = ["work.png"]
        self.current_background = Background()
        self.current_background_index = 0
        self.canvas_items = []
        self.detection_boxes = [
            {"label": "cat", "x": 20, "y": 10, "width": 20, "height": 10},
        ]
        self.detection_boxes_dict = {0: [self.detection_boxes[0].copy()]}
        self.selected_item = None
        self.show_labels_checkbox = type("Check", (), {"isChecked": lambda self: True})()

    def save_undo_state(self):
        pass


class CanvasStub(canvas_interaction.CanvasInteractionMixin):
    def __init__(self):
        self._editor = Editor()
        self.background_scale = 1
        self.mouse_pos = Point(41, 15)
        self.selected_box = 0
        self.selected_boxes = []
        self.selected_item_size = None
        self.updated = 0
        self.status_updates = 0
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.synced_indexes = []
        self.is_drawing_box = False
        self.draw_start_pos = None
        self.temp_draw_box = None
        self.box_drag_start = Point(0, 0)
        self.is_dragging_box = False
        self.is_resizing_box = False
        self.is_rotating_box = False

    def setCursor(self, *a):
        pass

    def update_status_label(self):
        self.status_updates += 1

    def get_background_rect(self):
        return Rect(0, 0, 100, 100)

    def update(self):
        self.updated += 1

    def _sync_detection_box_to_dict(self, index):
        self.synced_indexes.append(index)

    def _can_edit_canvas(self):
        return not getattr(self._editor, '_is_delete_view', False)

    def _current_modifiers(self):
        return 0

    def find_item_at_position(self, point):
        return None


def test_wheel_edge_adjustment_refreshes_status_label(monkeypatch):
    """滚轮调整单边框后必须刷新状态栏宽高（回归：只有 update() 没有文字刷新）。"""
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG, 'edge_step', 4)
    canvas = CanvasStub()

    canvas.wheelEvent(WheelEvent(-120))

    box = canvas._editor.detection_boxes[0]
    assert box['width'] == 16
    assert canvas.status_updates >= 1, "滚轮调整后状态栏宽高未刷新"


def test_wheel_box_scale_refreshes_status_label(monkeypatch):
    """滚轮缩放整框后同样必须刷新状态栏宽高。"""
    monkeypatch.setattr(canvas_interaction, "QRectF", Rect)
    monkeypatch.setitem(canvas_interaction.DETECTION_BOX_WHEEL_CONFIG,
                        'detection_box_scale_step', 0.1)
    canvas = CanvasStub()
    canvas.mouse_pos = Point(30, 15)

    canvas.wheelEvent(WheelEvent(120))

    box = canvas._editor.detection_boxes[0]
    assert round(box['width'], 1) == 22.0
    assert canvas.status_updates >= 1, "滚轮缩放后状态栏宽高未刷新"


def test_rotation_angle_status_text_formats_obb_angle():
    box = {"shape_type": "rotation",
           "points": [[0, 0], [100, 0], [100, 50], [0, 50]]}
    assert _rotation_angle_status_text([box], 0) == "A:0°"

    rotated = {"shape_type": "rotation",
               "points": [[0, 0], [0, 100], [-50, 100], [-50, 0]]}
    assert _rotation_angle_status_text([rotated], 0) == "A:90°"


def test_rotation_angle_status_text_skips_non_rotation():
    rect = {"shape_type": "rectangle", "points": [[0, 0], [1, 1]]}
    assert _rotation_angle_status_text([rect], 0) == ""
    assert _rotation_angle_status_text([], 0) == ""
    assert _rotation_angle_status_text([rect], None) == ""


class _Label:
    def __init__(self):
        self.text = None

    def setText(self, value):
        self.text = value


class _StatusEditor:
    _max_labels = 3
    _is_delete_view = False

    def __init__(self, boxes):
        self.current_background = object()
        self.status_label = _Label()
        self.detection_boxes = boxes

    def get_image_info(self):
        return {"width": 500, "height": 800,
                "box_count": len(self.detection_boxes), "paste_count": 0}

    def get_label_stats(self):
        return {}


class _StatusStub:
    def __init__(self, boxes, selected_box, mouse_inside=True):
        self._editor = _StatusEditor(boxes)
        self.mouse_inside = mouse_inside
        self.mouse_pos = Point(10, 20)
        self.background_scale = 1.0
        self.selected_item_size = None
        self.selected_box = selected_box
        self.is_drawing_box = False
        self.temp_draw_box = None

    def get_background_rect(self):
        return Rect(0, 0, 100, 100)


def test_status_label_shows_rotation_angle_for_selected_obb():
    """选中旋转框时状态栏必须出现角度（接线验证，非纯函数）。"""
    boxes = [{"label": "car", "x": -50, "y": 0, "width": 50, "height": 100,
              "shape_type": "rotation",
              "points": [[0, 0], [0, 100], [-50, 100], [-50, 0]]}]
    stub = _StatusStub(boxes, 0)

    Canvas.update_status_label(stub)

    assert "A:90°" in stub._editor.status_label.text


def test_status_label_hides_angle_for_plain_rectangle():
    boxes = [{"label": "cat", "x": 0, "y": 0, "width": 10, "height": 10}]
    stub = _StatusStub(boxes, 0)

    Canvas.update_status_label(stub)

    assert "A:" not in stub._editor.status_label.text


def test_status_label_shows_angle_when_mouse_outside_canvas():
    """鼠标移出画布后状态栏仍应保留选中旋转框的角度。"""
    boxes = [{"label": "car", "x": -50, "y": 0, "width": 50, "height": 100,
              "shape_type": "rotation",
              "points": [[0, 0], [0, 100], [-50, 100], [-50, 0]]}]
    stub = _StatusStub(boxes, 0, mouse_inside=False)

    Canvas.update_status_label(stub)

    assert "A:90°" in stub._editor.status_label.text


def test_scan_dataset_full_collects_task_types(tmp_path):
    """单趟扫描同时产出框类型，避免统计弹窗再扫一遍。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    (tmp_path / "a.json").write_text(json.dumps({"shapes": [
        {"label": "cat", "shape_type": "rectangle"},
        {"label": "cat", "shape_type": "polygon"},
        {"label": "dog", "shape_type": "rotation"},
    ]}), encoding="utf-8")

    tasks = {}
    labels, counts, statuses = scan_dataset_full([str(img)], tasks_out=tasks)

    assert labels == {"cat", "dog"}
    assert counts == {"cat": 2, "dog": 1}
    assert tasks == {"cat": {"det", "seg"}, "dog": {"obb"}}


def test_scan_dataset_full_without_tasks_out_unchanged(tmp_path):
    """不传 tasks_out 时行为保持不变（向后兼容）。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    (tmp_path / "a.json").write_text(json.dumps({"shapes": [
        {"label": "cat", "shape_type": "polygon"},
    ]}), encoding="utf-8")

    labels, counts, statuses = scan_dataset_full([str(img)])
    assert labels == {"cat"}
    assert counts == {"cat": 1}


def test_apply_dataset_labels_stores_task_types_in_stats_cache():
    """扫描结果写入缓存时必须带上 tasks，否则统计弹窗框类型为空。"""
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 1
    editor.background_images = ["a.png"]
    editor.global_labels = set()
    editor.background_dataset_labels = set()
    editor.label_color_map = {}
    editor.get_label_color = lambda label: "#000000"
    editor.update_label_list = lambda: None
    editor._processing_panel = None
    editor._memory_background_path = ""
    editor._cached_bg_label_stats_path = ""
    editor._dataset_stats_dirty = True

    editor._apply_dataset_labels(
        1, ("a.png",), {"cat"}, {"cat": 2}, None, {"cat": {"det", "seg"}})

    assert editor._cached_bg_label_stats == [
        {"label": "cat", "count": 2, "color": "#000000",
         "tasks": ["det", "seg"]},
    ]


def test_apply_dataset_labels_without_tasks_defaults_to_empty():
    """旧调用方不传 tasks 时缓存字段仍存在（向后兼容）。"""
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 1
    editor.background_images = ["a.png"]
    editor.global_labels = set()
    editor.background_dataset_labels = set()
    editor.label_color_map = {}
    editor.get_label_color = lambda label: "#000000"
    editor.update_label_list = lambda: None
    editor._processing_panel = None
    editor._memory_background_path = ""
    editor._cached_bg_label_stats_path = ""

    editor._apply_dataset_labels(1, ("a.png",), {"cat"}, {"cat": 1})

    assert editor._cached_bg_label_stats == [
        {"label": "cat", "count": 1, "color": "#000000", "tasks": []},
    ]


def test_worker_signal_carries_tasks():
    """扫描 worker 必须把 tasks 一起发给主线程。"""
    import inspect
    from labelforge.engine.image_loader.scan import DatasetLabelScanWorker

    sig = DatasetLabelScanWorker.labels_scanned
    assert sig is not None
    source = inspect.getsource(DatasetLabelScanWorker.run)
    assert "tasks_out=tasks" in source
    assert "statuses, tasks" in source
