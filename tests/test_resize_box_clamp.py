"""回归测试：检测框角柄 resize 必须夹紧到图像边界。

历史问题：br/tr/bl 角柄只做 `max(10, ...)`，不 clamp 到背景图尺寸，
拖动时框会超出画面底部（坐标可远大于图像高度）。
"""
from labelforge.canvas.canvas_drawing import CanvasDrawingMixin


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def __sub__(self, other):
        return Point(self._x - other.x(), self._y - other.y())


class Rect:
    def __init__(self, x=0, y=0, w=100, h=100):
        self._x, self._y, self._w, self._h = x, y, w, h

    def left(self):
        return self._x

    def top(self):
        return self._y

    def right(self):
        return self._x + self._w

    def bottom(self):
        return self._y + self._h

    def contains(self, point):
        return True


class Background:
    def __init__(self, w=400, h=300):
        self._w = w
        self._h = h

    def width(self):
        return self._w

    def height(self):
        return self._h


class Editor:
    def __init__(self):
        self._is_delete_view = False
        self.background_images = ["a.png"]
        self.current_background = Background(400, 300)
        self.current_background_index = 0
        self.detection_boxes = []
        self.detection_boxes_dict = {}
        self.selected_item = None
        self.global_labels = set()
        self.background_dataset_labels = set()

    def save_undo_state(self):
        pass

    def update_label_list(self):
        pass

    def get_label_color(self, _label):
        return "#00FF00"


class Canvas(CanvasDrawingMixin):
    def __init__(self):
        self._editor = Editor()
        self.background_scale = 1
        self.selected_box = None
        self.selected_boxes = []
        self.box_resize_start = None
        self.resize_handle = None
        self.mouse_pos = None
        self._needs_save = False
        self.updated = 0

    def get_background_rect(self):
        return Rect(0, 0, 400, 300)

    def _sync_detection_box_to_dict(self, index):
        pass

    def update(self):
        self.updated += 1


def _canvas_with_box(handle):
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "x": 10, "y": 10, "width": 50, "height": 40,
        "label": "a", "shape_type": "rectangle", "group_id": None,
    }]
    canvas.selected_box = 0
    canvas.selected_boxes = [0]
    canvas.is_resizing_box = True
    canvas.resize_handle = handle
    canvas.box_resize_start = Point(0, 0)
    return canvas


def test_br_handle_clamps_to_image_bottom_right():
    canvas = _canvas_with_box("br")
    canvas.mouse_pos = Point(100000, 100000)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    assert box["x"] + box["width"] <= 400
    assert box["y"] + box["height"] <= 300
    assert box["width"] == 390
    assert box["height"] == 290


def test_bl_handle_clamps_to_image_bottom():
    canvas = _canvas_with_box("bl")
    canvas.mouse_pos = Point(-100000, 100000)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    assert box["y"] + box["height"] <= 300
    assert box["height"] == 290


def test_tr_handle_clamps_to_image_right():
    canvas = _canvas_with_box("tr")
    canvas.mouse_pos = Point(100000, -100000)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    assert box["x"] + box["width"] <= 400
    assert box["width"] == 390


def test_tl_handle_stays_within_image():
    canvas = _canvas_with_box("tl")
    canvas.mouse_pos = Point(-100000, -100000)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    assert box["x"] >= 0
    assert box["y"] >= 0
    assert box["x"] + box["width"] <= 400
    assert box["y"] + box["height"] <= 300


def test_br_handle_still_respects_minimum_size():
    canvas = _canvas_with_box("br")
    canvas.mouse_pos = Point(-100000, -100000)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    assert box["width"] >= 10
    assert box["height"] >= 10
