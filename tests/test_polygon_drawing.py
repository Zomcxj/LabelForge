"""Polygon draw-mode state machine (no Qt window)."""
from pastelabel.canvas.canvas_drawing import CanvasDrawingMixin


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def setX(self, v):
        self._x = int(v)

    def setY(self, v):
        self._y = int(v)

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
        return self._x <= point.x() <= self._x + self._w and self._y <= point.y() <= self._y + self._h


class Background:
    def width(self):
        return 100

    def height(self):
        return 100


class Editor:
    def __init__(self):
        self._is_delete_view = False
        self.background_images = ["a.png"]
        self.current_background = Background()
        self.current_background_index = 0
        self.detection_boxes = []
        self.detection_boxes_dict = {}
        self.selected_item = None
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_list = None
        self.saved = 0
        self.undos = 0

    def save_undo_state(self):
        self.undos += 1

    def update_label_list(self):
        pass

    def get_label_color(self, _label):
        return "#00FF00"


class Canvas(CanvasDrawingMixin):
    def __init__(self):
        self._editor = Editor()
        self.background_scale = 1
        self.is_drawing_box = False
        self.is_drawing_polygon = False
        self.current_draw_mode = None
        self.draw_start_pos = None
        self.temp_draw_box = None
        self.temp_polygon_points = []
        self.selected_box = None
        self.selected_boxes = []
        self.updated = 0
        self.cursor = None
        self.box_drag_start = None
        self.box_resize_start = None
        self.resize_handle = None
        self._needs_save = False

    def _sync_detection_box_to_dict(self, index):
        pass

    def get_background_rect(self):
        return Rect()

    def update_status_label(self):
        pass

    def update(self):
        self.updated += 1

    def setCursor(self, c):
        self.cursor = c

    def _save_current_detection_boxes(self):
        self._editor.saved += 1

    def _sync_all_detection_boxes_to_dict(self):
        idx = self._editor.current_background_index
        self._editor.detection_boxes_dict[idx] = self._editor.detection_boxes.copy()


def test_polygon_clicks_accumulate_image_points():
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    canvas._handle_polygon_press(Point(10, 10))
    canvas._handle_polygon_press(Point(40, 10))
    canvas._handle_polygon_press(Point(40, 40))
    assert canvas.temp_polygon_points == [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]]
    assert canvas._can_close_polygon() is True


def test_polygon_cannot_close_with_two_points():
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    canvas._handle_polygon_press(Point(10, 10))
    canvas._handle_polygon_press(Point(40, 10))
    assert canvas._can_close_polygon() is False


def test_polygon_backspace_pops_last_point_then_cancels():
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    canvas._handle_polygon_press(Point(10, 10))
    canvas._handle_polygon_press(Point(40, 10))
    canvas._polygon_pop_last_point()
    assert canvas.temp_polygon_points == [[10.0, 10.0]]
    canvas._polygon_pop_last_point()
    assert canvas.temp_polygon_points == []
    assert canvas.is_drawing_polygon is False


def test_finish_polygon_pops_duplicate_doubleclick_point():
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    canvas._handle_polygon_press(Point(10, 10))
    canvas._handle_polygon_press(Point(40, 10))
    canvas._handle_polygon_press(Point(40, 40))
    canvas._handle_polygon_press(Point(40, 40))
    canvas._finish_polygon(label="person", pop_duplicate=True)
    assert len(canvas._editor.detection_boxes) == 1
    box = canvas._editor.detection_boxes[0]
    assert box["shape_type"] == "polygon"
    assert box["points"] == [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]]
    assert box["label"] == "person"
    assert canvas.temp_polygon_points == []
    assert canvas.is_drawing_polygon is False


def test_drag_polygon_moves_points_and_bbox():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    canvas.selected_box = 0
    canvas.box_drag_start = Point(20, 20)
    canvas.mouse_pos = Point(25, 22)
    canvas._drag_box()
    box = canvas._editor.detection_boxes[0]
    assert box["x"] == 15.0
    assert box["y"] == 12.0
    assert box["points"] == [[15.0, 12.0], [45.0, 12.0], [45.0, 42.0]]


def test_polygon_vertex_handle_at_pos():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    handle = canvas._box_handle_at_pos(Point(40, 10), 0)
    assert handle == "v1"


def test_drag_polygon_vertex_updates_point_and_bbox():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    canvas.selected_box = 0
    canvas.resize_handle = "v1"
    canvas.box_resize_start = Point(40, 10)
    canvas.mouse_pos = Point(50, 12)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    assert box["points"][1] == [50.0, 12.0]
    assert box["x"] == 10.0
    assert box["width"] == 40.0


def test_delete_polygon_vertex_updates_points_and_bbox():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    assert canvas._delete_polygon_vertex(0, 1) is True
    box = canvas._editor.detection_boxes[0]
    assert box["points"] == [[10.0, 10.0], [40.0, 40.0], [10.0, 40.0]]
    assert box["x"] == 10.0
    assert box["width"] == 30.0
    assert box["height"] == 30.0


def test_delete_polygon_vertex_refuses_when_only_three_points():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    assert canvas._delete_polygon_vertex(0, 1) is False
    assert len(canvas._editor.detection_boxes[0]["points"]) == 3


def test_polygon_press_ignores_points_beyond_max():
    from pastelabel.core.config import DETECTION_BOX_CONFIG
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    old = DETECTION_BOX_CONFIG.get("max_polygon_points")
    DETECTION_BOX_CONFIG["max_polygon_points"] = 3
    try:
        canvas._handle_polygon_press(Point(10, 10))
        canvas._handle_polygon_press(Point(40, 10))
        canvas._handle_polygon_press(Point(40, 40))
        canvas._handle_polygon_press(Point(10, 40))
        assert canvas.temp_polygon_points == [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]]
        assert canvas._polygon_at_max_points() is True
    finally:
        if old is None:
            DETECTION_BOX_CONFIG.pop("max_polygon_points", None)
        else:
            DETECTION_BOX_CONFIG["max_polygon_points"] = old


def test_insert_polygon_vertex_on_edge_updates_points_and_bbox():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    assert canvas._insert_polygon_vertex(0, 1, [55.0, 10.0]) is True
    box = canvas._editor.detection_boxes[0]
    assert box["points"] == [[10.0, 10.0], [55.0, 10.0], [40.0, 10.0],
                             [40.0, 40.0], [10.0, 40.0]]


def test_insert_polygon_vertex_respects_max_points():
    from pastelabel.core.config import DETECTION_BOX_CONFIG
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    old = DETECTION_BOX_CONFIG.get("max_polygon_points")
    DETECTION_BOX_CONFIG["max_polygon_points"] = 3
    try:
        assert canvas._insert_polygon_vertex(0, 1, [55.0, 10.0]) is False
        assert len(canvas._editor.detection_boxes[0]["points"]) == 3
    finally:
        if old is None:
            DETECTION_BOX_CONFIG.pop("max_polygon_points", None)
        else:
            DETECTION_BOX_CONFIG["max_polygon_points"] = old


def test_delete_selected_routes_to_vertex_when_hovering_vertex():
    from pastelabel.engine.event_handler import EventHandlerMixin

    class Canvas:
        def __init__(self):
            self.hover_resize_handle = "v1"
            self.selected_box = 0
            self.selected_boxes = [0]
            self._editor = None

        def update(self):
            pass

        _delete_polygon_vertex = CanvasDrawingMixin._delete_polygon_vertex
        _can_edit_canvas = CanvasDrawingMixin._can_edit_canvas
        _sync_polygon_bbox = CanvasDrawingMixin._sync_polygon_bbox

        def _sync_detection_box_to_dict(self, index):
            pass

        def _save_current_detection_boxes(self):
            self._editor.saved += 1

    class Editor(EventHandlerMixin):
        def __init__(self):
            self._is_delete_view = False
            self.canvas = Canvas()
            self.canvas._editor = self
            self.background_scale = 1
            self.detection_boxes = [{
                "label": "person", "shape_type": "polygon",
                "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
                "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
            }]
            self.detection_boxes_dict = {}
            self.current_background_index = -1
            self.saved = 0

        def update_label_list(self):
            pass

    editor = Editor()
    editor._delete_selected_box()
    box = editor.detection_boxes[0]
    assert len(box["points"]) == 3
    assert box["points"] == [[10.0, 10.0], [40.0, 40.0], [10.0, 40.0]]
    assert editor.saved == 1


def test_delete_selected_removes_box_when_not_hovering_vertex():
    from pastelabel.engine.event_handler import EventHandlerMixin

    class Canvas:
        def __init__(self):
            self.hover_resize_handle = None
            self.selected_box = 0
            self.selected_boxes = [0]

        def update(self):
            pass

    class Editor(EventHandlerMixin):
        def __init__(self):
            self._is_delete_view = False
            self.canvas = Canvas()
            self.detection_boxes = [{
                "label": "person", "shape_type": "polygon",
                "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
                "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
            }]
            self.detection_boxes_dict = {}
            self.current_background_index = -1
            self.current_background = None
            self.background_images = []
            self.selected_item = None
            self.canvas_items = []
            self.deleted = False

        def save_undo_state(self):
            self.deleted = True

        def update_label_list(self):
            pass

    editor = Editor()
    editor._delete_selected_box()
    assert editor.detection_boxes == []


def test_scale_selected_box_scales_polygon_points_proportionally():
    from pastelabel.canvas.canvas_interaction import CanvasInteractionMixin

    class Delta:
        def y(self):
            return 120

    class Ev:
        def angleDelta(self):
            return Delta()

    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "person", "shape_type": "polygon",
        "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
        "x": 10.0, "y": 10.0, "width": 30.0, "height": 30.0,
    }]
    canvas.selected_box = 0
    canvas._scale_selected_box = CanvasInteractionMixin._scale_selected_box.__get__(canvas)
    canvas._scale_selected_box(Ev())
    box = canvas._editor.detection_boxes[0]
    factor = 1.05
    cx = cy = 25.0
    expected = [[cx + (px - cx) * factor, cy + (py - cy) * factor]
                for px, py in [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]]]
    assert box["points"] == expected
    assert box["x"] == 9.25
    assert box["width"] == 31.5


def test_finish_polygon_stores_group_id():
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    canvas._handle_polygon_press(Point(10, 10))
    canvas._handle_polygon_press(Point(40, 10))
    canvas._handle_polygon_press(Point(40, 40))
    canvas._finish_polygon(label="person", group_id=3)
    box = canvas._editor.detection_boxes[0]
    assert box["group_id"] == 3
    assert box["label"] == "person"


def test_create_detection_box_with_group():
    canvas = Canvas()
    canvas._create_detection_box(1, 2, 30, 30, "person", group_id=5)
    box = canvas._editor.detection_boxes[0]
    assert box["group_id"] == 5


def test_finish_polygon_without_label_discards():
    canvas = Canvas()
    canvas.is_drawing_polygon = True
    canvas._handle_polygon_press(Point(10, 10))
    canvas._handle_polygon_press(Point(40, 10))
    canvas._handle_polygon_press(Point(40, 40))
    canvas._finish_polygon(label=None)
    assert canvas._editor.detection_boxes == []
    assert canvas.is_drawing_polygon is False
