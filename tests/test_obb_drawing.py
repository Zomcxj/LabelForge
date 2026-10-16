"""OBB (rotation) geometry, drawing and export (no Qt window)."""
import math

from labelforge.engine.shape_io import (
    rect_to_rotation_points,
    rotate_points,
    rotation_center,
    rotation_handle_point,
    rotation_points_bbox,
    move_rotation_vertex,
    yolo_obb_line,
    box_from_labelme_shape,
    labelme_shape_from_box,
    rotation_angle_degrees,
    box_display_label,
)
from tests.test_polygon_drawing import Point, Rect, Editor, Canvas


def test_rect_to_rotation_points_is_ccw_corners():
    pts = rect_to_rotation_points(10, 20, 30, 40)
    assert pts == [[10, 20], [40, 20], [40, 60], [10, 60]]


def test_rotation_center_is_diagonal_midpoint():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert rotation_center(pts) == (5, 5)


def test_rotate_points_90_degrees_around_center():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    rotated = rotate_points(pts, (5, 5), math.pi / 2)
    expected = [[10, 0], [10, 10], [0, 10], [0, 0]]
    for got, exp in zip(rotated, expected):
        assert abs(got[0] - exp[0]) < 1e-6
        assert abs(got[1] - exp[1]) < 1e-6


def test_rotation_handle_point_sits_outside_edge():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    hx, hy = rotation_handle_point(pts, 20)
    assert hx == 5
    assert hy == -20


def test_rotation_points_bbox_matches_axis_aligned_rect():
    pts = [[10, 20], [40, 20], [40, 60], [10, 60]]
    assert rotation_points_bbox(pts) == (10, 20, 30, 40)


def test_move_rotation_vertex_keeps_rectangle_shape():
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    moved = move_rotation_vertex(pts, 0, (-5, -5))
    # 对角点不动
    assert moved[2] == [10, 10]
    # 相邻两点仍在过对角的边上，角度保持直角
    v1 = (moved[1][0] - moved[0][0], moved[1][1] - moved[0][1])
    v2 = (moved[3][0] - moved[0][0], moved[3][1] - moved[0][1])
    assert abs(v1[0] * v2[0] + v1[1] * v2[1]) < 1e-6


def test_obb_roundtrips_through_labelme():
    box = {
        "label": "car", "shape_type": "rotation", "group_id": 3,
        "x": 0, "y": 0, "width": 10, "height": 10,
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }
    shape = labelme_shape_from_box(box)
    assert shape["shape_type"] == "rotation"
    loaded = box_from_labelme_shape(shape)
    assert loaded["shape_type"] == "rotation"
    assert loaded["points"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert loaded["group_id"] == 3


def test_yolo_obb_line_normalizes_eight_coords():
    box = {"points": [[0, 0], [50, 0], [50, 100], [0, 100]]}
    line = yolo_obb_line(box, 2, 100, 200)
    assert line == "2 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000 0.000000 0.500000"


def test_yolo_obb_line_rejects_non_four_points():
    assert yolo_obb_line({"points": [[0, 0], [1, 1]]}, 0, 10, 10) is None


def test_rotation_draw_mode_creates_rotation_shape(monkeypatch):
    from labelforge.canvas import canvas_drawing as cd
    from labelforge.ui import dialogs as dialogs_mod

    class RectF:
        def __init__(self, x, y, w, h):
            self._x, self._y, self._w, self._h = x, y, w, h

        def left(self):
            return self._x

        def top(self):
            return self._y

        def right(self):
            return self._x + self._w

        def bottom(self):
            return self._y + self._h

        def width(self):
            return self._w

        def height(self):
            return self._h

    monkeypatch.setattr(cd, "QRectF", RectF)
    monkeypatch.setattr(
        dialogs_mod.LabelSelectionDialog, "select_label",
        staticmethod(lambda *a, **k: ("car", None)),
    )
    canvas = Canvas()
    canvas.current_draw_mode = 'rotation'
    canvas.draw_start_pos = Point(10, 10)
    canvas.temp_draw_box = Rect(10, 10, 30, 40)
    canvas._editor.detection_boxes = []

    canvas._complete_drawing(Point(40, 50))

    box = canvas._editor.detection_boxes[0]
    assert box["shape_type"] == "rotation"
    assert box["points"] == [[10.0, 10.0], [40.0, 10.0], [40.0, 50.0], [10.0, 50.0]]


def test_drag_rotation_vertex_preserves_rectangle():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "car", "shape_type": "rotation",
        "x": 0, "y": 0, "width": 10, "height": 10,
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }]
    canvas.selected_box = 0
    canvas.resize_handle = "v2"
    canvas.box_resize_start = Point(10, 10)
    canvas.mouse_pos = Point(15, 15)
    canvas._resize_box()
    box = canvas._editor.detection_boxes[0]
    # 对角点不动
    assert box["points"][0] == [0, 0]
    # 保持矩形：相邻两边垂直
    v1 = (box["points"][1][0] - box["points"][2][0], box["points"][1][1] - box["points"][2][1])
    v2 = (box["points"][3][0] - box["points"][2][0], box["points"][3][1] - box["points"][2][1])
    assert abs(v1[0] * v2[0] + v1[1] * v2[1]) < 1e-6
    # bbox 更新
    assert box["x"] == 0 and box["y"] == 0
    assert box["width"] == 15 and box["height"] == 15


def test_rotate_selected_box_rotates_around_center():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "car", "shape_type": "rotation",
        "x": 0, "y": 0, "width": 10, "height": 10,
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }]
    canvas.selected_box = 0
    canvas.rotation_center = (5, 5)
    canvas.rotation_prev_angle = 0.0
    canvas.mouse_pos = Point(5, 15)  # 从右侧转到正下方 => 90°
    canvas._rotate_selected_box()
    box = canvas._editor.detection_boxes[0]
    assert box["points"][0][0] == 10 and abs(box["points"][0][1]) < 1e-6


def test_rotation_box_has_four_corner_handles():
    canvas = Canvas()
    canvas._editor.detection_boxes = [{
        "label": "car", "shape_type": "rotation",
        "x": 0, "y": 0, "width": 10, "height": 10,
        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }]
    assert canvas._box_handle_at_pos(Point(0, 0), 0) == "v0"
    assert canvas._box_handle_at_pos(Point(10, 10), 0) == "v2"


def test_obb_exporter_writes_eight_coordinates(tmp_path):
    from labelforge.engine.yolo_exporter import YoloExporter

    out = tmp_path / "out"
    exporter = YoloExporter(str(out), mode="obb")
    exporter._ensure_dirs()
    classes = ["car"]
    exporter._write_classes_file(classes)
    exporter._write_one({
        "stem": "img1", "width": 100, "height": 100,
        "boxes": [{
            "label": "car", "shape_type": "rotation",
            "points": [[0, 0], [50, 0], [50, 50], [0, 50]],
        }],
    }, classes)
    text = (out / "labels" / "img1.txt").read_text().strip()
    assert text == "0 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000 0.000000 0.500000"


def test_rotation_angle_zero_for_axis_aligned():
    pts = rect_to_rotation_points(0, 0, 10, 10)
    assert rotation_angle_degrees(pts) == 0.0


def test_rotation_angle_90_for_vertical_edge():
    pts = [[0, 0], [0, 10], [10, 10], [10, 0]]
    assert abs(abs(rotation_angle_degrees(pts)) - 90) < 1e-6


def test_box_display_label_appends_angle_for_rotation():
    box = {"label": "car", "shape_type": "rotation", "points": rect_to_rotation_points(0, 0, 10, 10)}
    assert box_display_label(box) == "car 0°"


def test_box_display_label_plain_for_rectangle():
    box = {"label": "car", "shape_type": "rectangle"}
    assert box_display_label(box) == "car"


def test_keyboard_rotate_uses_configured_step(monkeypatch):
    import math
    from labelforge.core.config import OBB_CONFIG
    from labelforge.engine.event_handler import EventHandlerMixin

    class Editor(EventHandlerMixin):
        def __init__(self):
            self.detection_boxes = [{
                "label": "car", "shape_type": "rotation",
                "x": 0, "y": 0, "width": 10, "height": 10,
                "points": rect_to_rotation_points(0, 0, 10, 10),
            }]
            self.detection_boxes_dict = {0: self.detection_boxes}
            self.current_background_index = 0
            self.undos = 0

        def save_undo_state(self):
            self.undos += 1

        def update_label_list(self):
            pass

    class CanvasStub:
        def __init__(self):
            self.selected_box = 0
            self.updates = 0

        def update(self):
            self.updates += 1

    monkeypatch.setitem(OBB_CONFIG, 'rotate_step', 10)
    editor = Editor()
    editor.canvas = CanvasStub()
    editor.rotate_selected_box(1)
    pts = editor.detection_boxes[0]["points"]
    assert abs(rotation_angle_degrees(pts) - 10) < 1e-6
    editor.rotate_selected_box(-1)
    back = rotation_angle_degrees(editor.detection_boxes[0]["points"])
    assert min(abs(back), abs(back - 360)) < 1e-6


def test_keyboard_rotate_ignores_non_obb():
    from labelforge.engine.event_handler import EventHandlerMixin

    class Editor(EventHandlerMixin):
        def __init__(self):
            self.detection_boxes = [{"label": "car", "x": 0, "y": 0, "width": 10, "height": 10}]
            self.detection_boxes_dict = {0: self.detection_boxes}
            self.current_background_index = 0

        def save_undo_state(self):
            raise AssertionError("should not rotate HBB")

        def update_label_list(self):
            pass

    class CanvasStub:
        selected_box = 0
        def update(self):
            pass

    editor = Editor()
    editor.canvas = CanvasStub()
    editor.rotate_selected_box(1)
    assert "points" not in editor.detection_boxes[0]
