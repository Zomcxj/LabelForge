"""Pose (point keypoint) drawing state machine (no Qt window)."""
from labelforge.canvas.canvas_drawing import CanvasDrawingMixin
from tests.test_polygon_drawing import Point, Rect, Background, Editor, Canvas


def make_canvas():
    return Canvas()


def test_create_point_stores_point_shape_with_group():
    canvas = make_canvas()
    canvas._create_point_at([12.0, 34.0], "nose", group_id=2)
    box = canvas._editor.detection_boxes[0]
    assert box["shape_type"] == "point"
    assert box["points"] == [[12.0, 34.0]]
    assert box["label"] == "nose"
    assert box["group_id"] == 2
    assert box["x"] == 12.0 and box["y"] == 34.0


def test_point_hit_near_point_only():
    from labelforge.canvas.canvas_menu import CanvasMenuMixin

    class Editor:
        detection_boxes = [{
            "label": "nose", "shape_type": "point",
            "points": [[10.0, 10.0]],
            "x": 10.0, "y": 10.0, "width": 1, "height": 1,
        }]

    class HitCanvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()
            self.background_scale = 1

        def get_background_rect(self):
            return Rect()

    class Pos:
        def __init__(self, x, y):
            self._x, self._y = x, y

        def x(self):
            return self._x

        def y(self):
            return self._y

    canvas = HitCanvas()
    assert canvas._find_detection_box_at(Pos(12, 12)) == 0
    assert canvas._find_detection_box_at(Pos(50, 50)) is None


def test_drag_point_moves_point_and_xy():
    canvas = make_canvas()
    canvas._editor.detection_boxes = [{
        "label": "nose", "shape_type": "point",
        "points": [[10.0, 10.0]],
        "x": 10.0, "y": 10.0, "width": 1, "height": 1,
    }]
    canvas.selected_box = 0
    canvas.box_drag_start = Point(20, 20)
    canvas.mouse_pos = Point(25, 22)
    canvas._drag_box()
    box = canvas._editor.detection_boxes[0]
    assert box["points"] == [[15.0, 12.0]]
    assert box["x"] == 15.0 and box["y"] == 12.0


def test_point_box_has_no_corner_handles():
    canvas = make_canvas()
    canvas._editor.detection_boxes = [{
        "label": "nose", "shape_type": "point",
        "points": [[10.0, 10.0]],
        "x": 10.0, "y": 10.0, "width": 1, "height": 1,
    }]
    from tests.test_polygon_drawing import Point
    assert canvas._box_handle_at_pos(Point(10, 10), 0) is None


def test_yolo_pose_line_format():
    from labelforge.engine.shape_io import yolo_pose_line
    line = yolo_pose_line(
        class_id=0, bbox=(50, 25, 100, 50),
        keypoints=[[50, 25], [150, 25]],
        image_width=200, image_height=100,
    )
    parts = line.split()
    assert parts[0] == "0"
    assert [float(p) for p in parts[1:5]] == [0.5, 0.5, 0.5, 0.5]
    # two keypoints, visibility=2
    assert len(parts) == 5 + 6
    assert [float(p) for p in parts[5:7]] == [0.25, 0.25]
    assert parts[7] == "2"
