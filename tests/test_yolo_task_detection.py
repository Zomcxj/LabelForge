"""YOLO 导出任务类型自动判定的测试。

任务类型（det/seg/obb/pose）不再由界面下拉框指定，而是按标注自身的
``shape_type`` 判定。判定错了会静默丢信息——多边形降级成外接框、旋转框
丢掉角度、关键点整组消失——所以每条判定规则都要有用例钉住。
"""
import pytest

from labelforge.engine.yolo_exporter import (
    MODE_TO_TASK,
    detect_task,
    detect_task_for_items,
)
from labelforge.engine.yolo_exporter import YoloExporter


def _box(shape_type=None, group_id=None):
    box = {"label": "x", "x": 0, "y": 0, "width": 10, "height": 10}
    if shape_type is not None:
        box["shape_type"] = shape_type
    if group_id is not None:
        box["group_id"] = group_id
    return box


# ---------- detect_task：单张图内的判定 ----------

def test_detect_task_defaults_to_det_for_plain_rectangles():
    assert detect_task([_box(), _box("rectangle")]) == "det"


def test_detect_task_picks_seg_when_polygon_present():
    assert detect_task([_box(), _box("polygon")]) == "seg"


def test_detect_task_picks_obb_when_rotation_present():
    """旋转框必须用 OBB 写，降级成 det 会丢掉角度。"""
    assert detect_task([_box(), _box("rotation")]) == "obb"


def test_detect_task_picks_pose_when_point_has_group():
    assert detect_task([_box(group_id=1), _box("point", group_id=1)]) == "pose"


def test_detect_task_ignores_ungrouped_points():
    """没有 group_id 的点无法与框配对，不该判定成 pose。"""
    assert detect_task([_box("point")]) == "det"


def test_detect_task_prefers_rotation_over_polygon():
    assert detect_task([_box("polygon"), _box("rotation")]) == "obb"


def test_detect_task_prefers_pose_over_polygon():
    """人体标注常是「矩形 + 点」，笔画框偶尔被标成多边形，先判 pose 才不丢关键点。"""
    boxes = [_box("polygon"), _box("point", group_id=2)]
    assert detect_task(boxes) == "pose"


def test_detect_task_handles_empty_and_missing_shape_type():
    assert detect_task([]) == "det"
    assert detect_task([{"label": "x"}]) == "det"


# ---------- detect_task_for_items：整批判定 ----------

def test_detect_task_for_items_scans_all_entries():
    entries = [
        {"boxes": [_box("rectangle")]},
        {"boxes": [_box("polygon")]},
    ]
    assert detect_task_for_items(entries) == "seg"


def test_detect_task_for_items_returns_det_when_all_plain():
    entries = [{"boxes": [_box()]}, {"boxes": []}]
    assert detect_task_for_items(entries) == "det"


# ---------- YoloExporter 的接线 ----------

def _item(stem, boxes, width=200, height=100):
    return {"stem": stem, "boxes": boxes, "width": width, "height": height,
            "image": None, "img_path": ""}


def _polygon(label="cat"):
    return {"label": label, "shape_type": "polygon",
            "points": [[0, 0], [100, 0], [100, 50]],
            "x": 0, "y": 0, "width": 100, "height": 50}


def _rotation(label="cat"):
    return {"label": label, "shape_type": "rotation",
            "points": [[10, 10], [110, 10], [110, 60], [10, 60]],
            "x": 10, "y": 10, "width": 100, "height": 50}


def test_exporter_auto_detects_seg_from_polygon(tmp_path):
    exp = YoloExporter(str(tmp_path))
    exp.run([], {}, ["cat"], input_data=[_item("a", [_polygon()])])

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    assert len(parts) == 1 + 6, "多边形应写成 3 个归一化点（seg 行格式）"


def test_exporter_auto_detects_obb_from_rotation(tmp_path):
    exp = YoloExporter(str(tmp_path))
    exp.run([], {}, ["cat"], input_data=[_item("a", [_rotation()])])

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    assert len(parts) == 1 + 8, "旋转框应写成 4 个归一化角点（obb 行格式）"


def test_exporter_auto_detects_det_from_plain_boxes(tmp_path):
    exp = YoloExporter(str(tmp_path))
    box = {"label": "cat", "x": 10, "y": 10, "width": 50, "height": 40}
    exp.run([], {}, ["cat"], input_data=[_item("a", [box])])

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    assert len(parts) == 5, "水平框应写成 cls + 4 个归一化值"


def test_explicit_mode_still_overrides_detection(tmp_path):
    """显式传 mode 时以 mode 为准（测试与特殊场景需要）。"""
    exp = YoloExporter(str(tmp_path), mode="seg")
    box = {"label": "cat", "x": 10, "y": 10, "width": 50, "height": 40}
    exp.run([], {}, ["cat"], input_data=[_item("a", [box])])

    # seg 模式下纯矩形没有多边形顶点，行会被跳过
    text = (tmp_path / "labels" / "a.txt").read_text().strip()
    assert text == ""


def test_mode_to_task_covers_all_supported_modes():
    assert MODE_TO_TASK["hbb"] == "det"
    assert MODE_TO_TASK["seg"] == "seg"
    assert MODE_TO_TASK["obb"] == "obb"
    assert MODE_TO_TASK["pose"] == "pose"
